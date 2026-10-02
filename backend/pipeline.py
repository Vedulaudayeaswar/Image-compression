from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import time
import uuid
from io import BytesIO
import zipfile
from typing import Callable, Any

import numpy as np
import torch

from backend.compression.dfc import serialize, deserialize
from backend.compression.quantization import quantize, inverse_quantize
from backend.config import ARTIFACT_DIR, settings
from backend.entropy.range_coder import encode as entropy_encode, decode as entropy_decode
from backend.entropy.model import FactorizedEntropyModel
from backend.experiments.store import save
from backend.metrics.image_metrics import lpips_distance, mse_psnr, ms_ssim_proxy
from backend.models.foundation import registry
from backend.models.parameter_estimator import ParameterEstimator
from backend.models.refiner import RefinerRegistry
from backend.utils.images import read_image, tensor_png, png_data_url, latent_preview, difference_preview


estimator = ParameterEstimator().eval()
entropy_model = FactorizedEntropyModel(channels=8).eval()
refiner = RefinerRegistry(Path(__file__).resolve().parents[1] / "models" / "refiner")
_demo_checkpoint_loaded = False


def _load_demo_checkpoint() -> None:
    """Restore the newest trained Demo checkpoint once per backend process."""
    global _demo_checkpoint_loaded
    if _demo_checkpoint_loaded:
        return
    _demo_checkpoint_loaded = True
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    explicit = __import__("os").getenv("DIFFUSIONCOMPRESS_CHECKPOINT")
    candidates = [Path(explicit)] if explicit else sorted(checkpoint_dir.glob("train-*-iter*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates or not candidates[0].is_file():
        return
    try:
        state = torch.load(candidates[0], map_location="cpu", weights_only=True)
        estimator.load_state_dict(state["parameter_estimator"])
        entropy_model.load_state_dict(state["entropy_model"])
    except Exception:
        # A stale/incompatible checkpoint must not prevent Demo mode from running.
        return


@dataclass
class CompressionConfig:
    mode: str = "demo"
    bitrate: float = 0.30
    lambda_value: float = 10.0
    steps: int = 4
    resolution: int | None = 256
    device: str = "auto"
    reconstruction_mode: str = "baseline"
    refinement_alpha: float = 0.35


def _stage(callback: Callable[[str, int], None] | None, name: str, percent: int) -> None:
    if callback: callback(name, percent)


def _file_url(name: str) -> str: return f"/api/artifacts/{name}"


def zip_container(dfc_bytes: bytes, experiment_id: str) -> bytes:
    """Optional ZIP wrapper; the compression algorithm remains the inner DFC."""
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{experiment_id}.dfc", dfc_bytes)
        archive.writestr("README.txt", "ZIP container for demonstration. The actual codec is the enclosed Diffusion File Compression (.dfc) file.\n")
    return output.getvalue()


def unwrap_zip(blob: bytes) -> bytes:
    with zipfile.ZipFile(BytesIO(blob), "r") as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith(".dfc") and not name.startswith("/") and ".." not in name.replace("\\", "/").split("/")]
        if len(candidates) != 1:
            raise ValueError("ZIP must contain exactly one safe .dfc payload.")
        return archive.read(candidates[0])


def compress(data: bytes, filename: str, config: CompressionConfig, progress: Callable[[str, int], None] | None = None) -> dict[str, Any]:
    if config.mode not in {"demo"}: raise ValueError("Only the local CPU codec is available.")
    if config.reconstruction_mode not in {"baseline", "refinement"}: raise ValueError("reconstruction_mode must be 'baseline' or 'refinement'.")
    if not 0.0 <= config.refinement_alpha <= 1.0: raise ValueError("refinement_alpha must be between 0 and 1.")
    if config.mode == "demo":
        _load_demo_checkpoint()
    started, device = time.perf_counter(), registry.device(config.device)
    _stage(progress, "Preprocessing image", 8)
    original_pil, image = read_image(data, config.resolution if config.mode == "demo" else config.resolution)
    original_shape = list(image.shape)
    _stage(progress, "VAE encoding", 18)
    stage_start = time.perf_counter(); latent, vae_name = registry.encode(image, config.mode, device); encode_seconds = time.perf_counter() - stage_start
    if latent.shape[1] not in {4, 8}: raise RuntimeError("The selected VAE must produce a 4- or 8-channel latent.")
    _stage(progress, "Parameter prediction", 31)
    with torch.inference_mode(): gamma, predicted_t = estimator(latent.cpu(), config.lambda_value)
    predicted_steps = max(1, min(30, int(round(float(torch.sigmoid(predicted_t[0]) * 10))) or config.steps))
    steps = config.steps if config.steps > 0 else predicted_steps
    _stage(progress, "Adaptive quantization", 42)
    # Lower selected bpp deliberately widens quantization bins. P_phi still
    # provides the per-channel affine parameters; this is an explicit inference
    # control until a trained rate-conditioned P_phi checkpoint is supplied.
    quantization_strength = max(0.20, min(3.0, 0.10 / max(config.bitrate, 0.001)))
    symbols, transformed, inverse_latent, error = quantize(latent.cpu(), gamma, quantization_strength)
    with torch.inference_mode(): estimated_entropy_bits = float(entropy_model.estimate_bits(symbols.float()).item())
    _stage(progress, "Arithmetic entropy coding", 55)
    entropy_payload = entropy_encode(symbols.flatten().tolist())
    experiment_id = "EXP-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5].upper()
    metadata = {
        "version": 1, "experiment_id": experiment_id, "latent_shape": list(symbols.shape),
        "gamma": gamma.tolist(), "steps": steps, "lambda": config.lambda_value,
        "mode": config.mode, "image_size": list(original_pil.size), "model": vae_name, "quantization_strength": quantization_strength,
        "entropy": "static arithmetic coder v1", "dtype": "int16", "codec_version": "LC3-8ch" if latent.shape[1] == 8 else "LC2-4ch",
        "reconstruction_mode": config.reconstruction_mode, "refinement_alpha": config.refinement_alpha,
    }
    dfc_bytes = serialize(metadata, entropy_payload)
    _stage(progress, "Inverse quantization", 65)
    # Round-trip from bytes, rather than preserving the in-memory symbols.
    decoded_meta, decoded_payload = deserialize(dfc_bytes)
    decoded_symbols = torch.tensor(entropy_decode(decoded_payload), dtype=torch.int16).reshape(decoded_meta["latent_shape"])
    recovered = inverse_quantize(decoded_symbols, torch.tensor(decoded_meta["gamma"], dtype=torch.float32), float(decoded_meta.get("quantization_strength", 1.0)))
    _stage(progress, "Latent refinement", 73)
    refinement_start = time.perf_counter()
    refined_latent, latent_residual, refinement_info = refiner.refine(recovered, config.refinement_alpha)
    refinement_seconds = time.perf_counter() - refinement_start
    _stage(progress, "VAE decoding", 88)
    baseline_reconstructed = registry.decode(recovered, config.mode, device).cpu()
    refined_reconstructed = registry.decode(refined_latent, config.mode, device).cpu()
    reconstructed = refined_reconstructed if config.reconstruction_mode == "refinement" and refinement_info["available"] else baseline_reconstructed
    _stage(progress, "Metrics", 96)
    mse, psnr = mse_psnr(image.cpu(), reconstructed)
    baseline_mse, baseline_psnr = mse_psnr(image.cpu(), baseline_reconstructed)
    refined_mse, refined_psnr = mse_psnr(image.cpu(), refined_reconstructed)
    latent_mse, latent_psnr = mse_psnr(latent.cpu(), recovered)
    baseline_metrics = {"mse": baseline_mse, "psnr": baseline_psnr, "ms_ssim": ms_ssim_proxy(image.cpu(), baseline_reconstructed), "lpips": lpips_distance(image.cpu(), baseline_reconstructed)}
    refined_metrics = {"mse": refined_mse, "psnr": refined_psnr, "ms_ssim": ms_ssim_proxy(image.cpu(), refined_reconstructed), "lpips": lpips_distance(image.cpu(), refined_reconstructed)}
    result_png = tensor_png(reconstructed)
    dfc_name, image_name = f"{experiment_id}.dfc", f"{experiment_id}.png"
    zip_name = f"{experiment_id}.zip"; zip_bytes = zip_container(dfc_bytes, experiment_id)
    (ARTIFACT_DIR / dfc_name).write_bytes(dfc_bytes); (ARTIFACT_DIR / zip_name).write_bytes(zip_bytes); (ARTIFACT_DIR / image_name).write_bytes(result_png)
    original_bytes, compressed_bytes = len(data), len(dfc_bytes)
    result: dict[str, Any] = {
        "experiment_id": experiment_id, "status": "completed", "mode": config.mode,
        "notice": "Local CPU codec uses a deterministic neural codec plus a separately trained latent refiner. It is a Lightweight Learned Latent Refinement variant, not a diffusion-model implementation.",
        "images": {
            "original": png_data_url(tensor_png(image)), "reconstructed": png_data_url(result_png),
            "baseline_reconstructed": png_data_url(tensor_png(baseline_reconstructed)),
            "refined_reconstructed": png_data_url(tensor_png(refined_reconstructed)),
            "pre_diffusion": png_data_url(tensor_png(baseline_reconstructed)),
            "latent_channels": [latent_preview(latent, channel) for channel in range(4)],
            "transformed_channels": [latent_preview(transformed, channel) for channel in range(4)],
            "quantized_channels": [latent_preview(symbols.float(), channel) for channel in range(4)],
            "inverse_quantized_channels": [latent_preview(recovered, channel) for channel in range(4)],
            "quantization_error_channels": [latent_preview(error, channel) for channel in range(4)],
            "refined_latent_channels": [latent_preview(refined_latent, channel) for channel in range(4)],
            "residual_channels": [latent_preview(latent_residual, channel) for channel in range(4)],
            "diffusion_steps": [],
            "difference_original_pre": difference_preview(image.cpu(), baseline_reconstructed),
            "difference_original_final": difference_preview(image.cpu(), reconstructed),
            "diffusion_contribution": difference_preview(baseline_reconstructed, refined_reconstructed),
        },
        "downloads": {"dfc": _file_url(dfc_name), "zip": _file_url(zip_name), "reconstruction": _file_url(image_name)},
        "compression": {"original_bytes": original_bytes, "compressed_bytes": compressed_bytes, "ratio": original_bytes / max(1, compressed_bytes), "bpp": len(dfc_bytes) * 8 / (image.shape[-1] * image.shape[-2]), "entropy_payload_bytes": len(entropy_payload), "estimated_entropy_bits": estimated_entropy_bits, "estimated_entropy_bpp": estimated_entropy_bits / (image.shape[-1] * image.shape[-2]), "coder": "Static arithmetic coding"},
        "metrics": {"mse": mse, "psnr": psnr, "ms_ssim": ms_ssim_proxy(image.cpu(), reconstructed), "lpips": (refined_metrics if config.reconstruction_mode == "refinement" else baseline_metrics)["lpips"], "fid": None, "latent_mse": latent_mse, "latent_psnr": latent_psnr, "baseline": baseline_metrics, "refined": refined_metrics},
        "pipeline": {"vae": vae_name, "refiner": refinement_info, "reconstruction_mode": config.reconstruction_mode, "refinement_alpha": config.refinement_alpha, "original_shape": original_shape, "latent_shape": list(latent.shape), "dtype": str(latent.dtype), "gamma": gamma.squeeze(0).tolist(), "quantization_strength": quantization_strength, "entropy_model": entropy_model.summary(), "predicted_timestep": predicted_steps, "actual_steps": steps, "latent_residual": {"mean": float(latent_residual.mean()), "max_abs": float(latent_residual.abs().max()), "l2_norm": float(torch.linalg.vector_norm(latent_residual))}, "timings_seconds": {"encode": encode_seconds, "refinement": refinement_seconds, "total": time.perf_counter() - started}},
        "scientific_note": "The decoder receives only the DFC latent and the shared refinement checkpoint. Refinement predicts a plausible latent correction; it cannot recover discarded source information exactly. LPIPS is omitted when its dependency or pretrained metric weights are unavailable.",
    }
    save(experiment_id, filename, asdict(config), result)
    _stage(progress, "Completed", 100)
    return result


def decompress(dfc_data: bytes) -> tuple[bytes, dict[str, Any]]:
    metadata, payload = deserialize(dfc_data)
    mode = metadata.get("mode", "demo")
    device = registry.device("auto")
    symbols = torch.tensor(entropy_decode(payload), dtype=torch.int16).reshape(metadata["latent_shape"])
    latent = inverse_quantize(symbols, torch.tensor(metadata["gamma"], dtype=torch.float32), float(metadata.get("quantization_strength", 1.0)))
    final = latent
    if metadata.get("reconstruction_mode", "baseline") == "refinement":
        final, _, _ = refiner.refine(latent, float(metadata.get("refinement_alpha", 1.0)))
    # Preserve decoding of older DFC files produced with the legacy fixed codec.
    if str(metadata.get("model", "")).startswith(("DemoVAE", "LegacyDemoVAE")):
        decoded = registry.demo_vae.decode(final.cpu())
    else:
        decoded = registry.decode(final, mode, device).cpu()
    return tensor_png(decoded), metadata
