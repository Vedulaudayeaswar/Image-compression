"""Training for the decoder-side latent refiner using actual codec quantization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable

from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F

from backend.compression.quantization import quantize
from backend.models.demo_vae import DemoVAE
from backend.models.learned_vae import load_learned_vae
from backend.models.parameter_estimator import ParameterEstimator
from backend.models.refiner import LatentRefinementNetwork, RefinerConfig, RefinerRegistry

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_BITRATES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


@dataclass
class TrainingConfig:
    dataset_path: str
    validation_dataset_path: str | None = None
    iterations: int = 1_000
    batch_size: int = 2
    crop_size: int = 128
    learning_rate: float = 1e-4
    target_bitrates: tuple[float, ...] = DEFAULT_BITRATES
    lambda_latent: float = 1.0
    lambda_pixel: float = 1.0
    lambda_perceptual: float = 0.1
    alpha: float = 1.0
    validation_every: int = 50
    validation_samples: int = 24
    patience: int = 8
    checkpoint_every: int = 50
    channels: int = 128
    residual_blocks: int = 8
    seed: int = 42
    device: str = "auto"
    mode: str = "demo"


class LocalImageDataset:
    def __init__(self, directory: str, crop_size: int) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise ValueError("Dataset path must be an existing image directory.")
        self.files = sorted(path for path in self.directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not self.files:
            raise ValueError("Dataset directory contains no PNG, JPEG, or WEBP files.")
        self.crop_size = crop_size

    def sample(self, generator: random.Random) -> torch.Tensor:
        source = generator.choice(self.files)
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if min(image.size) < self.crop_size:
                factor = self.crop_size / min(image.size)
                image = image.resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS)
            left = generator.randrange(0, image.width - self.crop_size + 1)
            top = generator.randrange(0, image.height - self.crop_size + 1)
            image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        values = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1)
        return values * 2 - 1

    def summary(self) -> dict[str, Any]:
        return {"directory": str(self.directory), "image_count": len(self.files), "crop_size": self.crop_size}


class _LPIPSLoss:
    def __init__(self, weight: float, device: torch.device) -> None:
        self.weight, self.model, self.reason = weight, None, None
        if weight == 0 or os.getenv("DIFFUSIONCOMPRESS_ENABLE_LPIPS", "0") != "1":
            self.reason = "disabled (set DIFFUSIONCOMPRESS_ENABLE_LPIPS=1 after installing pretrained LPIPS weights)"
            return
        try:
            import lpips
            self.model = lpips.LPIPS(net="alex").to(device).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        except Exception as error:
            self.reason = f"unavailable: {error}"

    def __call__(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return reconstructed.new_zeros(()) if self.model is None else self.model(reconstructed, target).mean()


class TrainingRunner:
    """Freeze DemoVAE/P_phi and learn y_q -> y with real quantize() outputs."""
    def __init__(self, estimator: ParameterEstimator, refiner_registry: RefinerRegistry) -> None:
        self.estimator, self.refiner_registry = estimator, refiner_registry
        self.jobs: dict[str, dict[str, Any]] = {}
        self._controls: dict[str, threading.Event] = {}

    def start(self, config: TrainingConfig, callback: Callable[[str], None] | None = None) -> str:
        if config.mode != "demo":
            raise ValueError("The local CPU codec is the only supported training mode.")
        if not config.target_bitrates or any(rate <= 0 for rate in config.target_bitrates):
            raise ValueError("target_bitrates must contain positive values.")
        if not 0 <= config.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        training_set = LocalImageDataset(config.dataset_path, config.crop_size)
        validation_set = LocalImageDataset(config.validation_dataset_path or config.dataset_path, config.crop_size)
        if training_set.directory == validation_set.directory:
            raise ValueError("Training and validation datasets must be separate directories.")
        job_id = "refiner-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.jobs[job_id] = {"id": job_id, "status": "waiting", "progress": 0, "config": asdict(config), "training_dataset": training_set.summary(), "validation_dataset": validation_set.summary(), "history": [], "message": "Queued"}
        self._controls[job_id] = threading.Event()
        threading.Thread(target=self._run, args=(job_id, config, training_set, validation_set, callback), daemon=True).start()
        return job_id

    @staticmethod
    def _strength(bitrate: float) -> float:
        return max(0.20, min(3.0, 0.10 / bitrate))

    def _codec_pair(self, batch: torch.Tensor, bitrate: float, vae: DemoVAE) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            original = vae.encode(batch)
            gamma, _ = self.estimator(original, 10.0)
            _, _, quantized, _ = quantize(original, gamma, self._strength(bitrate))
        return original, quantized

    def _validate(self, model: LatentRefinementNetwork, dataset: LocalImageDataset, config: TrainingConfig, generator: random.Random, vae: DemoVAE, device: torch.device, perceptual: _LPIPSLoss) -> dict[str, float]:
        model.eval(); totals = {"latent": 0.0, "pixel": 0.0, "perceptual": 0.0, "total": 0.0}
        with torch.inference_mode():
            for _ in range(config.validation_samples):
                batch = dataset.sample(generator).unsqueeze(0).to(device)
                target_latent, quantized_latent = self._codec_pair(batch, generator.choice(config.target_bitrates), vae)
                refined, _ = model(quantized_latent, config.alpha); reconstructed = vae.decode(refined)
                latent_loss = F.mse_loss(refined, target_latent); pixel_loss = F.mse_loss(reconstructed, batch); perceptual_loss = perceptual(reconstructed, batch)
                total = config.lambda_latent * latent_loss + config.lambda_pixel * pixel_loss + config.lambda_perceptual * perceptual_loss
                for name, value in (("latent", latent_loss), ("pixel", pixel_loss), ("perceptual", perceptual_loss), ("total", total)):
                    totals[name] += float(value)
        model.train()
        return {name: value / config.validation_samples for name, value in totals.items()}

    def _write_checkpoint(self, model: LatentRefinementNetwork, config: TrainingConfig, iteration: int, validation: dict[str, float] | None, best: bool, perceptual: _LPIPSLoss) -> None:
        root = self.refiner_registry.root; root.mkdir(parents=True, exist_ok=True)
        metadata = {"iteration": iteration, "parameter_count": model.parameter_count, "dataset": config.dataset_path, "validation_dataset": config.validation_dataset_path, "target_bitrates": list(config.target_bitrates), "loss_weights": {"latent": config.lambda_latent, "pixel": config.lambda_pixel, "perceptual": config.lambda_perceptual}, "vae_model": "LearnedConvAutoencoder LC2 (frozen)", "codec_version": "DFC v1 / LC2", "model_version": model.config.model_version, "device": str(next(model.parameters()).device), "perceptual": "LPIPS AlexNet" if perceptual.model else perceptual.reason, "validation": validation}
        blob = {"model": model.state_dict(), "architecture": asdict(model.config), "metadata": metadata}
        torch.save(blob, root / "latest_model.pth")
        if best: torch.save(blob, root / "best_model.pth")
        self.refiner_registry.save_metadata({"architecture": asdict(model.config), "training": asdict(config), "metadata": metadata})

    def _run(self, job_id: str, config: TrainingConfig, training_set: LocalImageDataset, validation_set: LocalImageDataset, callback: Callable[[str], None] | None) -> None:
        job, pause = self.jobs[job_id], self._controls[job_id]
        try:
            torch.manual_seed(config.seed); random_source, validation_source = random.Random(config.seed), random.Random(config.seed + 1)
            device = torch.device("cuda" if config.device in {"auto", "cuda", "gpu"} and torch.cuda.is_available() else "cpu")
            vae = load_learned_vae(Path("models/autoencoder"), device) or DemoVAE().to(device).eval(); vae.eval(); self.estimator.to(device).eval()
            for module in (vae, self.estimator):
                for parameter in module.parameters(): parameter.requires_grad_(False)
            model = LatentRefinementNetwork(RefinerConfig(config.channels, config.residual_blocks, 8)).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
            perceptual = _LPIPSLoss(config.lambda_perceptual, device)
            job.update({"status": "running", "message": "Training latent refinement on actual codec quantization", "device": str(device), "vae": type(vae).__name__, "parameter_count": model.parameter_count, "perceptual": "LPIPS AlexNet" if perceptual.model else perceptual.reason})
            best_loss, misses, latest_validation = float("inf"), 0, None
            for iteration in range(1, config.iterations + 1):
                while pause.is_set(): job.update({"status": "paused", "message": "Paused"}); time.sleep(0.15)
                job["status"] = "running"
                batch = torch.stack([training_set.sample(random_source) for _ in range(config.batch_size)]).to(device); bitrate = random_source.choice(config.target_bitrates)
                target_latent, quantized_latent = self._codec_pair(batch, bitrate, vae)
                refined, _ = model(quantized_latent, config.alpha); reconstructed = vae.decode(refined)
                latent_loss = F.mse_loss(refined, target_latent); pixel_loss = F.mse_loss(reconstructed, batch); perceptual_loss = perceptual(reconstructed, batch)
                total_loss = config.lambda_latent * latent_loss + config.lambda_pixel * pixel_loss + config.lambda_perceptual * perceptual_loss
                optimizer.zero_grad(set_to_none=True); total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                snapshot = {"iteration": iteration, "bitrate": bitrate, "total_loss": float(total_loss.detach()), "latent_loss": float(latent_loss.detach()), "pixel_loss": float(pixel_loss.detach()), "perceptual_loss": float(perceptual_loss.detach()), "learning_rate": optimizer.param_groups[0]["lr"]}
                if iteration == 1 or iteration % 5 == 0:
                    job["history"].append(snapshot); job["history"] = job["history"][-300:]; job.update({"last": snapshot, "progress": round(iteration / config.iterations * 100), "message": f"Iteration {iteration}/{config.iterations}"})
                if iteration % config.validation_every == 0 or iteration == config.iterations:
                    latest_validation = self._validate(model, validation_set, config, validation_source, vae, device, perceptual); scheduler.step(latest_validation["total"])
                    improved = latest_validation["total"] < best_loss; best_loss = min(best_loss, latest_validation["total"]); misses = 0 if improved else misses + 1
                    self._write_checkpoint(model, config, iteration, latest_validation, improved, perceptual); job["validation"] = latest_validation
                    if misses >= config.patience: job.update({"message": f"Early stopping after {iteration} iterations", "progress": 100}); break
                elif iteration % config.checkpoint_every == 0: self._write_checkpoint(model, config, iteration, latest_validation, False, perceptual)
                if callback: callback(job_id)
            self.refiner_registry.load(force=True)
            metrics = {"best_validation_total_loss": best_loss, "last_validation": latest_validation, "parameter_count": model.parameter_count, "checkpoint": "best_model.pth"}
            (self.refiner_registry.root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            job.update({"status": "completed", "progress": 100, "message": "Latent-refinement training completed", "checkpoint": str(self.refiner_registry.checkpoint_path), "metrics": metrics})
        except Exception as error: job.update({"status": "failed", "message": str(error)})
        finally:
            self.estimator.eval()
            if callback: callback(job_id)

    def pause(self, job_id: str) -> None:
        if job_id not in self._controls: raise KeyError(job_id)
        self._controls[job_id].set()
    def resume(self, job_id: str) -> None:
        if job_id not in self._controls: raise KeyError(job_id)
        self._controls[job_id].clear()
    def get(self, job_id: str) -> dict[str, Any] | None: return self.jobs.get(job_id)
