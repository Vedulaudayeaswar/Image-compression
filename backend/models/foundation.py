from __future__ import annotations

from dataclasses import dataclass
import torch

from backend.config import settings
from backend.models.demo_vae import DemoVAE
from backend.models.learned_vae import load_learned_vae


@dataclass
class FoundationStatus:
    available: bool
    loaded: bool
    model_id: str
    detail: str


class ModelRegistry:
    def __init__(self) -> None:
        self.demo_vae = DemoVAE().eval()
        self.learned_vae = None
        self.learned_vae_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "models" / "autoencoder"
        self.pipeline = None
        self._failure: str | None = None
        self._empty_prompt_embeddings = None

    def device(self, preferred: str = "auto") -> torch.device:
        if preferred == "gpu" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() and preferred != "cpu" else "cpu")

    def status(self) -> FoundationStatus:
        if self.pipeline is not None:
            return FoundationStatus(True, True, settings.model_id, f"Diffusers VAE and U-Net loaded from {settings.model_id}.")
        if self._failure:
            return FoundationStatus(False, False, settings.model_id, self._failure)
        return FoundationStatus(False, False, settings.model_id, "Not loaded. Enable model download or install local Diffusers weights for Research Mode.")

    def load_foundation(self, device: torch.device) -> None:
        if self.pipeline is not None:
            return
        try:
            from diffusers import StableDiffusionPipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                settings.model_id,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                local_files_only=not settings.allow_model_download,
                safety_checker=None,
                token=settings.hf_token,
            ).to(device)
            self.pipeline.set_progress_bar_config(disable=True)
        except Exception as error:  # research mode must fail clearly rather than substitute a model
            self._failure = f"Foundation model unavailable: {error}"
            raise RuntimeError(self._failure) from error

    @torch.inference_mode()
    def empty_prompt_embeddings(self, device: torch.device) -> torch.Tensor:
        """Return the real SD2.1 CLIP embedding required by its cross-attention U-Net."""
        if self._empty_prompt_embeddings is not None:
            return self._empty_prompt_embeddings.to(device)
        if self.pipeline is None:
            raise RuntimeError("Foundation pipeline is not loaded.")
        tokens = self.pipeline.tokenizer(
            [""], padding="max_length", max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        self._empty_prompt_embeddings = self.pipeline.text_encoder(tokens)[0].detach()
        return self._empty_prompt_embeddings

    @torch.inference_mode()
    def encode(self, image: torch.Tensor, mode: str, device: torch.device) -> tuple[torch.Tensor, str]:
        if mode == "research":
            self.load_foundation(device)
            assert self.pipeline is not None
            latent = self.pipeline.vae.encode(image.to(device)).latent_dist.mean * self.pipeline.vae.config.scaling_factor
            return latent, f"Foundation VAE ({settings.model_id})"
        if self.learned_vae is None:
            self.learned_vae = load_learned_vae(self.learned_vae_path, device)
        if self.learned_vae is not None:
            return self.learned_vae.encode(image.to(device)), "LearnedConvAutoencoder (LC3, 8-channel latent)"
        return self.demo_vae.encode(image.cpu()), "LegacyDemoVAE (fallback; train LC2 for quality)"

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor, mode: str, device: torch.device) -> torch.Tensor:
        if mode == "research":
            assert self.pipeline is not None
            return self.pipeline.vae.decode(latent.to(device) / self.pipeline.vae.config.scaling_factor).sample
        if self.learned_vae is None:
            self.learned_vae = load_learned_vae(self.learned_vae_path, device)
        if self.learned_vae is not None:
            return self.learned_vae.decode(latent.to(device))
        return self.demo_vae.decode(latent.cpu())


registry = ModelRegistry()
