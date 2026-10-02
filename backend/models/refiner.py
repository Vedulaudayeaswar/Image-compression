"""Decoder-side latent refinement trained against this codec's quantization path."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class RefinerConfig:
    channels: int = 128
    latent_channels: int = 8
    residual_blocks: int = 8
    model_version: str = "latent-refiner-v1"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


class LatentRefinementNetwork(nn.Module):
    """A compact residual network mapping the transmitted 4-channel latent to y_r.

    It predicts only a latent correction; it has no image input or image decoder.
    """
    def __init__(self, config: RefinerConfig = RefinerConfig()) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(nn.Conv2d(config.latent_channels, config.channels, 3, padding=1), nn.SiLU())
        self.residual_blocks = nn.Sequential(*(ResidualBlock(config.channels) for _ in range(config.residual_blocks)))
        self.feature_fusion = nn.Sequential(nn.Conv2d(config.channels, config.channels, 3, padding=1), nn.SiLU())
        self.output_projection = nn.Conv2d(config.channels, config.latent_channels, 3, padding=1)
        # An unloaded/new checkpoint is mathematically the exact VAE baseline.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, quantized_latent: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        features = self.feature_fusion(self.residual_blocks(self.input_projection(quantized_latent)))
        residual = self.output_projection(features)
        return quantized_latent + float(alpha) * residual, residual

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class RefinerRegistry:
    """Loads a shared decoder model; its weights never enter a DFC payload."""
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model: LatentRefinementNetwork | None = None
        self.metadata: dict[str, Any] = {}
        self._loaded = False

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "best_model.pth"

    def load(self, force: bool = False) -> None:
        if self._loaded and not force:
            return
        self._loaded = True
        self.model, self.metadata = None, {}
        if not self.checkpoint_path.is_file():
            return
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        config = RefinerConfig(**checkpoint.get("architecture", {}))
        model = LatentRefinementNetwork(config).eval()
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError:
            return
        self.model = model
        self.metadata = checkpoint.get("metadata", {})

    @torch.inference_mode()
    def refine(self, latent: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.load()
        if self.model is None:
            return latent, torch.zeros_like(latent), {"available": False, "name": "No trained latent-refinement checkpoint", "parameter_count": 0}
        started = __import__("time").perf_counter()
        refined, residual = self.model(latent.cpu(), alpha)
        elapsed_ms = (__import__("time").perf_counter() - started) * 1000
        return refined, residual, {
            "available": True,
            "name": "Lightweight Learned Latent Refinement Network",
            "parameter_count": self.model.parameter_count,
            "device": "cpu",
            "inference_ms": elapsed_ms,
            "model_version": self.model.config.model_version,
            "checkpoint": self.checkpoint_path.name,
        }

    def save_metadata(self, value: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
