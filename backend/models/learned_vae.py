"""Small trainable convolutional autoencoder used by the local codec."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F



class LearnedVAE(nn.Module):
    latent_scale = 0.18215

    def __init__(self, base_channels: int = 64, latent_channels: int = 8) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(base_channels, 3, 4, stride=2, padding=1), nn.Tanh(),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image) / self.latent_scale

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent * self.latent_scale).clamp(-1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(image))


def load_learned_vae(root: Path, device: torch.device) -> LearnedVAE | None:
    checkpoint = root / "best_model.pth"
    if not checkpoint.is_file():
        return None
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = LearnedVAE(int(state.get("base_channels", 64)), int(state.get("latent_channels", 8))).to(device).eval()
    try:
        model.load_state_dict(state["model"])
    except RuntimeError:
        # A checkpoint from the previous LC2 architecture is not silently
        # mixed with the deeper architecture; training will replace it.
        return None
    return model
