from __future__ import annotations

import torch
from torch import nn


class ParameterEstimator(nn.Module):
    """P_phi(y, lambda): 4-channel latent + spatial lambda -> 8 gamma values + t."""

    def __init__(self, in_channels: int = 9, latent_channels: int = 8) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        channels = (in_channels, 32, 64, 128, 192)
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(channels, channels[1:])):
            layers.extend((nn.Conv2d(source, target, 3, stride=2 if index % 2 == 0 else 1, padding=1), nn.SiLU()))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], latent_channels * 2 + 1))
        # A stable, conservative starting point before optional training.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, latent: torch.Tensor, lambda_value: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = latent.shape
        lam = torch.as_tensor(lambda_value, dtype=latent.dtype, device=latent.device).reshape(-1, 1, 1, 1)
        if lam.shape[0] == 1 and batch != 1:
            lam = lam.expand(batch, 1, height, width)
        else:
            lam = lam.expand(batch, 1, height, width)
        values = self.head(self.features(torch.cat((latent, lam), dim=1)))
        gamma = values[:, :self.latent_channels * 2].reshape(batch, self.latent_channels, 2)
        timestep = values[:, self.latent_channels * 2]
        return gamma, timestep
