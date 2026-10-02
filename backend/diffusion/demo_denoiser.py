from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DemoDenoiser(nn.Module):
    """A fixed convolutional denoiser for the explicitly reduced Demo Mode."""
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16
        self.register_buffer("kernel", kernel.reshape(1, 1, 3, 3))

    @torch.inference_mode()
    def forward(self, latent: torch.Tensor, strength: float = 0.12) -> torch.Tensor:
        blur = F.conv2d(latent, self.kernel.expand(latent.shape[1], 1, 3, 3), padding=1, groups=latent.shape[1])
        return latent.lerp(blur, strength)
