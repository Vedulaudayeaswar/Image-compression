from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DemoVAE(nn.Module):
    """Small deterministic 4-channel codec used for the local CPU codec.

    It is deliberately not presented as Stable Diffusion's VAE or a paper-equivalent
    model. It provides a real tensor encode/decode path when foundation weights are
    unavailable locally.
    """

    latent_scale = 0.18215

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(image, kernel_size=8, stride=8)
        luminance = 0.299 * pooled[:, 0:1] + 0.587 * pooled[:, 1:2] + 0.114 * pooled[:, 2:3]
        return torch.cat((pooled, luminance), dim=1) / self.latent_scale

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        rgb = latent[:, :3] * self.latent_scale
        return F.interpolate(rgb, scale_factor=8, mode="bilinear", align_corners=False).clamp(-1, 1)
