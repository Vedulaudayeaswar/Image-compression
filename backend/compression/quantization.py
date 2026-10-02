from __future__ import annotations

import torch


def gamma_to_affine(gamma: torch.Tensor, strength: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Map unconstrained P_phi output to positive per-channel scales and offsets."""
    if strength <= 0:
        raise ValueError("Quantization strength must be positive.")
    channels = gamma.shape[-2]
    scale = (0.03 + torch.nn.functional.softplus(gamma[..., 0])).reshape(-1, channels, 1, 1) * strength
    offset = gamma[..., 1].reshape(-1, channels, 1, 1)
    return scale, offset


def quantize(latent: torch.Tensor, gamma: torch.Tensor, strength: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale, offset = gamma_to_affine(gamma, strength)
    transformed = latent / scale + offset
    symbols = torch.round(transformed).clamp(-32768, 32767).to(torch.int16)
    restored = inverse_quantize(symbols, gamma, strength)
    return symbols, transformed, restored, latent - restored


def inverse_quantize(symbols: torch.Tensor, gamma: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    scale, offset = gamma_to_affine(gamma, strength)
    return (symbols.float() - offset) * scale
