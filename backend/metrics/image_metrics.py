from __future__ import annotations

import math
import os
import torch
import torch.nn.functional as F


def mse_psnr(reference: torch.Tensor, reconstruction: torch.Tensor) -> tuple[float, float]:
    mse = float(F.mse_loss(reference, reconstruction).item())
    return mse, 99.0 if mse == 0 else 10 * math.log10(4.0 / mse)  # tensors are in [-1, 1]


def ms_ssim_proxy(reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """A computed multi-scale structural similarity approximation without hidden dependencies."""
    scores = []
    x, y = reference, reconstruction
    for _ in range(3):
        ux, uy = x.mean((-1, -2), keepdim=True), y.mean((-1, -2), keepdim=True)
        vx = ((x - ux) ** 2).mean((-1, -2), keepdim=True); vy = ((y - uy) ** 2).mean((-1, -2), keepdim=True)
        covariance = ((x - ux) * (y - uy)).mean((-1, -2), keepdim=True)
        score = ((2 * ux * uy + 0.0001) * (2 * covariance + 0.0009)) / ((ux ** 2 + uy ** 2 + 0.0001) * (vx + vy + 0.0009))
        scores.append(score.mean())
        if min(x.shape[-2:]) < 4: break
        x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    return float(torch.stack(scores).clamp(0, 1).mean().item())


_lpips_model = None


def lpips_distance(reference: torch.Tensor, reconstruction: torch.Tensor) -> float | None:
    """Compute LPIPS when its explicitly requested dependency/checkpoint is available."""
    global _lpips_model
    # LPIPS depends on pretrained AlexNet weights. Never make an inference request
    # unexpectedly download them or block a CPU decode; enable only after setup.
    if os.getenv("DIFFUSIONCOMPRESS_ENABLE_LPIPS", "0") != "1":
        return None
    try:
        import lpips
        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net="alex").eval()
        with torch.inference_mode():
            return float(_lpips_model(reference, reconstruction).mean().item())
    except Exception:
        # Do not substitute a metric and label it LPIPS.
        return None
