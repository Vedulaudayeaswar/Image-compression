from __future__ import annotations

import math
import torch
from torch import nn


class FactorizedEntropyModel(nn.Module):
    """Replaceable, trainable factorized logistic entropy-model baseline.

    It supplies a differentiable rate estimate for P_phi/entropy training. The
    serialized payload is still encoded by the separate static arithmetic coder,
    whose actual size is reported to the user.
    """

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def likelihood(self, values: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp().clamp(0.03, 100.0)
        upper = torch.sigmoid((values + 0.5 - self.mean) / scale)
        lower = torch.sigmoid((values - 0.5 - self.mean) / scale)
        return (upper - lower).clamp_min(1e-9)

    def estimate_bits(self, values: torch.Tensor) -> torch.Tensor:
        return -torch.log2(self.likelihood(values)).sum()

    def summary(self) -> dict[str, list[float]]:
        return {"mean": self.mean.detach().flatten().tolist(), "scale": self.log_scale.detach().exp().flatten().tolist(), "family": "factorized logistic baseline"}
