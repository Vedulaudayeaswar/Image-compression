import torch
from backend.compression.quantization import quantize, inverse_quantize


def test_quantization_inverse_is_consistent():
    latent = torch.randn(1, 4, 5, 7)
    gamma = torch.zeros(1, 4, 2)
    symbols, _, restored, _ = quantize(latent, gamma)
    assert symbols.dtype == torch.int16
    assert torch.allclose(restored, inverse_quantize(symbols, gamma))
