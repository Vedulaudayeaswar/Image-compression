import torch
from backend.entropy.model import FactorizedEntropyModel


def test_factorized_entropy_model_estimates_positive_differentiable_rate():
    model = FactorizedEntropyModel()
    values = torch.randn(2, 4, 3, 3, requires_grad=True)
    bits = model.estimate_bits(values)
    assert bits.item() > 0
    bits.backward()
    assert values.grad is not None and model.log_scale.grad is not None
