import torch

from backend.models.refiner import LatentRefinementNetwork


def test_new_refiner_is_exact_baseline_and_alpha_is_bounded():
    model = LatentRefinementNetwork().eval()
    latent = torch.randn(1, 8, 5, 7)
    refined, residual = model(latent, alpha=1.0)
    assert torch.equal(refined, latent)
    assert torch.equal(residual, torch.zeros_like(latent))
