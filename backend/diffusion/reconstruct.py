from __future__ import annotations

import os
import torch

from backend.models.foundation import registry
from backend.diffusion.demo_denoiser import DemoDenoiser


demo_denoiser = DemoDenoiser().eval()


@torch.inference_mode()
def denoise(latent: torch.Tensor, mode: str, steps: int, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor], str]:
    """Denoise from inverse-quantized y_hat_t, never from independent Gaussian noise."""
    steps = max(1, min(int(steps), 30))
    states: list[torch.Tensor] = []
    current = latent
    if mode == "demo":
        for _ in range(steps):
            current = demo_denoiser(current)
            states.append(current.detach().cpu())
        return current, states, "Fixed convolutional DemoDenoiser (not a foundation diffusion prior)"
    # A generic text-to-image U-Net is not a valid inverse for a compressed
    # image latent. Running it here can replace the image with a hallucinated
    # constant, especially on CPU and without a prompt. The default research
    # path therefore keeps the transmitted latent unchanged and uses the
    # pretrained SD VAE as a fast, faithful decoder. Opt in only for
    # experimentation with DIFFUSIONCOMPRESS_ENABLE_REFINEMENT=1.
    if os.getenv("DIFFUSIONCOMPRESS_ENABLE_REFINEMENT", "0") != "1":
        current = latent.to(device)
        return current, [current.detach().cpu()], "SD-Turbo VAE latent reconstruction (U-Net refinement disabled for fidelity)"
    registry.load_foundation(device)
    pipe = registry.pipeline
    assert pipe is not None
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=device)
    current = latent.to(device)
    text_embeddings = registry.empty_prompt_embeddings(device)
    # Start at the transmitted inverse-quantized latent. This is an experimental
    # DDIM-style latent refinement, not a claim of paper-checkpoint reproduction.
    for timestep in scheduler.timesteps:
        model_input = scheduler.scale_model_input(current, timestep)
        noise = pipe.unet(model_input, timestep, encoder_hidden_states=text_embeddings).sample
        current = scheduler.step(noise, timestep, current).prev_sample
        states.append(current.detach().cpu())
    return current, states, f"Diffusers U-Net scheduler refinement ({registry.status().model_id})"
