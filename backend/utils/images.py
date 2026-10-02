from __future__ import annotations

from io import BytesIO
import base64
from PIL import Image, ImageOps
import numpy as np
import torch


ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}


def read_image(data: bytes, resolution: int | None = None) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(BytesIO(data)); image.verify()
    image = Image.open(BytesIO(data)); image = ImageOps.exif_transpose(image).convert("RGB")
    if resolution:
        image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
    width, height = image.size
    # Models need dimensions divisible by the 8x VAE scale factor.
    width, height = max(8, width - width % 8), max(8, height - height % 8)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return image, tensor * 2 - 1


def tensor_png(tensor: torch.Tensor) -> bytes:
    tensor = ((tensor.detach().cpu().squeeze(0).clamp(-1, 1) + 1) / 2 * 255).byte()
    image = Image.fromarray(tensor.permute(1, 2, 0).numpy(), "RGB")
    out = BytesIO(); image.save(out, format="PNG"); return out.getvalue()


def png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def latent_preview(latent: torch.Tensor, channel: int = 0) -> str:
    field = latent.detach().cpu()[0, channel]
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    image = Image.fromarray((field.numpy() * 255).astype("uint8"), "L").resize((field.shape[1] * 4, field.shape[0] * 4))
    out = BytesIO(); image.save(out, format="PNG"); return png_data_url(out.getvalue())


def difference_preview(first: torch.Tensor, second: torch.Tensor) -> str:
    """Color map of actual absolute RGB difference, normalized only for viewing."""
    field = (first.detach().cpu() - second.detach().cpu()).abs().mean(dim=1)[0]
    normalized = (field - field.min()) / (field.max() - field.min() + 1e-8)
    red = normalized
    green = normalized.sqrt() * 0.45
    blue = (1 - normalized) * 0.18
    colored = torch.stack((red, green, blue), dim=-1).mul(255).byte().numpy()
    out = BytesIO(); Image.fromarray(colored, "RGB").save(out, format="PNG")
    return png_data_url(out.getvalue())
