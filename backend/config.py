from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPERIMENT_DIR = DATA_DIR / "experiments"
ARTIFACT_DIR = DATA_DIR / "artifacts"
for directory in (DATA_DIR, EXPERIMENT_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = 15 * 1024 * 1024
    # SD-Turbo is public and substantially more capable than the tiny smoke-test
    # checkpoint, while remaining usable with the existing Diffusers pipeline.
    # Override these with environment variables when using a gated/air-gapped model.
    model_id: str = os.getenv("DIFFUSIONCOMPRESS_MODEL_ID", "stabilityai/sd-turbo")
    allow_model_download: bool = os.getenv("DIFFUSIONCOMPRESS_ALLOW_MODEL_DOWNLOAD", "1") == "1"
    hf_token: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


settings = Settings()
