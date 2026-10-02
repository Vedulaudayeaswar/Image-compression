"""Reproducible command-line training entry point for the latent refiner."""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline import _load_demo_checkpoint, estimator, refiner
from backend.training.runner import DEFAULT_BITRATES, TrainingConfig, TrainingRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training-image directory")
    parser.add_argument("--validation", required=True, help="Separate validation-image directory")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-lpips", action="store_true", help="Set perceptual loss weight to zero")
    args = parser.parse_args()
    _load_demo_checkpoint()
    runner = TrainingRunner(estimator, refiner)
    job_id = runner.start(TrainingConfig(dataset_path=args.train, validation_dataset_path=args.validation, iterations=args.iterations, batch_size=args.batch_size, crop_size=args.crop_size, validation_every=args.validation_every, target_bitrates=DEFAULT_BITRATES, lambda_perceptual=0.0 if args.no_lpips else 0.1, device=args.device))
    while True:
        job = runner.get(job_id)
        assert job is not None
        print(f"{job['status']}: {job['message']}", flush=True)
        if job["status"] in {"completed", "failed"}:
            if job["status"] == "failed": raise SystemExit(job["message"])
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
