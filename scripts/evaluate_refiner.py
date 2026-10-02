"""Evaluate VAE baseline and shared latent refiner on held-out images."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.pipeline import CompressionConfig, compress

RATES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--output", default="data/refiner_rate_distortion")
    args = parser.parse_args()
    images = [path for path in sorted(Path(args.dataset).rglob("*")) if path.suffix.lower() in SUFFIXES][:args.limit]
    if not images: raise SystemExit("No evaluation images found.")
    rows: list[dict[str, float | int | None]] = []
    process = psutil.Process()
    for bitrate in RATES:
        values: list[dict] = []
        for path in images:
            result = compress(path.read_bytes(), path.name, CompressionConfig(mode="demo", bitrate=bitrate, resolution=args.resolution, reconstruction_mode="refinement"))
            values.append(result)
        baseline, refined = [item["metrics"]["baseline"] for item in values], [item["metrics"]["refined"] for item in values]
        rows.append({
            "target_bpp": bitrate, "actual_bpp": mean([item["compression"]["bpp"] for item in values]),
            "baseline_psnr": mean([item["psnr"] for item in baseline]), "refined_psnr": mean([item["psnr"] for item in refined]),
            "baseline_ms_ssim": mean([item["ms_ssim"] for item in baseline]), "refined_ms_ssim": mean([item["ms_ssim"] for item in refined]),
            "baseline_lpips": mean([item["lpips"] for item in baseline]), "refined_lpips": mean([item["lpips"] for item in refined]),
            "refiner_ms": mean([item["pipeline"]["refiner"].get("inference_ms") for item in values]),
            "decode_total_seconds": mean([item["pipeline"]["timings_seconds"]["total"] for item in values]),
            "images": len(values), "rss_bytes": process.memory_info().rss,
        })
        print(json.dumps(rows[-1]), flush=True)
    prefix = Path(args.output); prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
