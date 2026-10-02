from __future__ import annotations

from html import escape
from typing import Any


def render_html(experiment: dict[str, Any]) -> str:
    """Self-contained HTML report built strictly from saved experiment results."""
    result, config = experiment["result"], experiment["config"]
    metrics, compression, pipeline, images = result["metrics"], result["compression"], result["pipeline"], result["images"]
    def number(value: float, digits: int = 4) -> str: return f"{value:.{digits}f}"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{escape(experiment['id'])} report</title>
    <style>body{{font:14px system-ui;background:#101516;color:#e7eeee;max-width:1100px;margin:auto;padding:32px}}h1{{color:#c8ff69}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #35484a;padding:8px;text-align:left}}img{{max-width:48%;background:#080b0d;margin:4px}}.note{{border-left:3px solid #c8ff69;padding:10px;background:#172523}}</style></head><body>
    <h1>DiffusionCompress research report</h1><p>Experiment <b>{escape(experiment['id'])}</b> · {escape(experiment['created_at'])}</p>
    <p class='note'>{escape(result['notice'])}</p><h2>Configuration</h2><pre>{escape(str(config))}</pre>
    <h2>Compression and reconstruction</h2><table><tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Original bytes</td><td>{compression['original_bytes']}</td></tr><tr><td>Compressed bytes</td><td>{compression['compressed_bytes']}</td></tr><tr><td>Actual bpp</td><td>{number(compression['bpp'])}</td></tr><tr><td>Compression ratio</td><td>{number(compression['ratio'],2)}×</td></tr><tr><td>PSNR</td><td>{number(metrics['psnr'],2)} dB</td></tr><tr><td>MS-SSIM</td><td>{number(metrics['ms_ssim'])}</td></tr><tr><td>LPIPS</td><td>Not implemented</td></tr></table>
    <h2>Architecture / latent statistics</h2><pre>{escape(str({'vae': pipeline['vae'], 'denoiser': pipeline['denoiser'], 'latent_shape': pipeline['latent_shape'], 'gamma': pipeline['gamma'], 'steps': pipeline['actual_steps'], 'timings': pipeline['timings_seconds']}))}</pre>
    <h2>Images</h2><img src='{images['original']}'><img src='{images['reconstructed']}'><h3>Difference maps</h3><img src='{images['difference_original_pre']}'><img src='{images['difference_original_final']}'><img src='{images['diffusion_contribution']}'>
    <p class='note'>{escape(result['scientific_note'])}</p></body></html>"""
