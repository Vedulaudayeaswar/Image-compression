# Requirement-by-requirement implementation status

This is the current honest status against the supplied project brief—not a claim that every requested feature is complete.

| Area | Status | Evidence / limitation |
|---|---|---|
| Input validation, background compression job, FastAPI, SQLite experiment store | Implemented | `backend/main.py`; PNG/JPEG/WEBP MIME check, 15 MB limit, sanitized artifact names. |
| VAE encode/decode | Demo implemented; Research conditional | Demo uses a real small 4-channel tensor codec. Research Mode calls Stable Diffusion v2.1 only and errors if its model cannot load. Current machine has no loaded checkpoint. |
| P_phi architecture | Implemented | 4 latent channels + spatial lambda; convolutional SiLU model predicts 8 gamma values and t. It is trainable; untrained unless a produced checkpoint is kept loaded in the running process. |
| Adaptive quantization/inverse quantization | Implemented | Per-channel affine transform, integer symbols, inverse transform, actual quantization error visualizations. Target bpp changes explicit quantization strength until rate-conditioned training is available. |
| Entropy coding / `.dfc` | Implemented baseline | Trainable factorized logistic rate estimator plus a separately working static arithmetic coder. It is not Entroformer. `.dfc` is checksummed and independently decodable. |
| Latent refinement | Implemented for Demo Mode | A checkpointed 2.52M-parameter residual latent-refinement network runs after inverse quantization and before VAE decoding. It is trained against the real codec quantization path; it is not diffusion. |
| Intermediate visualizations | Implemented baseline | All four latent channels, transformed/quantized/error maps, every returned diffusion state, and computed absolute-difference maps are sourced from the executed pipeline. |
| Single-image metrics | Partial | MSE, PSNR and a computed multi-scale SSIM approximation are available for baseline and refiner. LPIPS is computed only when explicitly enabled with its pretrained metric weights; FID is correctly not offered for one image. |
| Training jobs | Implemented for latent refinement | Separate training/validation local image directories, actual codec quantization at six rates, frozen VAE/P_phi, AdamW, validation, scheduler, early stopping, and `best_model.pth` / `latest_model.pth` metadata are provided. |
| Training, paper, and experiment-history pages | Partial | `/training`, `/paper`, and `/experiments` now exist; history/training values come from the backend. Rate-distortion sweep, synchronized comparison, and reproduction page remain `Not implemented`. |
| Full training dashboard, Vimeo-90k adapter, dataset evaluation/FID | Partial | Local-image Demo training is real; Vimeo-90k, foundation/DDIM-x0 training, and dataset FID require data/checkpoints. |
| ZIP, PDF report, intermediate-file downloads | Partial | HTML research report and `.dfc`/PNG/metrics downloads exist; ZIP/PDF and packaged intermediate downloads do not. |
| Docker | Partial | CPU-oriented backend Dockerfile and compose file exist. CUDA runtime image and GPU orchestration are not supplied. |

See `docs/deviations.md` for each deliberate paper deviation and its expected effect.
