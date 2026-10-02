# Implementation deviations and status

## Foundation model in Demo Mode

**Paper:** Stable Diffusion v2.1 provides the frozen VAE and latent diffusion backbone.

**Our implementation:** Demo Mode uses a deterministic 4-channel, 8× down/up sampling neural codec and a separately trained lightweight residual latent-refinement network.

**Reason:** This makes the full encode / `.dfc` / independent decode pipeline executable on CPU without silently downloading a multi-gigabyte checkpoint.

**Expected effect:** Demo outputs are not comparable with paper metrics and do not have a learned generative prior. The refiner learns codec-specific residual corrections rather than diffusion sampling. The UI labels this explicitly. Research Mode refuses to substitute the demo codec; it requires actual Stable Diffusion v2.1 weights.

## Lightweight Learned Latent Refinement

**Paper:** Uses a foundation diffusion prior to generate plausible detail after quantization.

**Our implementation:** A 2.52M-parameter 3×3-convolution residual network maps the actual inverse-quantized latent `y_q` to `y_q + alpha * delta_y` before VAE decoding. The decoder receives only `.dfc` plus the shared checkpoint, never an original image. It is trained and validated on disjoint BSDS300 splits at six codec quantization strengths.

**Expected effect:** It can reduce quantization error but cannot restore detail already removed by the DemoVAE’s 8× pooling. It must not be called Stable Diffusion reconstruction or exact recovery.

## Research checkpoint access

**Paper:** Uses Stable Diffusion v2.1 foundation weights.

**Our implementation:** The loader is wired to `stabilityai/stable-diffusion-2-1-base`, but the repository is gated. This environment received HTTP 401 because no Hugging Face token/license authorization was available.

**Reason:** Credentials and acceptance of a third-party model license cannot be invented or bypassed by the application.

**Expected effect:** Research Mode remains correctly unavailable until `HF_TOKEN` is supplied. It never silently substitutes SD 1.5, SD-Turbo, or the DemoVAE.

## Tiny public smoke-test checkpoint

**Paper:** Stable Diffusion v2.1.

**Our implementation:** `diffusers/tiny-stable-diffusion-torch` can be selected explicitly through `DIFFUSIONCOMPRESS_MODEL_ID` and was downloaded and executed successfully on CPU for a 32×32 research-path smoke compression.

**Reason:** It is public and small enough to validate the real Diffusers VAE/U-Net integration without a gated token.

**Expected effect:** It is a test checkpoint, not SD2.1, and its output must not be used for paper comparisons or reported research results.

## Parameter estimator and entropy model checkpoints

**Paper:** Compression-specific P_phi and Entroformer are trained on the stated training setup.

**Our implementation:** P_phi has the specified convolutional 9-output form but begins untrained; entropy coding is a working static arithmetic-coding baseline rather than Entroformer.

**Reason:** No supplied trained checkpoints or training dataset are in the repository.

**Expected effect:** The pipeline, serialization and metrics are real, but rate-distortion quality is a baseline rather than a reproduction claim. Training orchestration is marked `Not implemented` instead of simulating curves.
