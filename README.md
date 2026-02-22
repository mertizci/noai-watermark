<p align="center">
<img src="https://raw.githubusercontent.com/mertizci/noai-watermark/refs/heads/main/example/screenshot.gif" width="560" />
</p>


# noai-watermark

<a href="https://www.paypal.com/donate/?hosted_button_id=8BKTHWAHUPWPG">
<img src="https://img.shields.io/badge/Donate-PayPal-blue.svg?logo=paypal" alt="Donate via PayPal" />
</a>

**Remove invisible watermarks and manage AI image metadata.**

AI image generators (Google Gemini, DALL-E, Midjourney, Stable Diffusion, etc.) embed invisible markers into every image they produce. These markers come in two forms:

- **Invisible watermarks** — signals hidden directly in the pixel data (e.g. [SynthID](https://deepmind.google/technologies/synthid/), StableSignature, TreeRing). They survive file format conversions, screenshots, and basic editing. Standard image editors cannot see or remove them.
- **AI metadata** — text fields stored alongside the image (EXIF tags, PNG text chunks, [C2PA](https://c2pa.org/) provenance manifests). They record the model, prompt, seed, and generation parameters.

**noai-watermark** removes both. It uses diffusion-based image regeneration — encoding the image into latent space, injecting noise to break watermark patterns, and reconstructing via reverse diffusion — so the output is visually faithful but no longer carries the hidden signal. All AI metadata is automatically stripped from the output as well.

The controllable regeneration approach is based on [Liu et al. (arXiv:2410.05470)](https://arxiv.org/abs/2410.05470) and the [CtrlRegen](https://github.com/yepengliu/CtrlRegen) repository.

---

## Table of Contents

1. [Example](#example)
2. [Ethics and Responsible Use](#ethics-and-responsible-use)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Pipeline Profiles](#pipeline-profiles)
6. [CLI Reference](#cli-reference)
7. [Python API](#python-api)
8. [Watermark Removal Guide](#watermark-removal-guide)
9. [Verification](#verification)
10. [AI Metadata Types](#ai-metadata-types)
11. [Troubleshooting](#troubleshooting)
12. [Project Structure](#project-structure)
13. [Testing](#testing)
14. [Acknowledgements](#acknowledgements)

---

## Example

Default settings (`--strength 0.04 --steps 50`) — watermark removed, image visually unchanged:

<table>
<tr>
<th>Source (SynthID watermarked)</th>
<th>Cleaned (watermark removed)</th>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/mertizci/noai-watermark/refs/heads/main/example/source.png" width="384" /></td>
<td><img src="https://raw.githubusercontent.com/mertizci/noai-watermark/refs/heads/main/example/cleaned.png" width="384" /></td>
</tr>
</table>

**SynthID verification result on cleaned image:**

> *"Based on a SynthID analysis, this image was not made with Google AI. However, it is not possible to determine if it was generated or edited using other AI tools."*

```bash
noai-watermark source.png --strength 0.04 --steps 50 -o cleaned.png
```

---

## Ethics and Responsible Use

### Why this tool exists

Invisible watermarks like SynthID, StableSignature, and TreeRing are being positioned as the backbone of AI content detection. Companies and platforms present them as robust, reliable proof of AI origin. But how robust are they really?

A single img2img pass at low strength is enough to fool SynthID in most cases. If these systems are supposed to underpin trust and content authenticity on the internet, the public needs to know how fragile they actually are — not just researchers behind closed doors.

This project exists to make that fragility visible. If watermark-based detection can be defeated by a few lines of open-source code, it shouldn't be sold as bulletproof. Public scrutiny is how we get to better, more honest solutions.

### Intended use

- **Security research** — stress-testing watermark robustness, measuring false positive/negative rates
- **Defensive analysis** — validating whether your provenance pipeline actually holds up
- **Interoperability testing** — evaluating how watermarks behave across formats, edits, and re-encoding

### What not to do

Don't use this to strip attribution from content that isn't yours. Don't use it to bypass platform policies or misrepresent authorship. Keep original files when running experiments. Comply with applicable laws and terms of service.

---

## Quick Start

```bash
pip install noai-watermark

# Default pipeline (img2img, fast)
noai-watermark source.png -o cleaned.png

# CtrlRegen pipeline (best quality)
noai-watermark source.png --model-profile ctrlregen -o cleaned.png
```

---

## Installation

### From PyPI

```bash
pip install noai-watermark
```

Both watermark removal pipelines (default img2img + CtrlRegen) are included out of the box.

> **macOS (Homebrew Python):** If you get `externally-managed-environment` error, use `pipx` or a virtual environment:
>
> ```bash
> # Option 1: pipx (recommended for CLI tools)
> brew install pipx
> pipx install noai-watermark
>
> # Option 2: virtual environment
> python3 -m venv ~/.noai-venv
> source ~/.noai-venv/bin/activate
> pip install noai-watermark
> ```

### From GitHub

```bash
pip install "git+https://github.com/mertizci/noai-watermark.git"
```

### Local Development

```bash
git clone https://github.com/mertizci/noai-watermark.git
cd noai-watermark
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- `pillow >= 10.0.0`, `piexif >= 1.1.3`, `torch >= 2.0.0`, `diffusers >= 0.25.0`, `transformers >= 4.35.0`, `accelerate >= 0.25.0`, `controlnet-aux`, `color-matcher`, `safetensors`
- Supported formats: PNG, JPEG

### System Requirements

|  | Default pipeline | CtrlRegen pipeline |
|---|---|---|
| **RAM** | 8 GB minimum | 16 GB recommended |
| **Storage** | ~4 GB (model weights) | ~10 GB (multiple models) |
| **GPU** | Optional | Optional (recommended) |
| **OS** | macOS, Linux, Windows | macOS, Linux, Windows |

No GPU required. The device is selected automatically: **CUDA** (NVIDIA GPU) > **MPS** (Apple Silicon) > **CPU**. If you have a compatible GPU it will be used by default. You can override with `--device cpu`, `--device cuda`, or `--device mps`.

> **Note:** MPS (Apple Silicon) can sometimes be slower than CPU for this workload. If you experience slow performance on Mac, try `--device cpu`.

---

## Pipeline Profiles

Two regeneration pipelines are available. Both use diffusion-based reconstruction — they differ in quality, speed, and download size.

| | `default` | `ctrlregen` |
|---|---|---|
| **Method** | Img2img — adds noise then denoises to reconstruct the image | ControlNet (canny edges) + DINOv2 IP-Adapter (semantic guidance) + histogram color matching |
| **Quality** | Good — may drift on fine details at high strength | Best — preserves structure and color more faithfully |
| **Resolution** | Any size (processed at original resolution) | Any size (tile-based processing, see below) |
| **Speed** | Faster | Slower (multiple models in the pipeline) |
| **Install** | Included | Included |

> **Recommendation:** Both pipelines are installed with `pip install noai-watermark`. Start with `default` for quick iteration. Switch to `ctrlregen` when output quality is the priority.

Download sizes are estimated dynamically from HuggingFace Hub before the first run. Models are cached locally after download — subsequent runs are instant.

### CtrlRegen Tile-Based Processing

The original CtrlRegen pipeline operates on 512×512px images — a hard limitation of SD 1.5-based ControlNet. To support **arbitrary resolution** images without downscaling, noai-watermark uses an automatic tiling strategy:

1. **Split** — the input image is divided into overlapping 512×512 tiles across a grid.
2. **Process** — each tile is independently processed through the CtrlRegen pipeline (canny edge detection → ControlNet → DINOv2 IP-Adapter → diffusion).
3. **Blend** — tiles are merged back using cosine-weighted blending masks that create smooth transitions in the overlap regions, eliminating visible seams.

This means a 2000×2000px photo is processed as a grid of overlapping tiles (e.g. 4×4), each at the native 512px resolution the model was trained on, then seamlessly reassembled at full resolution. No downscaling, no quality loss from resolution mismatch.

<details>
<summary>CtrlRegen model breakdown</summary>

| Model | Role |
|-------|------|
| `SG161222/Realistic_Vision_V4.0_noVAE` | SD 1.5 base model |
| `yepengliu/ctrlregen` | ControlNet + IP-Adapter weights |
| `facebook/dinov2-giant` | DINOv2 image encoder |
| `stabilityai/sd-vae-ft-mse` | High-quality VAE |

</details>

### `--model-profile` vs `--model`

These are two different CLI flags:

- **`--model-profile`** selects the **pipeline architecture** — `default` (simple img2img) or `ctrlregen` (ControlNet + IP-Adapter).
- **`--model`** selects the **base Stable Diffusion checkpoint** used inside that pipeline. Any SD 1.5-compatible HuggingFace model ID works.

Example: `--model-profile default --model runwayml/stable-diffusion-v1-5` uses the simple img2img pipeline with SD v1.5 weights instead of the default DreamShaper.

---

## CLI Reference

All commands use `noai-watermark`. Add `-v` for verbose output.

Watermark removal is the **default mode** — no flag needed. Use `--metadata` to switch to metadata operations.

### Watermark Removal (default)

```bash
# Remove watermark with default settings (strength=0.04, steps=50)
noai-watermark source.png -o cleaned.png

# Force CPU inference (try this if MPS is slow on Mac)
noai-watermark source.png --device cpu -o cleaned.png

# Higher strength for stubborn watermarks
noai-watermark source.png --strength 0.15 --steps 60 -o cleaned.png

# Use a different base model
noai-watermark source.png --model runwayml/stable-diffusion-v1-5 -o cleaned.png

# Photorealistic model (better for real photos)
noai-watermark source.png --model SG161222/Realistic_Vision_V5.1_noVAE -o cleaned.png

# CtrlRegen pipeline (best quality, larger download)
noai-watermark source.png --model-profile ctrlregen -o cleaned.png

# Skip the download confirmation prompt
noai-watermark source.png -y -o cleaned.png

# Authenticate with HuggingFace (or set HF_TOKEN env var)
noai-watermark source.png --hf-token hf_xxxxx -o cleaned.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | overwrites source | Output file path |
| `--strength` | `0.04` | Regeneration intensity (0.0–1.0) |
| `--steps` | `50` | Denoising iterations |
| `--model` | `Lykon/dreamshaper-8` | Any SD 1.5-compatible HuggingFace model |
| `--model-profile` | `default` | Pipeline: `default` or `ctrlregen` |
| `--device` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |
| `--hf-token` | — | HuggingFace API token |
| `-y, --yes` | — | Skip download confirmation |
| `-v, --verbose` | — | Show detailed processing info |

### Metadata Operations

Use `--metadata` to switch to metadata mode. `--check-ai` and `--remove-ai` imply `--metadata` automatically.

```bash
# Clone all metadata from source to target
noai-watermark source.png target.png --metadata -o output.png

# Clone only AI-generated metadata
noai-watermark source.png target.png --metadata --ai-only -o output.png

# Check if an image contains AI metadata
noai-watermark source.png --check-ai

# Remove AI metadata (keeps standard EXIF/XMP)
noai-watermark source.png --remove-ai -o cleaned.png

# Remove all metadata (AI + standard)
noai-watermark source.png --remove-ai --remove-all-metadata -o cleaned.png
```

| Flag | Description |
|------|-------------|
| `--metadata` | Switch to metadata mode |
| `--check-ai` | Check for AI metadata (implies `--metadata`) |
| `--remove-ai` | Remove AI metadata (implies `--metadata`) |
| `--remove-all-metadata` | Also remove standard EXIF/XMP (use with `--remove-ai`) |
| `-a, --ai-only` | Clone only AI metadata (for cloning mode) |

---

## Python API

### Watermark Removal

```python
from pathlib import Path
from watermark_remover import WatermarkRemover, remove_watermark, is_watermark_removal_available

if is_watermark_removal_available():
    # Quick one-off usage
    remove_watermark(
        image_path=Path("watermarked.png"),
        output_path=Path("cleaned.png"),
        strength=0.04,
    )

    # Persistent instance (recommended for batch/repeated use)
    remover = WatermarkRemover(model_id="Lykon/dreamshaper-8", device="cpu")
    remover.remove_watermark(
        image_path=Path("watermarked.png"),
        output_path=Path("cleaned.png"),
        strength=0.04,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
    )

    # Batch mode
    remover.remove_watermark_batch(
        input_dir=Path("input_images"),
        output_dir=Path("cleaned_images"),
        strength=0.04,
    )
```

### Metadata Operations

```python
from pathlib import Path
from metadata_handler import (
    clone_metadata, extract_metadata, extract_ai_metadata,
    has_ai_metadata, remove_ai_metadata, has_c2pa_metadata, extract_c2pa_info,
)

# Clone metadata between images
clone_metadata(Path("source.png"), Path("target.png"), Path("output.png"))

# Inspect AI metadata
ai_meta = extract_ai_metadata(Path("image.png"))
print(has_ai_metadata(Path("image.png")))

# C2PA provenance
if has_c2pa_metadata(Path("image.png")):
    print(extract_c2pa_info(Path("image.png")))

# Strip AI metadata
remove_ai_metadata(Path("image.png"), Path("cleaned.png"))
```

---

## Watermark Removal Guide

### How It Works

1. **Encode** — project the image into diffusion latent space via the VAE encoder.
2. **Noise** — inject controlled noise according to `strength`, disrupting hidden watermark patterns.
3. **Denoise** — reconstruct via reverse diffusion over `steps` iterations.
4. **Decode** — convert clean latents back to pixel space.

This targets **invisible/embedded** watermarks (SynthID, StableSignature, TreeRing), not visible logos or text overlays.

### Recommended Presets

| Use Case | Flags |
|----------|-------|
| Minimal change (default) | `--strength 0.04 --steps 50` |
| Balanced | `--strength 0.15 --steps 50` |
| Aggressive | `--strength 0.35 --steps 60` |
| Maximum removal | `--strength 0.7 --steps 60` |

### Tuning Tips

- **Watermark still detected?** Increase `--strength` by 0.05–0.1.
- **Image changed too much?** Decrease `--strength`.
- **Output noisy?** Increase `--steps` by 10–20.
- **Too slow?** Reduce `--steps`, or use a GPU.
- **MPS out of memory?** Use `--device cpu`.

For full flag reference, see [CLI Reference](#cli-reference). For compatible base models, see [Pipeline Profiles](#pipeline-profiles).

---

## Verification

Test watermark removal end-to-end with Google SynthID:

1. **Generate** a watermarked image at [Google AI Studio](https://aistudio.google.com/) or [Gemini](https://gemini.google.com/).
2. **Remove** the watermark:

```bash
noai-watermark image.png -o cleaned.png
```

3. **Verify** by uploading both images to [AI Studio](https://aistudio.google.com/) and using the SynthID detection tool.

| Image | Expected SynthID Result |
|-------|-------------------------|
| Original | *"This image contains a SynthID watermark, which indicates that all or part of it was generated or edited using Google AI."* |
| Cleaned | *"This image was not made with Google AI."* |

See the [Example](#example) section for a real before/after comparison with default settings.

Results vary with `strength`, `steps`, and model choice.

---

## AI Metadata Types

The following AI metadata sources are detected and can be cloned or stripped:

| Source | Fields |
|--------|--------|
| Stable Diffusion WebUI | `parameters`, `postprocessing`, `extras` |
| ComfyUI | `workflow`, `prompt` |
| Common AI keys | `prompt`, `seed`, `model`, `sampler`, etc. |
| C2PA provenance | Google Imagen, OpenAI DALL-E, Adobe Firefly, Microsoft Designer |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError` for torch / diffusers | `pip install noai-watermark` (all dependencies are included by default) |
| `ImportError` for controlnet-aux / color-matcher | `pip install noai-watermark` (all dependencies are included by default) |
| `externally-managed-environment` | Use `pipx install noai-watermark` or a virtual environment. See [Installation](#installation). |
| HuggingFace Hub rate limit | Set `HF_TOKEN` env var or pass `--hf-token` |
| `MPS backend out of memory` | Use `--device cpu`, or lower `--strength` and `--steps` |
| Output too different from input | Decrease `--strength` |
| Very slow on CPU | Reduce `--steps`, or use a GPU with `--device cuda` |

---

## Project Structure

```text
src/
  __init__.py            # Package root and public API re-exports
  metadata_handler.py    # Public API facade for metadata operations
  constants.py           # AI metadata detection lists and config
  utils.py               # Format helpers
  c2pa.py                # C2PA chunk detection / extraction / injection
  extractor.py           # Read-only metadata extraction
  injector.py            # Write metadata into images
  cleaner.py             # AI metadata identification and removal
  cloner.py              # High-level extract -> inject pipeline
  watermark_remover.py   # WatermarkRemover class and orchestration
  watermark_profiles.py  # Model IDs, strength presets, profile detection
  img2img_runner.py      # Img2img execution with progress and MPS fallback
  noai_cli.py            # CLI argument parsing and routing
  noai_cli_watermark.py  # Watermark removal handler
  download_ui.py         # Download progress bars, size estimation, prompts
  progress.py            # Terminal animation and shared pipeline helpers
  ctrlregen/             # CtrlRegen sub-package (optional)
    __init__.py
    engine.py            # Pipeline orchestration and single-image inference
    tiling.py            # Tile-based processing for large images
    pipeline.py          # SD + ControlNet + IP-Adapter pipeline
    ip_adapter.py        # DINOv2-based IP-Adapter mixin
    color.py             # Histogram color matching

tests/
  conftest.py              test_constants.py
  test_utils.py            test_c2pa.py
  test_extractor.py        test_injector.py
  test_cleaner.py          test_cloner.py
  test_metadata_handler.py test_watermark_remover.py
  test_watermark_profiles.py
  test_download_ui.py      test_progress.py
  test_ctrlregen.py
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest
pytest --cov=src --cov-report=html
```

---

## Acknowledgements

The CtrlRegen integration is adapted from [yepengliu/CtrlRegen](https://github.com/yepengliu/CtrlRegen) (Apache-2.0) by Yepeng Liu, Yiren Song, Hai Ci, Yu Zhang, Haofan Wang, Mike Zheng Shou, and Yuheng Bu.

```bibtex
@article{liu2024ctrlregen,
  title   = {Image watermarks are removable using controllable regeneration from clean noise},
  author  = {Liu, Yepeng and Song, Yiren and Ci, Hai and Zhang, Yu and Wang, Haofan and Shou, Mike Zheng and Bu, Yuheng},
  journal = {arXiv preprint arXiv:2410.05470},
  year    = {2024}
}
```
