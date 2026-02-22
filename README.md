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
noai-watermark source.png --remove-watermark --strength 0.04 --steps 50 -o cleaned.png
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
pip install "noai-watermark[watermark]"

noai-watermark source.png --remove-watermark -o cleaned.png
```

For best quality (larger download):

```bash
pip install "noai-watermark[ctrlregen]"

noai-watermark source.png --remove-watermark --model-profile ctrlregen -o cleaned.png
```

---

## Installation

### From PyPI

```bash
# Metadata tools only (no ML dependencies)
pip install noai-watermark

# Default watermark removal (img2img)
pip install "noai-watermark[watermark]"

# CtrlRegen watermark removal (best quality)
pip install "noai-watermark[ctrlregen]"
```

> **macOS (Homebrew Python):** If you get `externally-managed-environment` error, use `pipx` or a virtual environment:
>
> ```bash
> # Option 1: pipx (recommended for CLI tools)
> brew install pipx
> pipx install "noai-watermark[watermark]"
>
> # Option 2: virtual environment
> python3 -m venv ~/.noai-venv
> source ~/.noai-venv/bin/activate
> pip install "noai-watermark[watermark]"
> ```

### From GitHub

```bash
pip install "git+https://github.com/mertizci/noai-watermark.git"
pip install "noai-watermark[watermark] @ git+https://github.com/mertizci/noai-watermark.git"
```

### Local Development

```bash
git clone https://github.com/mertizci/noai-watermark.git
cd noai-watermark
pip install -e ".[dev,watermark,ctrlregen]"
```

### Requirements

- Python >= 3.10
- Core: `pillow >= 10.0.0`, `piexif >= 1.1.3`
- Watermark removal: `torch >= 2.0.0`, `diffusers >= 0.25.0`, `transformers >= 4.35.0`, `accelerate >= 0.25.0`
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
| **Speed** | Faster | Slower (multiple models in the pipeline) |
| **Install** | `pip install "noai-watermark[watermark]"` | `pip install "noai-watermark[ctrlregen]"` |

> **Recommendation:** Start with `default` for quick iteration. Switch to `ctrlregen` when output quality is the priority.

Download sizes are estimated dynamically from HuggingFace Hub before the first run. Models are cached locally after download — subsequent runs are instant.

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

All commands use `noai-watermark` (alias: `photo-metadata`). Add `-v` to any command for verbose output.

### Watermark Removal

```bash
# Basic removal
noai-watermark source.png --remove-watermark -o cleaned.png

# Force CPU inference
noai-watermark source.png --remove-watermark --device cpu -o cleaned.png

# Custom strength and steps
noai-watermark source.png --remove-watermark --strength 0.7 --steps 60 -o cleaned.png

# Use a different base model
noai-watermark source.png --remove-watermark --model runwayml/stable-diffusion-v1-5 -o cleaned.png

# Photorealistic model for photo inputs
noai-watermark source.png --remove-watermark --model SG161222/Realistic_Vision_V5.1_noVAE -o cleaned.png

# CtrlRegen profile
noai-watermark source.png --remove-watermark --model-profile ctrlregen -o cleaned.png

# Skip download confirmation prompt
noai-watermark source.png --remove-watermark -y -o cleaned.png

# With HuggingFace token (or set HF_TOKEN env var)
noai-watermark source.png --remove-watermark --hf-token hf_xxxxx -o cleaned.png
```

### Metadata Operations

```bash
# Clone all metadata from source to target
noai-watermark source.png target.png -o output.png

# Clone only AI metadata
noai-watermark source.png target.png -o output.png --ai-only

# Check for AI metadata
noai-watermark source.png --check-ai

# Remove AI metadata (keeps standard EXIF/XMP)
noai-watermark source.png --remove-ai -o cleaned.png

# Remove all metadata
noai-watermark source.png --remove-ai --remove-all-metadata -o cleaned.png
```

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

1. **Encode** — the input image is projected into diffusion latent space via the VAE encoder.
2. **Noise** — controlled noise is injected according to `strength`, disrupting hidden watermark patterns embedded in the pixel data.
3. **Denoise** — the diffusion model reconstructs the image over `steps` iterations via reverse diffusion.
4. **Decode** — the VAE decoder converts the clean latents back to pixel space, producing an output with reduced or eliminated watermark artifacts.

This targets **invisible/embedded** watermark traces (SynthID, StableSignature, TreeRing, etc.), not visible logos or text overlays.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--strength` | `0.04` | How strongly the image is regenerated (0.0–1.0). Higher = more watermark removal but more visual change. |
| `--steps` | `50` | Denoising iterations. More steps = better quality, slower. |
| `--model-profile` | `default` | Pipeline architecture: `default` or `ctrlregen`. See [Pipeline Profiles](#pipeline-profiles). |
| `--model` | `Lykon/dreamshaper-8` | Base SD checkpoint. Any SD 1.5-compatible HuggingFace model ID. |
| `--device` | `auto` | Inference device: `auto`, `cpu`, `mps`, or `cuda`. |
| `--hf-token` | — | HuggingFace API token. Falls back to `HF_TOKEN` env var. |
| `-y` | — | Skip the download confirmation prompt. |

### Recommended Presets

| Use Case | Flags |
|----------|-------|
| Minimal change (default) | `--strength 0.04 --steps 50` |
| Balanced | `--strength 0.15 --steps 50` |
| Aggressive cleanup | `--strength 0.35 --steps 60` |
| Maximum removal | `--strength 0.7 --steps 60` |

### Tuning Tips

- **Watermark still detected?** Increase `--strength` by 0.05–0.1.
- **Image changed too much?** Decrease `--strength`, keep `--steps` moderate.
- **Output noisy or rough?** Increase `--steps` by 10–20.
- **Too slow?** Reduce `--steps` first, then consider a smaller model.
- **MPS out of memory?** Use `--device cpu` or lower `--strength`.

### Compatible Base Models

Any SD 1.5-compatible model from HuggingFace works with `--model`:

| Model | Notes |
|-------|-------|
| `Lykon/dreamshaper-8` | Default. Balanced quality and speed. |
| `runwayml/stable-diffusion-v1-5` | Classic baseline. |
| `SG161222/Realistic_Vision_V5.1_noVAE` | Photorealistic output. |
| `segmind/tiny-sd` | Lower memory usage, smaller download. |

Browse more at [HuggingFace](https://huggingface.co/models?pipeline_tag=text-to-image).

---

## Verification

Test watermark removal end-to-end with Google SynthID:

1. **Generate** a watermarked image at [Google AI Studio](https://aistudio.google.com/) or [Gemini](https://gemini.google.com/).
2. **Remove** the watermark:

```bash
noai-watermark image.png --remove-watermark -o cleaned.png
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
| `ImportError` for torch / diffusers | `pip install "noai-watermark[watermark]"` |
| `externally-managed-environment` | Use `pipx install "noai-watermark[watermark]"` or a virtual environment. See [Installation](#installation). |
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
  cli.py                 # CLI argument parsing and routing
  cli_watermark.py       # --remove-watermark command handler
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
