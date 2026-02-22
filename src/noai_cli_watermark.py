"""CLI handler for the default watermark removal mode.

Extracted from ``cli.py`` so the main CLI module stays focused on
argument parsing and routing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path


def handle_remove_watermark(args: argparse.Namespace) -> int:
    """Run diffusion-based invisible watermark removal."""
    if args.verbose:
        from watermark_remover import (
            is_watermark_removal_available,
            WatermarkRemover,
        )
        from watermark_profiles import get_model_id_for_profile
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="CUDA is not available or torch_xla is imported. Disabling autocast.",
                category=UserWarning,
            )
            from watermark_remover import (
                is_watermark_removal_available,
                WatermarkRemover,
            )
            from watermark_profiles import get_model_id_for_profile

    if not is_watermark_removal_available():
        missing_pkgs = ["torch", "diffusers", "transformers", "accelerate"]
        print("Installing missing dependencies...", file=sys.stderr)
        import subprocess
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing_pkgs],
            )
            from importlib import reload
            import watermark_remover as _wr
            reload(_wr)
            from watermark_remover import is_watermark_removal_available as _check
            if not _check():
                raise RuntimeError("Still missing after install")
        except Exception:
            print("Error: Failed to install dependencies automatically.", file=sys.stderr)
            print("Try manually: pip install --force-reinstall noai-watermark", file=sys.stderr)
            return 1

    output_path = args.output if args.output else args.source
    selected_model_id = args.model or get_model_id_for_profile(args.model_profile)

    if args.model is None and args.model_profile == "ctrlregen":
        print(
            "Using CtrlRegen profile (ControlNet + DINOv2 IP-Adapter)."
        )

    from download_ui import get_models_to_download, preload_silently, prompt_for_download

    pending = get_models_to_download(selected_model_id, args.model_profile)

    if not prompt_for_download(pending, skip_prompt=args.yes):
        return 1

    try:
        progress_state: dict[str, str] = {"message": "Starting watermark removal..."}

        def set_progress(message: str) -> None:
            progress_state["message"] = message

        needs_download = bool(pending)

        remover = WatermarkRemover(
            model_id=selected_model_id,
            device=args.device,
            progress_callback=set_progress,
            hf_token=args.hf_token,
        )

        if args.verbose:
            print(f"Using device: {remover.device}")
            print(f"Model dtype: {remover.torch_dtype}")

        if needs_download and not args.verbose:
            _nc = bool(os.environ.get("NO_COLOR"))
            if _nc:
                print("\n  Downloading model weights\n")
            else:
                print(f"\n  \033[36m⬇\033[0m  \033[1mDownloading model weights\033[0m\n")
            preload_silently(remover)
            print()

        def run_remove() -> Path:
            set_progress("Preparing diffusion pipeline...")
            return remover.remove_watermark(
                image_path=args.source,
                output_path=output_path,
                strength=args.strength,
                num_inference_steps=args.steps,
            )

        if args.verbose:
            logging.basicConfig(level=logging.INFO)
            print(f"Source: {args.source}")
            print(f"Output: {output_path}")
            print(f"Strength: {args.strength}")
            print(f"Steps: {args.steps}")
            print(f"Model profile: {args.model_profile}")
            print(f"Model: {selected_model_id}")
            if args.model and args.model_profile != "default":
                print("Note: --model overrides --model-profile")
            print(f"Device: {args.device}")
            print()
            result_path = run_remove()
        else:
            from progress import run_with_progress, silence_library_output
            result_path = run_with_progress(
                silence_library_output(run_remove, set_progress),
                progress_state,
            )

        print(f"Successfully removed watermark from: {result_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
