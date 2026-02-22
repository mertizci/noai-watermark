"""Tests for the ctrlregen sub-package."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

class TestCtrlRegenAvailability:
    """Test is_ctrlregen_available under different dependency states."""

    def test_available_when_all_deps_present(self) -> None:
        with (
            patch("ctrlregen.engine._HAS_DIFFUSERS", True),
            patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True),
            patch("ctrlregen.engine._HAS_COLOR_MATCHER", True),
        ):
            from ctrlregen.engine import is_ctrlregen_available

            assert is_ctrlregen_available() is True

    def test_unavailable_without_diffusers(self) -> None:
        with (
            patch("ctrlregen.engine._HAS_DIFFUSERS", False),
            patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True),
            patch("ctrlregen.engine._HAS_COLOR_MATCHER", True),
        ):
            from ctrlregen.engine import is_ctrlregen_available

            assert is_ctrlregen_available() is False

    def test_unavailable_without_controlnet_aux(self) -> None:
        with (
            patch("ctrlregen.engine._HAS_DIFFUSERS", True),
            patch("ctrlregen.engine._HAS_CONTROLNET_AUX", False),
            patch("ctrlregen.engine._HAS_COLOR_MATCHER", True),
        ):
            from ctrlregen.engine import is_ctrlregen_available

            assert is_ctrlregen_available() is False

    def test_unavailable_without_color_matcher(self) -> None:
        with (
            patch("ctrlregen.engine._HAS_DIFFUSERS", True),
            patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True),
            patch("ctrlregen.engine._HAS_COLOR_MATCHER", False),
        ):
            from ctrlregen.engine import is_ctrlregen_available

            assert is_ctrlregen_available() is False


# ---------------------------------------------------------------------------
# CtrlRegenEngine init
# ---------------------------------------------------------------------------

class TestCtrlRegenEngineInit:
    """Test CtrlRegenEngine constructor validation."""

    @patch("ctrlregen.engine._HAS_DIFFUSERS", False)
    @patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True)
    @patch("ctrlregen.engine._HAS_COLOR_MATCHER", True)
    @patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip"))
    def test_init_raises_when_deps_missing(self, _mock_pip: MagicMock) -> None:
        from ctrlregen.engine import CtrlRegenEngine

        with pytest.raises(ImportError, match="Failed to auto-install"):
            CtrlRegenEngine(device="cpu")

    @patch("ctrlregen.engine._HAS_DIFFUSERS", True)
    @patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True)
    @patch("ctrlregen.engine._HAS_COLOR_MATCHER", True)
    @patch("ctrlregen.engine.torch")
    def test_init_stores_defaults(self, mock_torch: MagicMock) -> None:
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"

        from ctrlregen.engine import CtrlRegenEngine, DEFAULT_BASE_MODEL

        engine = CtrlRegenEngine(device="cpu")
        assert engine.base_model_id == DEFAULT_BASE_MODEL
        assert engine.device == "cpu"
        assert engine.torch_dtype == "float32"
        assert engine._pipeline is None

    @patch("ctrlregen.engine._HAS_DIFFUSERS", True)
    @patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True)
    @patch("ctrlregen.engine._HAS_COLOR_MATCHER", True)
    @patch("ctrlregen.engine.torch")
    def test_init_custom_model(self, mock_torch: MagicMock) -> None:
        mock_torch.float16 = "float16"

        from ctrlregen.engine import CtrlRegenEngine

        engine = CtrlRegenEngine(
            base_model_id="my/custom-model", device="cuda"
        )
        assert engine.base_model_id == "my/custom-model"
        assert engine.torch_dtype == "float16"

    @patch("ctrlregen.engine._HAS_DIFFUSERS", True)
    @patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True)
    @patch("ctrlregen.engine._HAS_COLOR_MATCHER", True)
    @patch("ctrlregen.engine.torch")
    def test_init_hf_token_from_env(self, mock_torch: MagicMock) -> None:
        mock_torch.float32 = "float32"

        from ctrlregen.engine import CtrlRegenEngine

        with patch.dict("os.environ", {"HF_TOKEN": "test_token_123"}):
            engine = CtrlRegenEngine(device="cpu")
            assert engine.hf_token == "test_token_123"

    @patch("ctrlregen.engine._HAS_DIFFUSERS", True)
    @patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True)
    @patch("ctrlregen.engine._HAS_COLOR_MATCHER", True)
    @patch("ctrlregen.engine.torch")
    def test_init_explicit_hf_token(self, mock_torch: MagicMock) -> None:
        mock_torch.float32 = "float32"

        from ctrlregen.engine import CtrlRegenEngine

        engine = CtrlRegenEngine(device="cpu", hf_token="explicit_token")
        assert engine.hf_token == "explicit_token"


# ---------------------------------------------------------------------------
# CtrlRegenEngine progress callback
# ---------------------------------------------------------------------------

class TestCtrlRegenProgress:
    """Test progress callback behaviour."""

    @patch("ctrlregen.engine._HAS_DIFFUSERS", True)
    @patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True)
    @patch("ctrlregen.engine._HAS_COLOR_MATCHER", True)
    @patch("ctrlregen.engine.torch")
    def test_set_progress_invokes_callback(self, mock_torch: MagicMock) -> None:
        mock_torch.float32 = "float32"

        from ctrlregen.engine import CtrlRegenEngine

        messages: list[str] = []
        engine = CtrlRegenEngine(
            device="cpu", progress_callback=messages.append
        )
        engine._set_progress("hello")
        assert messages == ["hello"]

    @patch("ctrlregen.engine._HAS_DIFFUSERS", True)
    @patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True)
    @patch("ctrlregen.engine._HAS_COLOR_MATCHER", True)
    @patch("ctrlregen.engine.torch")
    def test_set_progress_none_callback_noop(self, mock_torch: MagicMock) -> None:
        mock_torch.float32 = "float32"

        from ctrlregen.engine import CtrlRegenEngine

        engine = CtrlRegenEngine(device="cpu")
        engine._set_progress("should not crash")

    @patch("ctrlregen.engine._HAS_DIFFUSERS", True)
    @patch("ctrlregen.engine._HAS_CONTROLNET_AUX", True)
    @patch("ctrlregen.engine._HAS_COLOR_MATCHER", True)
    @patch("ctrlregen.engine.torch")
    def test_set_progress_swallows_callback_errors(
        self, mock_torch: MagicMock
    ) -> None:
        mock_torch.float32 = "float32"

        from ctrlregen.engine import CtrlRegenEngine

        def bad_callback(msg: str) -> None:
            raise RuntimeError("oops")

        engine = CtrlRegenEngine(device="cpu", progress_callback=bad_callback)
        engine._set_progress("should not raise")


# ---------------------------------------------------------------------------
# WatermarkRemover ctrlregen dispatch
# ---------------------------------------------------------------------------

class TestWatermarkRemoverCtrlRegenDispatch:
    """Ensure WatermarkRemover routes to _run_ctrlregen for ctrlregen profile."""

    @patch("watermark_remover._HAS_TORCH", True)
    @patch("watermark_remover._HAS_DIFFUSERS", True)
    @patch("watermark_remover.torch")
    @patch("watermark_remover.StableDiffusionImg2ImgPipeline")
    def test_ctrlregen_model_sets_profile(
        self,
        _mock_pipe: MagicMock,
        mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.backends.mps.is_available.return_value = False

        from watermark_remover import WatermarkRemover

        remover = WatermarkRemover(
            model_id="yepengliu/ctrlregen", device="cpu"
        )
        assert remover.model_profile == "ctrlregen"

    @patch("watermark_remover._HAS_TORCH", True)
    @patch("watermark_remover._HAS_DIFFUSERS", True)
    @patch("watermark_remover.torch")
    @patch("watermark_remover.StableDiffusionImg2ImgPipeline")
    def test_default_model_sets_default_profile(
        self,
        _mock_pipe: MagicMock,
        mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.backends.mps.is_available.return_value = False

        from watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="cpu")
        assert remover.model_profile == "default"


# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------

class TestEngineConstants:
    """Verify exported constant values."""

    def test_ctrlregen_hf_repo(self) -> None:
        from ctrlregen.engine import CTRLREGEN_HF_REPO

        assert CTRLREGEN_HF_REPO == "yepengliu/ctrlregen"

    def test_spatial_subfolder(self) -> None:
        from ctrlregen.engine import SPATIAL_SUBFOLDER

        assert "spatial_control" in SPATIAL_SUBFOLDER

    def test_semantic_weight_name(self) -> None:
        from ctrlregen.engine import SEMANTIC_WEIGHT_NAME

        assert SEMANTIC_WEIGHT_NAME.endswith(".bin")

    def test_default_base_model(self) -> None:
        from ctrlregen.engine import DEFAULT_BASE_MODEL

        assert DEFAULT_BASE_MODEL == "SG161222/Realistic_Vision_V4.0_noVAE"

    def test_custom_vae_id(self) -> None:
        from ctrlregen.engine import CUSTOM_VAE_ID

        assert CUSTOM_VAE_ID == "stabilityai/sd-vae-ft-mse"

    def test_default_guidance_scale(self) -> None:
        from ctrlregen.engine import DEFAULT_GUIDANCE_SCALE

        assert DEFAULT_GUIDANCE_SCALE == 2.0

    def test_quality_prompt(self) -> None:
        from ctrlregen.engine import QUALITY_PROMPT

        assert "quality" in QUALITY_PROMPT.lower()

    def test_negative_prompt(self) -> None:
        from ctrlregen.engine import NEGATIVE_PROMPT

        assert "low quality" in NEGATIVE_PROMPT.lower()

    def test_canny_thresholds(self) -> None:
        from ctrlregen.engine import CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD

        assert CANNY_LOW_THRESHOLD == 100
        assert CANNY_HIGH_THRESHOLD == 150
