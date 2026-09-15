"""Tests for the ctrlregen sub-package."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image


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
# Tile compositing
# ---------------------------------------------------------------------------

class _PassthroughPipeline:
    """Return the padded tile so tests exercise edge compositing only."""

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(images=[kwargs["image"][0]])


class _PassthroughCanny:
    """Avoid model work while keeping the tiled ControlNet call shape intact."""

    def __call__(self, image: Image.Image, **_kwargs: object) -> Image.Image:
        return image


class TestCtrlRegenTiling:
    @pytest.mark.parametrize("dimensions", [(720, 480), (480, 720)])
    def test_clips_padded_edge_tile_to_canvas(
        self, dimensions: tuple[int, int]
    ) -> None:
        """Wide and tall inputs must not blend 512px padding outside the canvas."""
        from ctrlregen.tiling import run_tiled

        source_color = (17, 23, 31)
        result = run_tiled(
            pipeline=_PassthroughPipeline(),
            canny_detector=_PassthroughCanny(),
            image=Image.new("RGB", dimensions, color=source_color),
            strength=0.04,
            num_inference_steps=50,
            guidance_scale=2.0,
            seed=1,
            tile_size=512,
            tile_overlap=192,
            quality_prompt="test",
            negative_prompt="test",
            canny_low=100,
            canny_high=150,
            device="cpu",
            set_progress=lambda _message: None,
        )

        assert result.size == dimensions
        assert result.getpixel((dimensions[0] // 2, dimensions[1] // 2)) == source_color


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


@pytest.mark.parametrize("dimensions", [(1024, 768), (768, 1024)])
def test_engine_uses_local_semantic_reference_for_each_tile(monkeypatch, dimensions):
    """Exercise engine -> tiling so a whole-portrait reference cannot regress."""
    from ctrlregen.engine import CtrlRegenEngine
    import ctrlregen.engine as engine_module

    references = []

    class RecordingPipeline:
        def __call__(self, **kwargs):
            tile = kwargs["image"][0]
            reference = kwargs["ip_adapter_image"][0]
            references.append(reference)
            assert reference is tile
            return SimpleNamespace(images=[tile])

    engine = CtrlRegenEngine.__new__(CtrlRegenEngine)
    engine.device = "cpu"
    engine._pipeline = RecordingPipeline()
    engine._canny_detector = _PassthroughCanny()
    engine._progress_callback = None
    monkeypatch.setattr(engine, "load", lambda: None)
    monkeypatch.setattr(engine_module, "color_match", lambda reference, source: source)
    source = Image.new("RGB", dimensions, (90, 100, 110))
    result = engine.run(source, strength=0.25, num_inference_steps=4, seed=7)
    assert result.size == dimensions
    assert len(references) > 1
    assert all(reference is not source for reference in references)
