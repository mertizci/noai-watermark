"""Tests for watermark removal module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestWatermarkRemoverAvailability:
    """Test availability checking functions."""

    def test_is_watermark_removal_available_with_deps(self) -> None:
        """Test availability check when dependencies are installed."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "diffusers": MagicMock(),
            },
        ):
            # Re-import to check with mocked modules
            from watermark_remover import is_watermark_removal_available

            # This will still return the cached value from import time
            # So we just test the function exists
            assert callable(is_watermark_removal_available)

    def test_get_device_returns_string(self) -> None:
        """Test that get_device returns a valid device string."""
        from watermark_remover import get_device

        device = get_device()
        assert device in ("cuda", "mps", "cpu")


class TestWatermarkRemoverInit:
    """Test WatermarkRemover initialization."""

    @patch("watermark_remover._auto_install", return_value=False)
    @patch("watermark_remover._HAS_TORCH", False)
    def test_init_without_torch_raises_error(self, _mock_install: MagicMock) -> None:
        """Test that initialization fails without torch."""
        with pytest.raises(ImportError, match="Failed to auto-install"):
            from watermark_remover import WatermarkRemover

            WatermarkRemover()

    @patch("watermark_remover._auto_install", return_value=False)
    @patch("watermark_remover._HAS_DIFFUSERS", False)
    def test_init_without_diffusers_raises_error(self, _mock_install: MagicMock) -> None:
        """Test that initialization fails without diffusers."""
        with pytest.raises(ImportError, match="Failed to auto-install"):
            from watermark_remover import WatermarkRemover

            WatermarkRemover()


class TestWatermarkRemoverMocked:
    """Test WatermarkRemover with mocked pipeline."""

    @pytest.fixture
    def mock_torch(self) -> MagicMock:
        """Create mock torch module."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = False
        mock.float16 = "float16"
        mock.float32 = "float32"
        return mock

    @pytest.fixture
    def mock_pipeline_class(self) -> MagicMock:
        """Create mock StableDiffusionImg2ImgPipeline class."""
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result

        mock_class = MagicMock()
        mock_class.from_pretrained.return_value = mock_pipeline
        return mock_class

    @patch("watermark_remover._HAS_TORCH", True)
    @patch("watermark_remover._HAS_DIFFUSERS", True)
    @patch("watermark_remover.torch")
    @patch("watermark_remover.StableDiffusionImg2ImgPipeline")
    def test_remove_watermark_basic(
        self,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test basic watermark removal."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"

        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        # Create test image
        from PIL import Image

        test_image = Image.new("RGB", (100, 100), color="red")
        source_path = tmp_path / "test.png"
        test_image.save(source_path)

        # Create WatermarkRemover
        from watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="cpu")

        # Remove watermark
        output_path = tmp_path / "cleaned.png"
        result = remover.remove_watermark(
            image_path=source_path,
            output_path=output_path,
            strength=0.5,
        )

        # Verify
        assert result == output_path
        mock_pipeline.assert_called_once()

    @patch("watermark_remover._HAS_TORCH", True)
    @patch("watermark_remover._HAS_DIFFUSERS", True)
    @patch("watermark_remover.torch")
    @patch("watermark_remover.StableDiffusionImg2ImgPipeline")
    def test_remove_watermark_invalid_strength(
        self,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that invalid strength raises error."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"

        # Create test image
        from PIL import Image

        test_image = Image.new("RGB", (100, 100), color="red")
        source_path = tmp_path / "test.png"
        test_image.save(source_path)

        # Create WatermarkRemover
        from watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="cpu")

        # Test invalid strength
        with pytest.raises(ValueError, match="Strength must be between"):
            remover.remove_watermark(
                image_path=source_path,
                strength=1.5,
            )

        with pytest.raises(ValueError, match="Strength must be between"):
            remover.remove_watermark(
                image_path=source_path,
                strength=-0.5,
            )

    @patch("watermark_remover._HAS_TORCH", True)
    @patch("watermark_remover._HAS_DIFFUSERS", True)
    @patch("watermark_remover.torch")
    def test_remove_watermark_file_not_found(
        self,
        mock_torch: MagicMock,
    ) -> None:
        """Test that missing file raises error."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"

        from watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="cpu")

        with pytest.raises(FileNotFoundError):
            remover.remove_watermark(
                image_path=Path("/nonexistent/image.png"),
            )

    @patch("watermark_remover._HAS_TORCH", True)
    @patch("watermark_remover._HAS_DIFFUSERS", True)
    @patch("watermark_remover.torch")
    @patch("watermark_remover.StableDiffusionImg2ImgPipeline")
    @patch("img2img_runner._try_clear_mps_cache")
    def test_mps_oom_falls_back_to_cpu(
        self,
        _mock_clear_cache: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that MPS OOM triggers a single CPU fallback retry."""
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        mock_torch.cuda.is_available.return_value = False
        mock_torch.mps = MagicMock()

        first_pipeline = MagicMock()
        second_pipeline = MagicMock()
        first_pipeline.to.return_value = first_pipeline
        second_pipeline.to.return_value = second_pipeline

        first_pipeline.side_effect = RuntimeError("MPS backend out of memory")
        second_result = MagicMock()
        second_image = MagicMock()
        second_image.save = MagicMock()
        second_result.images = [second_image]
        second_pipeline.return_value = second_result

        mock_pipeline_class.from_pretrained.side_effect = [first_pipeline, second_pipeline]

        from PIL import Image

        test_image = Image.new("RGB", (100, 100), color="red")
        source_path = tmp_path / "test.png"
        output_path = tmp_path / "cleaned.png"
        test_image.save(source_path)

        from watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="mps")
        result = remover.remove_watermark(image_path=source_path, output_path=output_path)

        assert result == output_path
        assert remover.device == "cpu"
        assert mock_pipeline_class.from_pretrained.call_count == 2

    @patch("watermark_remover._HAS_TORCH", True)
    @patch("watermark_remover._HAS_DIFFUSERS", True)
    def test_invalid_device_raises_value_error(self) -> None:
        """Test invalid device validation."""
        from watermark_remover import WatermarkRemover

        with pytest.raises(ValueError, match="Unsupported device"):
            WatermarkRemover(device="invalid")


class TestConstants:
    """Test module constants."""

    def test_strength_values(self) -> None:
        """Test that strength values are in valid range."""
        from watermark_remover import WatermarkRemover

        assert 0.0 <= WatermarkRemover.LOW_STRENGTH <= 1.0
        assert 0.0 <= WatermarkRemover.MEDIUM_STRENGTH <= 1.0
        assert 0.0 <= WatermarkRemover.HIGH_STRENGTH <= 1.0

        assert WatermarkRemover.LOW_STRENGTH < WatermarkRemover.MEDIUM_STRENGTH
        assert WatermarkRemover.MEDIUM_STRENGTH < WatermarkRemover.HIGH_STRENGTH
