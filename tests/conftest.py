"""Test configuration and fixtures."""

from __future__ import annotations

import struct
import tempfile
import zlib
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_png(temp_dir: Path) -> Path:
    """Create a sample PNG image for testing."""
    img_path = temp_dir / "sample.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def sample_jpg(temp_dir: Path) -> Path:
    """Create a sample JPG image for testing."""
    img_path = temp_dir / "sample.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def sample_png_with_ai_metadata(temp_dir: Path) -> Path:
    """Create a PNG image with AI metadata (Stable Diffusion style)."""
    from PIL.PngImagePlugin import PngInfo

    img_path = temp_dir / "ai_sample.png"
    img = Image.new("RGB", (512, 512), color="green")

    metadata = PngInfo()
    metadata.add_text(
        "parameters",
        "A beautiful landscape, Steps: 30, Sampler: Euler a, CFG scale: 7.5, Seed: 12345, Size: 512x512",
    )
    metadata.add_text("Model", "v1-5-pruned-emaonly")
    metadata.add_text("Software", "Stable Diffusion WebUI")

    img.save(img_path, "PNG", pnginfo=metadata)
    return img_path


@pytest.fixture
def sample_png_with_standard_metadata(temp_dir: Path) -> Path:
    """Create a PNG image with standard metadata."""
    from PIL.PngImagePlugin import PngInfo

    img_path = temp_dir / "standard_sample.png"
    img = Image.new("RGB", (100, 100), color="yellow")

    metadata = PngInfo()
    metadata.add_text("Author", "Test Author")
    metadata.add_text("Title", "Test Image")
    metadata.add_text("Description", "A test image for unit tests")
    metadata.add_text("Copyright", "Test Copyright")

    img.save(img_path, "PNG", pnginfo=metadata)
    return img_path


@pytest.fixture
def sample_png_rgba(temp_dir: Path) -> Path:
    """Create a PNG image with alpha channel."""
    img_path = temp_dir / "sample_rgba.png"
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def c2pa_png(temp_dir: Path) -> Path:
    """Copy the real Google Imagen C2PA PNG to temp directory for testing."""
    import shutil

    source = _PROJECT_ROOT / "example" / "source.png"
    if not source.exists():
        pytest.skip("C2PA test file (example/source.png) not found")

    dest = temp_dir / "c2pa_test.png"
    shutil.copy(source, dest)
    return dest


def _build_c2pa_chunk(payload: bytes) -> bytes:
    """Build a raw PNG caBX chunk (header + data + CRC) from payload bytes."""
    chunk_type = b"caBX"
    length = struct.pack(">I", len(payload))
    crc = struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    return length + chunk_type + payload + crc


@pytest.fixture
def openai_png(temp_dir: Path) -> Path:
    """Create a synthetic PNG with OpenAI-style C2PA metadata for testing.

    Embeds a caBX chunk whose payload contains the byte signatures that
    the C2PA parser scans for: issuer (OpenAI), AI tool (GPT-4o),
    action (c2pa.created), timestamp, and source type.
    """
    from c2pa import inject_c2pa_chunk

    base_path = temp_dir / "openai_base.png"
    Image.new("RGB", (64, 64), color="purple").save(base_path, "PNG")

    payload = b"\x00".join([
        b"jumb",
        b"c2pa",
        b"OpenAI",
        b"Truepic",
        b"GPT-4o",
        b"ChatGPT",
        b"c2pa.created",
        b"c2pa.edited",
        b"trainedAlgorithmicMedia",
        b"20260101120000Z",
    ])
    c2pa_chunk = _build_c2pa_chunk(payload)

    dest = temp_dir / "openai_test.png"
    inject_c2pa_chunk(base_path, dest, c2pa_chunk)
    return dest
