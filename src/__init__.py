"""noai-watermark — AI metadata and invisible watermark toolkit.

This package exposes two main subsystems:

1. **Metadata pipeline** (always available): clone, extract, detect,
   and remove AI-generation metadata from PNG/JPEG images.
2. **Watermark removal** (optional ``[watermark]`` extra): diffusion-
   based regeneration attack for stripping invisible watermarks.
"""

__version__ = "0.1.16"

from metadata_handler import (
    clone_metadata,
    clone_ai_metadata,
    extract_metadata,
    extract_ai_metadata,
    has_ai_metadata,
    is_supported_format,
    remove_ai_metadata,
    has_ai_content,
    get_ai_metadata_summary,
)

# Watermark profiles (always available — pure config, no ML deps)
from watermark_profiles import (
    get_recommended_strength,
    get_model_id_for_profile,
)

# Watermark removal (optional - requires extra dependencies)
try:
    from watermark_remover import (
        WatermarkRemover,
        remove_watermark,
        is_watermark_removal_available,
    )
except ImportError:
    pass  # Watermark dependencies not installed

__all__ = [
    # Metadata handling
    "clone_metadata",
    "clone_ai_metadata",
    "extract_metadata",
    "extract_ai_metadata",
    "has_ai_metadata",
    "is_supported_format",
    "remove_ai_metadata",
    "has_ai_content",
    "get_ai_metadata_summary",
    # Watermark removal (optional)
    "WatermarkRemover",
    "remove_watermark",
    "get_recommended_strength",
    "get_model_id_for_profile",
    "is_watermark_removal_available",
]
