"""
ModelCache — Centralized model download and caching.

Extracted from ``BaseVisionTool._resolve_model_path()`` + ``download_ckpt()``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from vision_tools.utils.locations import MODELS_CACHE_DIR, CACHE_DIR

logger = logging.getLogger(__name__)

# Note: Environment variable VISION_TOOLS_CACHE_DIR is now handled in vision_tools.utils.locations


class ModelCache:
    """Manages model file caching and downloads.

    Provides a simple get-or-download interface:
    1. If model_id is an existing file → return it.
    2. If it's cached → return cached path.
    3. Otherwise → call the downloader and cache the result.

    Args:
        cache_dir: Directory for cached model files.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir or MODELS_CACHE_DIR).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_download(
        self,
        model_id: str,
        downloader: Callable[[str, Path], Path] | None = None,
    ) -> str:
        """Return local path for a model, downloading if necessary.

        Args:
            model_id: Model identifier or path.
            downloader: Optional callable(model_id, destination) → Path.
                        If not provided and model isn't cached, returns
                        model_id as-is (for library-managed models).

        Returns:
            Local file path as string.
        """
        # 1. Already a local file
        if os.path.isfile(model_id):
            logger.debug(f"ModelCache: '{model_id}' is a local file.")
            return model_id

        # 2. Check cache
        filename = Path(model_id).name
        cached_path = self.cache_dir / filename
        if cached_path.exists():
            logger.debug(f"ModelCache: found in cache: {cached_path}")
            return str(cached_path)

        # 3. Download
        if downloader is not None:
            logger.info(f"ModelCache: downloading '{model_id}' → {cached_path}")
            try:
                downloaded = downloader(model_id, cached_path)
                if downloaded and os.path.exists(str(downloaded)):
                    return str(downloaded)
            except NotImplementedError:
                logger.info(
                    f"ModelCache: downloader not implemented, "
                    f"passing through '{model_id}'"
                )
                return model_id
            except Exception as e:
                logger.error(f"ModelCache: download failed for '{model_id}': {e}")

        # 4. Pass through (library-managed models like HuggingFace IDs)
        logger.debug(f"ModelCache: passing through '{model_id}'")
        return model_id

    def clear(self) -> None:
        """Remove all cached model files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ModelCache: cleared {self.cache_dir}")
