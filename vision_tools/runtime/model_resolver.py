"""
ModelResolver — Selects the appropriate model variant based on hardware and mode.

Extracted from ``BaseVisionTool._resolve_model_from_config()``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelResolver:
    """Selects model variant based on hardware resources and user preference.

    The resolver reads a ``models`` dict from config with tiered variants
    (speed, balanced, accuracy) and picks the best one that fits.

    Example config::

        {
            "models": {
                "speed": {"id": "yolov8n", "min_vram_gb": 0},
                "balanced": {"id": "yolov8m", "min_vram_gb": 4},
                "accuracy": {"id": "yolov8x", "min_vram_gb": 8}
            },
            "model": "yolov8n"  # fallback
        }
    """

    def __init__(self, gpu_vram_gb: float | None = None) -> None:
        self._gpu_vram_gb = gpu_vram_gb

    @property
    def gpu_vram_gb(self) -> float:
        """Lazy-detect GPU VRAM if not set."""
        if self._gpu_vram_gb is None:
            self._gpu_vram_gb = self._detect_vram()
        return self._gpu_vram_gb

    def resolve(
        self,
        variants: dict[str, Any] | None = None,
        mode: str = "auto",
        fallback: str | None = None,
    ) -> str:
        """Select the best model variant.

        Args:
            variants: Tiered model variants dict
                      (keys: "speed", "balanced", "accuracy").
            mode: Selection mode — "auto", "speed", "balanced", "accuracy".
            fallback: Fallback model ID if variants aren't defined.

        Returns:
            Model identifier string.

        Raises:
            ValueError: If no model can be resolved.
        """
        # No variants → use fallback directly
        if not variants:
            if fallback is None:
                raise ValueError(
                    "No 'models' variants and no 'model' fallback in config."
                )
            return fallback

        # Explicit mode selection
        if mode != "auto":
            if mode in variants:
                logger.info(f"ModelResolver: using '{mode}' variant")
                return variants[mode]["id"]

            # Fallback to first available
            first = next(iter(variants.values()))
            logger.warning(
                f"ModelResolver: mode '{mode}' not in variants, "
                f"using fallback"
            )
            return first.get("id", fallback or "")

        # Auto mode: pick best that fits in available VRAM
        vram = self.gpu_vram_gb
        for tier in ["accuracy", "balanced", "speed"]:
            if tier in variants:
                min_vram = variants[tier].get("min_vram_gb", 0)
                if vram >= min_vram:
                    logger.info(
                        f"ModelResolver: auto-selected '{tier}' "
                        f"(VRAM: {vram:.1f}GB)"
                    )
                    return variants[tier]["id"]

        # Last resort: speed variant or fallback
        if "speed" in variants:
            return variants["speed"]["id"]
        if fallback:
            return fallback

        raise ValueError("ModelResolver: could not resolve any model variant.")

    @staticmethod
    def _detect_vram() -> float:
        """Detect GPU VRAM. Returns 0.0 if no GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024 ** 3)
        except Exception:
            pass
        return 0.0
