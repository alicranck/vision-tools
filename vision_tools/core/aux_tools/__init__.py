"""
Auxiliary tools — non-model-based processing nodes for the VisionPipeline.

These tools perform deterministic transformations (cropping, resizing,
filtering, etc.) and do not load ML models.
"""
from .crop import CropTool

__all__ = ["CropTool"]
