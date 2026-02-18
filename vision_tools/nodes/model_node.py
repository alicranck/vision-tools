"""
ModelNode — Base class for nodes that wrap ML models.

ModelNode extends Node with:
- **Lifecycle management**: load → warmup → verify → process → unload
- **Backend delegation**: model loading, inference, and postprocessing are
  handled by an injected Backend, not by the node itself.
- **Runtime services**: ModelResolver for variant selection,
  ModelCache for download/caching.
- **Training hooks**: async train/fine-tune interface.
- **Schema enforcement**: output is validated against OutputSchema on every call.

Task nodes (ObjectDetector, Embedder, etc.) inherit from ModelNode
and set their OutputSchema + resolve their Backend from config.
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vision_tools.backends.base import Backend
from vision_tools.core.node import Node, NodeContext, NodeState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults for optional services
# ---------------------------------------------------------------------------

def _default_model_resolver():
    """Lazy import to avoid circular deps."""
    from vision_tools.runtime.model_resolver import ModelResolver
    return ModelResolver()


def _default_model_cache():
    """Lazy import to avoid circular deps."""
    from vision_tools.runtime.model_cache import ModelCache
    return ModelCache()


# ---------------------------------------------------------------------------
# ModelNode
# ---------------------------------------------------------------------------

class ModelNode(Node):
    """Base class for nodes that wrap an ML model.

    Subclasses (task nodes) must:
    1. Set ``OutputSchema`` to a Pydantic model.
    2. Resolve and pass a ``Backend`` instance to ``__init__``.
    3. Optionally override ``preprocess()`` for task-specific input transforms.

    The ``Backend`` handles model-specific operations (loading, inference,
    postprocessing), while ``ModelNode`` handles lifecycle, validation,
    and service orchestration.
    """

    def __init__(
        self,
        node_id: str,
        config: dict[str, Any] | None = None,
        backend: Backend | None = None,
        model_cache=None,
        model_resolver=None,
    ) -> None:
        super().__init__(node_id, config)
        self._state = NodeState.UNLOADED
        self.model: Any = None
        self.backend = backend
        self.model_cache = model_cache or _default_model_cache()
        self.model_resolver = model_resolver or _default_model_resolver()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Resolve model variant, download if needed, load via backend.

        After loading, state transitions to READY.
        """
        if self._state == NodeState.READY:
            logger.info(f"{self.node_id}: already loaded, skipping.")
            return

        if self.backend is None:
            raise RuntimeError(
                f"{self.node_id}: no backend set. "
                "Task node must resolve a backend before loading."
            )

        # Resolve which model variant to use
        model_id = self.model_resolver.resolve(
            variants=self.config.get("models", {}),
            mode=self.config.get("mode", "auto"),
            fallback=self.config.get("model"),
        )

        # Download if needed
        model_path = self.model_cache.get_or_download(
            model_id, downloader=self._download
        )

        # Load via backend
        device = self.config.get("runtime", "auto")
        logger.info(
            f"{self.node_id}: loading model={model_id} "
            f"device={device} backend={self.backend.__class__.__name__}"
        )
        self.model = self.backend.load_model(model_path, device)
        self._state = NodeState.READY
        logger.info(f"{self.node_id}: loaded successfully.")

    def unload(self) -> None:
        """Release model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._state = NodeState.UNLOADED
        logger.info(f"{self.node_id}: unloaded.")

    def warmup(self, rounds: int = 4) -> None:
        """Run dummy inference rounds to warm caches and JIT.

        Args:
            rounds: Number of warmup iterations.
        """
        if self._state != NodeState.READY:
            logger.warning(f"{self.node_id}: cannot warmup in state {self._state}")
            return

        logger.info(f"{self.node_id}: warming up ({rounds} rounds)...")
        try:
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            ctx = NodeContext()
            for _ in range(rounds):
                inputs = self.preprocess(dummy, ctx)
                self.backend.infer(self.model, inputs)
            logger.info(f"{self.node_id}: warmup complete.")
        except Exception as e:
            logger.warning(f"{self.node_id}: warmup failed: {e}")

    def verify(self) -> bool:
        """Run dummy inference and validate output against OutputSchema.

        Returns:
            True if verification passes, False otherwise.
        """
        if self._state != NodeState.READY:
            logger.warning(f"{self.node_id}: cannot verify in state {self._state}")
            return False

        try:
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            ctx = NodeContext()
            result = self.process(dummy, ctx)

            if self.OutputSchema is not None:
                self.OutputSchema.model_validate(result)

            logger.info(f"{self.node_id}: verification passed.")
            return True
        except Exception as e:
            logger.error(f"{self.node_id}: verification failed: {e}")
            self._state = NodeState.FAILED
            return False

    # ------------------------------------------------------------------
    # Processing: delegates to backend
    # ------------------------------------------------------------------

    def process(self, data: Any, context: NodeContext) -> Any:
        """Full processing pipeline: preprocess → infer → postprocess → validate.

        Args:
            data: Input data (typically a numpy frame or dict matching InputSchema).
            context: Runtime context with frame metadata and upstream results.

        Returns:
            Dict validated against OutputSchema (if set).

        Raises:
            RuntimeError: If not in READY state.
            ValidationError: If output doesn't match OutputSchema.
        """
        if self._state != NodeState.READY:
            raise RuntimeError(
                f"{self.node_id}: cannot process in state {self._state}. "
                "Call load() first."
            )

        inputs = self.preprocess(data, context)
        raw_output = self.backend.infer(self.model, inputs)
        result = self.backend.postprocess(raw_output, context)

        # Enforce output schema
        if self.OutputSchema is not None:
            validated = self.OutputSchema.model_validate(result)
            return validated.model_dump()

        return result

    def preprocess(self, data: Any, context: NodeContext) -> Any:
        """Task-specific preprocessing. Override in task nodes.

        Default: pass through data unchanged.
        """
        return data

    # ------------------------------------------------------------------
    # Training (optional)
    # ------------------------------------------------------------------

    async def train(self, dataset_ref: str, config: dict | None = None) -> None:
        """Fine-tune the model. Manages state transitions.

        Args:
            dataset_ref: Path or reference to training dataset.
            config: Training hyperparameters.
        """
        config = config or {}
        self._state = NodeState.TRAINING
        logger.info(f"{self.node_id}: starting training on {dataset_ref}")

        try:
            await self._train_impl(dataset_ref, config)
            self._state = NodeState.READY
            logger.info(f"{self.node_id}: training complete.")
        except Exception as e:
            self._state = NodeState.FAILED
            logger.error(f"{self.node_id}: training failed: {e}")
            raise

    async def _train_impl(self, dataset_ref: str, config: dict) -> None:
        """Override in trainable task nodes."""
        raise NotImplementedError(
            f"{self.node_id}: training not supported. "
            "Override _train_impl() to add training."
        )

    # ------------------------------------------------------------------
    # Model download hook
    # ------------------------------------------------------------------

    def _download(self, model_id: str, destination: Path) -> Path:
        """Download model to destination. Override if needed.

        Default: returns model_id as-is (for library-managed models).
        """
        return Path(model_id)
