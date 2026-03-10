from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vision_tools.backends.base import Backend
from vision_tools.core.graph_types import Image
from vision_tools.core.node import Node, NodeContext, NodeState

logger = logging.getLogger(__name__)


def _default_model_resolver():
    from vision_tools.runtime.model_resolver import ModelResolver

    return ModelResolver()


def _default_model_cache():
    from vision_tools.runtime.model_cache import ModelCache

    return ModelCache()


class ModelNode(Node):
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

    def load(self) -> None:
        if self._state == NodeState.READY:
            return

        if self.backend is None:
            raise RuntimeError(
                f"{self.node_id}: no backend set. Task node must resolve a backend before loading."
            )

        fallback_model_id = self.config.get("checkpoint_id") or self.config.get("model")
        model_id = self.model_resolver.resolve(
            variants=self.config.get("models", {}),
            mode=self.config.get("mode", "auto"),
            fallback=fallback_model_id,
        )
        model_path = self.model_cache.get_or_download(model_id, downloader=self._download)
        device = self.config.get("runtime", "auto")
        self.model = self.backend.load_model(model_path, device)
        self._state = NodeState.READY

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self._state = NodeState.UNLOADED

    def warmup(self, rounds: int = 1) -> None:
        if self._state != NodeState.READY:
            return

        try:
            import numpy as np
        except ImportError:  # pragma: no cover - environment dependent
            logger.warning("%s: numpy not available, skipping warmup.", self.node_id)
            return

        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        image = Image(data=dummy, width=32, height=32, channels=3)
        context = NodeContext(frame_shape=(32, 32, 3))
        for _ in range(rounds):
            self.process({"image": image}, context)

    def verify(self) -> bool:
        if self._state != NodeState.READY:
            return False

        try:
            self.warmup(rounds=1)
            return True
        except Exception as exc:  # pragma: no cover - verification failure path
            logger.error("%s: verification failed: %s", self.node_id, exc)
            self._state = NodeState.FAILED
            return False

    def process(self, inputs: Any, context: NodeContext) -> dict[str, Any]:
        if self._state != NodeState.READY:
            raise RuntimeError(
                f"{self.node_id}: cannot process in state {self._state}. Call load() first."
            )

        if not isinstance(inputs, dict):
            if len(self.get_input_ports()) != 1:
                raise TypeError(f"{self.node_id}: expected a dict of named inputs.")
            inputs = {next(iter(self.get_input_ports())): inputs}

        validated_inputs = self.validate_inputs(inputs)
        model_inputs = self.preprocess(validated_inputs, context)
        if hasattr(self.backend, "preprocess"):
            model_inputs = self.backend.preprocess(model_inputs)

        raw_output = self.backend.infer(self.model, model_inputs)
        outputs = self.backend.postprocess(raw_output, context)
        normalized = self.normalize_outputs(outputs)
        return self.validate_outputs(normalized)

    def preprocess(self, inputs: dict[str, Any], context: NodeContext) -> Any:
        return inputs

    def normalize_outputs(self, outputs: Any) -> dict[str, Any]:
        if not isinstance(outputs, dict):
            raise TypeError(f"{self.node_id}: backend output must be a dict.")
        return outputs

    async def train(self, dataset_ref: str, config: dict | None = None) -> None:
        config = config or {}
        self._state = NodeState.TRAINING
        try:
            await self._train_impl(dataset_ref, config)
            self._state = NodeState.READY
        except Exception:
            self._state = NodeState.FAILED
            raise

    async def _train_impl(self, dataset_ref: str, config: dict) -> None:
        raise NotImplementedError(
            f"{self.node_id}: training not supported. Override _train_impl() to add training."
        )

    def _download(self, model_id: str, destination: Path) -> Path:
        return destination
