from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vision_tools.backends.base import Backend
from vision_tools.core.graph_types import Image
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.training.artifacts import TrainedArtifact
from vision_tools.training.dataset import VisionDataset

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

        artifact_path = self.config.get("artifact_path")
        if artifact_path:
            model_path = Path(str(artifact_path))
        else:
            fallback_model_id = self.config.get("checkpoint_id")
            model_id = self.model_resolver.resolve(
                variants=self.config.get("models", {}),
                mode=self.config.get("mode", "auto"),
                fallback=fallback_model_id,
            )
            model_path = self.model_cache.get_or_download(model_id, downloader=self._download)
        device = self.config.get("runtime", "auto")
        self.model = self.backend.load_model(str(model_path), device)
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

        model_inputs = self.preprocess(inputs, context)
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

    async def train(
        self,
        dataset: VisionDataset,
        config: dict | None = None,
    ) -> TrainedArtifact:
        config = config or {}
        if not self.supports_training():
            raise RuntimeError(f"{self.node_id}: training is not supported for this node.")
        self._state = NodeState.TRAINING
        try:
            artifact = await self._train_impl(dataset, config)
            self._state = NodeState.READY
            return artifact
        except Exception:
            self._state = NodeState.FAILED
            raise

    async def _train_impl(
        self,
        dataset: VisionDataset,
        config: dict,
    ) -> TrainedArtifact:
        backend = self.backend
        if backend is None or not hasattr(backend, "train"):
            raise NotImplementedError(
                f"{self.node_id}: training not supported. Override _train_impl() to add training."
            )

        errors = self.validate_training_dataset(dataset, config)
        if errors:
            joined = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"{self.node_id}: dataset is incompatible with training:\n{joined}")

        model_ref = str(
            self.config.get("artifact_path")
            or self.config.get("checkpoint_id")
            or ""
        )
        if not model_ref:
            raise ValueError(f"{self.node_id}: missing model reference for training.")
        return backend.train(model_ref, dataset, config, callbacks=None)

    def _download(self, model_id: str, destination: Path) -> Path:
        return destination

    def supports_training(self) -> bool:
        return bool(
            self.backend is not None
            and hasattr(self.backend, "supports_training")
            and self.backend.supports_training(self.config)
        )

    def validate_training_dataset(
        self,
        dataset: VisionDataset,
        config: dict | None = None,
    ) -> list[str]:
        if not self.backend or not hasattr(self.backend, "validate_training_dataset"):
            return ["Training backend is not available."]
        return self.backend.validate_training_dataset(dataset, config or self.config)
