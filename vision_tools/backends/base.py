"""
Backend Protocol — defines how model-specific operations are performed.

Backends are the "how" in vision-tools: they handle model loading,
raw inference, and output postprocessing. Task nodes (the "what")
delegate to backends so that the same task node (e.g., ObjectDetector)
can use different model families (YOLO detector, RT-DETR) and runtimes
(PyTorch, OpenVINO, ONNX) without any code changes.

Example::

    @BackendRegistry.register(task="detection", model="yolo_detector")
    class YoloDetectorBackend:
        def load_model(self, model_path, device="auto"):
            return YOLO(model_path)

        def infer(self, model, inputs):
            return model.track(inputs, persist=True)

        def postprocess(self, raw_output, context):
            ...  # Convert YOLO results → DetectionResult dict

        def get_available_runtimes(self):
            return ["pytorch", "openvino", "onnx"]
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vision_tools.core.node import NodeContext
from vision_tools.training.artifacts import TrainedArtifact
from vision_tools.training.dataset import VisionDataset


@runtime_checkable
class Backend(Protocol):
    """Protocol that all model backends must satisfy.

    Backends encapsulate model-specific operations:
    - **load_model**: Load a model from disk into memory.
    - **infer**: Run raw model inference.
    - **postprocess**: Convert raw model output into typed port-compatible values.
    - **get_available_runtimes**: Report supported runtimes for introspection.

    Backends are *implementation details* — the pipeline never interacts
    with them directly. Only ``ModelNode`` delegates to backends.
    """

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        """Load a model from the given path onto the specified device.

        Args:
            model_path: Path to the model file or identifier.
            device: Target device/runtime — "auto", "cpu", "cuda",
                    "openvino", "onnx".

        Returns:
            The loaded model object (framework-specific).
        """
        ...

    def infer(self, model: Any, inputs: Any) -> Any:
        """Run raw inference.

        Args:
            model: The loaded model object (from ``load_model``).
            inputs: Preprocessed input data.

        Returns:
            Raw model output (framework-specific).
        """
        ...

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        """Convert raw model output into a typed output-port dict.

        Args:
            raw_output: Raw output from ``infer``.
            context: Runtime context with frame metadata.

        Returns:
            Dict that validates against the task node's declared output ports.
        """
        ...

    def get_available_runtimes(self) -> list[str]:
        """Return list of supported runtime identifiers.

        Returns:
            E.g. ``["pytorch", "openvino", "onnx"]``.
        """
        ...

    def supports_training(self, config: dict[str, Any] | None = None) -> bool:
        """Return whether this backend supports fine-tuning for the given config."""
        ...

    def validate_training_dataset(
        self,
        dataset: VisionDataset,
        config: dict[str, Any] | None = None,
    ) -> list[str]:
        """Validate whether a VisionDataset is compatible with backend training."""
        ...

    def train(
        self,
        model_ref: str,
        dataset: VisionDataset,
        config: dict[str, Any] | None = None,
        callbacks: dict[str, Any] | None = None,
    ) -> TrainedArtifact:
        """Run backend-specific training and return the produced artifact metadata."""
        ...
