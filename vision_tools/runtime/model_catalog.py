from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vision_tools.core.config import DeviceTarget, InferenceTask, ModelIntent, ModelSize


class ResolvedModelSpec(BaseModel):
    """Concrete model selection resolved internally by vision-tools."""
    backend_task: str
    backend_model: str
    checkpoint_id: str
    runtime: str
    model_config = ConfigDict(extra="forbid")


class _CatalogEntry(BaseModel):
    checkpoint_id: str
    runtime_by_device: dict[DeviceTarget, str] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ModelCatalog:
    """Static model catalog used to resolve app/LLM intent to concrete models."""

    _CATALOG: dict[InferenceTask, dict[str, dict[ModelSize, _CatalogEntry]]] = {
        InferenceTask.DETECTION: {
            "yolo_detector": {
                ModelSize.SMALL: _CatalogEntry(
                    checkpoint_id="yolo11s.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "openvino",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.MEDIUM: _CatalogEntry(
                    checkpoint_id="yolo11m.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "openvino",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.LARGE: _CatalogEntry(
                    checkpoint_id="yolo11l.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "openvino",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
            },
            "rtdetr": {
                ModelSize.SMALL: _CatalogEntry(
                    checkpoint_id="rtdetr-l.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.MEDIUM: _CatalogEntry(
                    checkpoint_id="rtdetr-x.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.LARGE: _CatalogEntry(
                    checkpoint_id="rtdetr-x.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
            },
        },
        InferenceTask.OPEN_VOCAB_DETECTION: {
            "yolo": {
                ModelSize.SMALL: _CatalogEntry(
                    checkpoint_id="yoloe-11s-seg.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "openvino",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.MEDIUM: _CatalogEntry(
                    checkpoint_id="yoloe-11m-seg.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "openvino",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.LARGE: _CatalogEntry(
                    checkpoint_id="yoloe-11l-seg.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "openvino",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
            },
        },
        InferenceTask.CLASSIFICATION: {
            "yolo_cls": {
                ModelSize.SMALL: _CatalogEntry(
                    checkpoint_id="yolo11n-cls.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.MEDIUM: _CatalogEntry(
                    checkpoint_id="yolo11s-cls.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.LARGE: _CatalogEntry(
                    checkpoint_id="yolo11m-cls.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
            },
        },
        InferenceTask.SEGMENTATION: {
            "yolo_seg": {
                ModelSize.SMALL: _CatalogEntry(
                    checkpoint_id="yolo11n-seg.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.MEDIUM: _CatalogEntry(
                    checkpoint_id="yolo11s-seg.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
                ModelSize.LARGE: _CatalogEntry(
                    checkpoint_id="yolo11m-seg.pt",
                    runtime_by_device={
                        DeviceTarget.AUTO: "auto",
                        DeviceTarget.CPU: "cpu",
                        DeviceTarget.GPU: "pytorch",
                    },
                ),
            },
        },
    }
    _SIZE_ORDER: dict[ModelSize, int] = {
        ModelSize.SMALL: 0,
        ModelSize.MEDIUM: 1,
        ModelSize.LARGE: 2,
    }

    @classmethod
    def resolve(cls, intent: ModelIntent) -> ResolvedModelSpec:
        by_family = cls._CATALOG.get(intent.task)
        if not by_family:
            raise ValueError(f"No model catalog entries for task={intent.task.value!r}")

        by_size = by_family.get(intent.model_family)
        if not by_size:
            supported = sorted(by_family.keys())
            raise ValueError(
                f"Unsupported model_family={intent.model_family!r} for task={intent.task.value!r}. "
                f"Supported: {supported}"
            )

        entry = by_size.get(intent.size)
        if not entry:
            supported = sorted(size.value for size in by_size.keys())
            raise ValueError(
                f"Unsupported size={intent.size.value!r} for "
                f"task={intent.task.value!r}, model_family={intent.model_family!r}. "
                f"Supported: {supported}"
            )

        runtime = entry.runtime_by_device.get(intent.device)
        if runtime is None:
            runtime = entry.runtime_by_device.get(DeviceTarget.AUTO, "auto")

        backend_task = "detection" if intent.task == InferenceTask.OPEN_VOCAB_DETECTION else intent.task.value
        return ResolvedModelSpec(
            backend_task=backend_task,
            backend_model=intent.model_family,
            checkpoint_id=entry.checkpoint_id,
            runtime=runtime,
        )

    @classmethod
    def list_options(cls, task: InferenceTask | None = None) -> dict[str, Any]:
        tasks = [task] if task else list(cls._CATALOG.keys())
        output: dict[str, Any] = {}
        for task_key in tasks:
            families = cls._CATALOG.get(task_key, {})
            output[task_key.value] = {
                family: {
                    "sizes": [
                        size.value
                        for size in sorted(by_size.keys(), key=lambda s: cls._SIZE_ORDER.get(s, 999))
                    ],
                    "devices": sorted(
                        {
                            device.value
                            for entry in by_size.values()
                            for device in entry.runtime_by_device.keys()
                        }
                    ),
                }
                for family, by_size in families.items()
            }
        return output
