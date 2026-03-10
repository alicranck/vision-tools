from __future__ import annotations

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import Keypoint, PoseKeypoints, Poses
from vision_tools.nodes.model.model_node import ModelNode


class PoseEstimatorConfig(BaseModel):
    model: str = Field("yolo_pose", description="Model backend: 'yolo_pose'")
    runtime: str = Field("auto", description="Runtime: 'auto', 'pytorch', 'openvino'")
    imgsz: int = Field(640, description="Input image size")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")


class PoseEstimator(ModelNode):
    InputPorts = {"image": "Image"}
    OutputPorts = {"poses": "Poses"}

    def __init__(self, node_id: str = "pose_estimator", config: dict | None = None, **kwargs) -> None:
        config = config or {}
        model_name = config.get("model", "yolo_pose")
        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(task="pose", model=model_name)
        super().__init__(node_id=node_id, config=config, **kwargs)

    def preprocess(self, inputs, context):
        return inputs["image"].data

    def normalize_outputs(self, outputs):
        if isinstance(outputs, Poses):
            return {"poses": outputs}
        if isinstance(outputs, dict) and "poses" in outputs and isinstance(outputs["poses"], Poses):
            return outputs
        if not isinstance(outputs, dict):
            raise TypeError(f"{self.node_id}: backend output must be a dict.")
        raw_poses = outputs.get("poses", outputs.get("items", []))
        normalized = []
        for pose in raw_poses:
            if hasattr(pose, "model_dump"):
                pose = pose.model_dump()
            normalized.append(
                PoseKeypoints(
                    person_id=pose["person_id"],
                    keypoints=[
                        Keypoint.model_validate(kpt)
                        for kpt in pose.get("keypoints", [])
                    ],
                )
            )
        return {"poses": Poses(items=normalized)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return PoseEstimatorConfig
