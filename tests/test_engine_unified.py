import numpy as np

from vision_tools.core.graph_types import Alerts
from vision_tools.engine.video_engine import VideoInferenceEngine
from vision_tools.utils.schemas import BatchPayload, FrameMetadata


class _V2PipelineStub:
    def run_batch(self, frames):
        return [{"det": {"boxes": []}} for _ in frames]


class _LegacyPipelineStub:
    def process_batch_payload(self, payload):
        payload.frame_results = []
        for md in payload.frame_metadatas:
            payload.frame_results.append(
                {
                    "metadata": md,
                    "results": {"legacy": True},
                }
            )
        return payload


def _build_payload():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]
    metadatas = [FrameMetadata(frame_idx=0), FrameMetadata(frame_idx=1)]
    return BatchPayload(batch_id="b1", frames=frames, frame_metadatas=metadatas)


def test_engine_process_batch_v2_pipeline_adapts_to_frame_results():
    engine = VideoInferenceEngine(_V2PipelineStub(), video_path="dummy.mp4")
    payload = _build_payload()
    processed = engine._process_batch(payload)

    assert len(processed.frame_results) == 2
    assert processed.frame_results[0].results["det"]["boxes"] == []
    assert processed.frame_results[0].tools_run is True


def test_engine_process_batch_legacy_pipeline_passthrough():
    class LegacyPayloadStub:
        def process_batch_payload(self, payload):
            return payload

    engine = VideoInferenceEngine(LegacyPayloadStub(), video_path="dummy.mp4")
    payload = _build_payload()
    processed = engine._process_batch(payload)
    assert processed is payload


def test_engine_process_batch_serializes_typed_outputs():
    class _TypedPipelineStub:
        def run_batch(self, frames):
            return [{"alerts": Alerts(items=[])} for _ in frames]

    engine = VideoInferenceEngine(_TypedPipelineStub(), video_path="dummy.mp4")
    payload = _build_payload()
    processed = engine._process_batch(payload)
    assert processed.frame_results[0].results["alerts"] == {"items": []}
