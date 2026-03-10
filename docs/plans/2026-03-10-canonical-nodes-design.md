# Canonical Nodes Documentation & Implementation Plan

## Overview
This document specifies the design for a new set of **Canonical Nodes** in `vision-tools`. The primary goal is to provide simple, predictable, and composable building blocks that an LLM can easily string together without having to write custom logic for standard operations. This replaces tight-coupled, monolithic nodes and delegates highly custom business logic (like complex rule engines) purely to the LLM via `DynamicLogicNode`.

## Core Philosophy
- **Single Responsibility**: Each canonical node does exactly one thing well.
- **Composable**: The output of one node should seamlessly plug into another (e.g., `DetectionNode` -> `ZoneNode` -> `FilterNode` -> `BufferNode`).
- **Declarative for LLMs**: Node configurations should be simple and declarative, minimizing the need for the LLM to write "glue code" for common operations.

---

## 1. Node Portfolio Specification

Below is the spec for the new canonical nodes that will be added to `vision_tools.nodes`.

### 1.1 `FilterNode`
**Purpose**: Filters collections of items (usually detections or tracks) based on a declarative configuration.
**Inputs**: `FrameResult` (must contain a list of bounding boxes or items).
**Outputs**: `FrameResult` (with the filtered list).
**Config Parameters**:
- `min_confidence` (float): Discard detections below this threshold.
- `allowed_classes` (List[str]): Only keep detections matching these class names.
- `target_field` (str): Which field to filter on (default: `"boxes"`).

### 1.2 `ZoneNode`
**Purpose**: Determines if bounding boxes overlap with defined spatial polygons.
**Inputs**: `FrameResult` (containing detections/boxes).
**Outputs**: `FrameResult` (detections updated with a new field, e.g., `in_zones: List[str]`).
**Config Parameters**:
- `zones` (Dict[str, List[Tuple[float, float]]]): Dictionary mapping zone names to a list of (x,y) coordinates.
- `mode` (str): `"annotate"` (adds flags to existing boxes) or `"filter"` (removes boxes outside of any zone).

### 1.3 `BufferNode`
**Purpose**: Maintains a rolling window of history in memory and emits the temporal window. This fully replaces `AggregatorNode` and the state accumulation logic of `StateManager`.
**Inputs**: `FrameResult`
**Outputs**: `Dict[str, Any]` (containing a `"history"` field which is a `List[FrameResult]`).
**Config Parameters**:
- `window_size_seconds` (float): How many seconds of history to keep.
- `stride_frames` (int): Emit the accumulated history every N frames.

### 1.4 `TrackNode`
**Purpose**: Assigns stable `track_id`s to detections across frames.
**Inputs**: `FrameResult` (containing detections).
**Outputs**: `FrameResult` (detections augmented with `track_id`).
**Config Parameters**:
- `method` (str): e.g., `"iou"`, `"botsort"` (depends on the underlying tracker implementation).
- `max_age` (int): Number of frames to keep a track alive without a detection.

### 1.5 `CropNode`
**Purpose**: Extracts image crops corresponding to bounding boxes. Essential for pipelines feeding localized patches into VLMs or embedding models.
**Inputs**: `FrameResult` (containing original frame and detections).
**Outputs**: `FrameResult` (augmented with a list of base64/numpy image crops inside `.results["crops"]`).
**Config Parameters**:
- `padding` (float): Extra margin to add around the crop (relative to box size).
- `target_field` (str): The list of boxes to crop from (defaults to `"boxes"`).

---

## 2. Changes to Codebase

To bring this architecture to life, the following changes will need to be executed:

### Phase 1: Architectural Cleanup
1. **Remove `StateManager`**: Delete `vision_tools.core.state_manager` as its dual purpose (buffering + rule engine) is highly specific and monolithic.
2. **Remove `AggregatorNode`**: Delete `vision_tools.nodes.aggregator_node` as it will be fully subsumed by `BufferNode`.

### Phase 2: Implementing Canonical Nodes
1. Create `vision_tools/nodes/filter_node.py` and implement `FilterNode`.
2. Create `vision_tools/nodes/zone_node.py` and implement `ZoneNode`.
3. Create `vision_tools/nodes/buffer_node.py` and implement `BufferNode`.
4. Create `vision_tools/nodes/track_node.py` and implement simple IoU-based tracking.
5. Create `vision_tools/nodes/crop_node.py` and implement box-based image cropping.

### Phase 3: Rule Evaluation / Dynamic Logic
1. Retain `DynamicLogicNode` (`vision_tools/nodes/dynamic_logic_node.py`) in its current form.
2. The LLM will use `DynamicLogicNode` to implement custom rules (e.g., "Alert if object count > 5", "Object absent for 10s") by processing the history emitted by a `BufferNode`.

### Phase 4: Registry & Tests
1. Register all new nodes in `vision_tools.core.registry.NodeRegistry`.
2. Ensure tests are written for each new canonical node.
3. Remove test files for `StateManager` and `AggregatorNode`.

---

## Summary of the New Flow
Instead of configuring a `StateManager` with `RuleType.ZONE_ENTRY` and `RuleType.OBJECT_PRESENCE`, a pipeline will now look like this:

`DetectionNode` -> `FilterNode` (`min_confidence: 0.5`) -> `ZoneNode` (`mode: annotate`) -> `BufferNode` (`window_size_seconds: 5`) -> `DynamicLogicNode` (Runs python snippet checking if any box entered the zone in the last 5 seconds).
