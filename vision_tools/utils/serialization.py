import numpy as np
from ultralytics.engine.results import Boxes, Masks

def serialize_data(data: dict) -> dict:
    """
    Recursively converts complex types (numpy, Boxes) in the data dict 
    into JSON-friendly formats (lists, dicts, primitives).
    """
    out = {}
    for k, v in data.items():
        out[k] = _convert(v)
    return out

def _convert(obj):
    if isinstance(obj, Boxes):
        # Convert Ultralytics Boxes object to a list of dicts
        # xyxy: box coordinates, conf: confidence, cls: class index
        boxes_list = []
        if obj.orig_shape:
            # Ensure we have data
            for i in range(len(obj)):
                box = {
                    "xyxy": obj.xyxy[i].tolist(),
                    "conf": float(obj.conf[i]),
                    "cls": int(obj.cls[i])
                }
                boxes_list.append(box)
        return boxes_list

    elif isinstance(obj, Masks):
        # Convert Ultralytics Masks object to a list of polygons
        # xy: list of segments (each segment is an array of [x, y] points)
        masks_list = []
        if obj.xy:
             for segment in obj.xy:
                 # segment is a numpy array of shape (N, 2)
                 masks_list.append(segment.tolist())
        return masks_list
    
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
        
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
        
    elif isinstance(obj, list):
        return [_convert(i) for i in obj]
        
    elif isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
        
    return obj
