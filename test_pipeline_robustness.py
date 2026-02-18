
import logging
import sys
import numpy as np
from unittest.mock import MagicMock, patch
from vision_tools.core.tools.pipeline import VisionPipeline, PipelineConfig, PipelineNode

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def test_pipeline_robustness():
    # Define a simple pipeline config
    nodes = [
        PipelineNode(
            node_id="detector",
            tool_type="ov_detection",
            tool_config={
                "model": "yolov8n.pt", 
                "vocabulary": ["person"],
            }
        )
    ]
    
    config = PipelineConfig(nodes=nodes)
    
    with patch("vision_tools.core.tools.detection.OpenVocabularyDetector._load_model") as mock_load, \
         patch("vision_tools.core.tools.detection.OpenVocabularyDetector.inference") as mock_inference, \
         patch("vision_tools.core.tools.detection.OpenVocabularyDetector.download_ckpt") as mock_download:
        
        mock_load.return_value = MagicMock()
        
        # Test 1: Successful verification
        # Return valid inference result
        mock_inference.return_value = {
            "tracks": {}, 
            "class_names": {0: "person"}
        }

        try:
            print("Test 1: Initializing pipeline with valid mocks (expecting success)...")
            pipeline = VisionPipeline(config)
            print("SUCCESS: Pipeline initialized.")
        except Exception as e:
            print(f"FAILED: Pipeline initialization failed unexpectedly: {e}")
            raise

        # Test 2: Failed verification
        print("\nTest 2: Testing validation failure with INVALID types...")
        
        # We patch postprocess to return a list of boxes that are NOT BoundingBox objects
        # This MUST fail schema validation because strings are not BoundingBoxes.
        with patch("vision_tools.core.tools.detection.OpenVocabularyDetector.postprocess") as mock_post:
             mock_post.return_value = {"boxes": "this_is_not_a_list_of_boxes"}
             
             try:
                 pipeline = VisionPipeline(config)
                 print("FAILED: Pipeline initialized but should have caught the invalid type!")
             except RuntimeError as e:
                 print(f"SUCCESS: Caught expected verification error: {e}")
             except Exception as e:
                 print(f"FAILED: Caught wrong error type: {type(e)}: {e}")

if __name__ == "__main__":
    test_pipeline_robustness()
