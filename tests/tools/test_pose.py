import pytest
import cv2
import numpy as np
import os
from vision_tools.core.tools.pose_estimation import PoseEstimator
from vision_tools.utils.image_utils import load_image_opencv
from test_utils import load_config


def test_pose_estimator():
    # Setup
    test_image_path = os.path.join(os.path.dirname(__file__), "../assets", "test_image.png")
    image = load_image_opencv(test_image_path)

    config = load_config("pose_estimation")
    
    estimator = PoseEstimator(config["model"], config)
    
    # Run
    results, did_run = estimator.process(image, {})
    
    # Assert
    assert "poses" in results
    assert isinstance(results["poses"], list)
    
    if len(results["poses"]) > 0:
        first_pose = results["poses"][0]
        assert "keypoints" in first_pose
        assert len(first_pose["keypoints"]) == 17 # standard COCO pose
        
    print(f"Detected {len(results['poses'])} poses.")

if __name__ == "__main__":
    test_pose_estimator()
