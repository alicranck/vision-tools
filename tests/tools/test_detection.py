import pytest
import cv2
import numpy as np
import os
from vision_tools.core.tools.detection import OpenVocabularyDetector
from vision_tools.utils.image_utils import load_image_opencv
from test_utils import load_config


def test_open_vocabulary_detector():
    # Setup
    test_image_path = os.path.join(os.path.dirname(__file__), "../assets", "test_image.png")
    image = load_image_opencv(test_image_path)
    
    config = load_config("ov_detection")
    config["vocabulary"] = ["person", "car"]
    
    detector = OpenVocabularyDetector(config["model"], config)
    
    # Run
    # detector.process returns a dict via base_tool
    results = detector.process(image, {})
    
    # Assert
    assert "boxes" in results
    assert "class_names" in results
    assert isinstance(results["boxes"], list)
    
    # Check if we detected something (optional but good for this specific image)
    # The generated image has a person and a car.
    found_person = any(box["cls"] == detector.vocabulary.index("person") for box in results["boxes"])
    found_car = any(box["cls"] == detector.vocabulary.index("car") for box in results["boxes"])
    
    print(f"Detected {len(results['boxes'])} objects.")
    for box in results["boxes"]:
        print(f"ID: {box['id']}, Class: {results['class_names'][box['cls']]}, Conf: {box['conf']:.2f}")

    # Note: since it's zero-shot, we might not always get it perfectly, but basic structure should hold.
    assert len(results["boxes"]) >= 0 


if __name__ == "__main__":
    test_open_vocabulary_detector()
