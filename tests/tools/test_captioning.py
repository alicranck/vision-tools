import pytest
import os
import numpy as np
from vision_tools.core.tools.captioning import Captioner
from vision_tools.utils.image_utils import load_image_opencv
from test_utils import load_config


@pytest.mark.skip(reason="SmolVLM2 can be very heavy for CI/CD environments")
def test_captioner_smolvlm():
    # Setup
    test_image_path = os.path.join(os.path.dirname(__file__), "../assets", "test_image.png")
    image = load_image_opencv(test_image_path)
    
    config = load_config("captioning")
    
    captioner = Captioner(config["model"], config)
    
    # Run
    results, did_run = captioner.process(image, {})
    
    # Assert
    assert "caption" in results
    assert isinstance(results["caption"], str)
    assert len(results["caption"]) > 0
    
    print(f"Generated caption: {results['caption']}")

if __name__ == "__main__":
    test_captioner_smolvlm()
