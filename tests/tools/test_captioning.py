"""
Captioning tool tests.

Skipped by default since SmolVLM2 / llama.cpp are heavy for CI.
"""
import pytest


@pytest.mark.skip(reason="SmolVLM2 is too heavy for local CI")
class TestCaptioner:
    """Test Captioner with SmolVLM2 model."""

    def test_caption_single_frame(self, test_image):
        from vision_tools.core.tools.captioning import Captioner
        from tests.tools.conftest import load_config

        config = load_config("captioning")
        captioner = Captioner(config.get("model"), config)
        results, did_run = captioner.process(test_image, {})

        assert did_run is True
        assert "caption" in results
        # CaptionResult.model_dump() structure
        cap = results["caption"]
        assert "text" in cap
        assert isinstance(cap["text"], str)
        assert len(cap["text"]) > 0

        print(f"Caption: {cap['text']}")
