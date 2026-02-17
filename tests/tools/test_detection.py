"""
Detection tool tests — single-frame and batch.
"""


class TestDetection:
    """Test OpenVocabularyDetector with real model."""

    def test_detection_single_frame(self, detector, test_image):
        """Basic detection on test image (person + car scene)."""
        results, did_run = detector.process(test_image, {})

        assert did_run is True
        assert "boxes" in results
        assert "class_names" in results
        assert isinstance(results["boxes"], list)

        # Verify box structure matches DetectionResult.model_dump()
        if results["boxes"]:
            box = results["boxes"][0]
            assert "xyxy" in box
            assert "class_id" in box
            assert "confidence" in box
            assert len(box["xyxy"]) == 4

        print(f"Detected {len(results['boxes'])} objects")
        for box in results["boxes"]:
            cname = box.get("class_name", results["class_names"].get(box["class_id"]))
            print(f"  {cname}: conf={box['confidence']:.2f}, xyxy={box['xyxy']}")

    def test_detection_finds_person(self, detector, test_image):
        """Should detect at least one person in the test image."""
        results, _ = detector.process(test_image, {})
        person_boxes = [b for b in results["boxes"] if b.get("class_name") == "person"]
        assert len(person_boxes) >= 1, "Expected at least one person detection"

    def test_detection_batch(self, detector, test_image):
        """Batch of duplicated images should produce consistent results."""
        batch = [test_image, test_image]
        batch_results = detector.process_batch(batch)

        assert len(batch_results) == 2
        for result in batch_results:
            assert "boxes" in result
            assert isinstance(result["boxes"], list)

        # Both frames are identical, so detection counts should match
        assert len(batch_results[0]["boxes"]) == len(batch_results[1]["boxes"])
