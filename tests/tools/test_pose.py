"""
Pose estimation tool tests — single-frame and batch.
"""


class TestPoseEstimator:
    """Test PoseEstimator with real model."""

    def test_pose_single_frame(self, pose_estimator, test_image):
        """Detect pose keypoints in test image."""
        results, did_run = pose_estimator.process(test_image, {})

        assert did_run is True
        assert "poses" in results
        assert isinstance(results["poses"], list)

        # PoseResult.model_dump() structure
        if results["poses"]:
            pose = results["poses"][0]
            assert "person_id" in pose
            assert "keypoints" in pose
            assert len(pose["keypoints"]) == 17  # COCO keypoints

            kp = pose["keypoints"][0]
            assert "x" in kp
            assert "y" in kp
            assert "confidence" in kp

        print(f"Detected {len(results['poses'])} poses")

    def test_pose_finds_person(self, pose_estimator, test_image):
        """Should detect at least one pose in the test image (has a person)."""
        results, _ = pose_estimator.process(test_image, {})
        assert len(results["poses"]) >= 1, "Expected at least one pose detection"

    def test_pose_batch(self, pose_estimator, test_image):
        """Batch of duplicated images should produce consistent results."""
        batch_results = pose_estimator.process_batch([test_image, test_image])

        assert len(batch_results) == 2
        for r in batch_results:
            assert "poses" in r

        # Same image → same pose count
        assert len(batch_results[0]["poses"]) == len(batch_results[1]["poses"])
