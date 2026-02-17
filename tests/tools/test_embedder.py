"""
Embedder tool tests — single-frame and batch.
"""
import numpy as np


class TestSigLIP2Embedder:
    """Test SigLIP2 embedding generation."""

    def test_embedding_single_frame(self, embedder, test_image):
        """Generate embedding for test image."""
        results, did_run = embedder.process(test_image, {})

        assert did_run is True
        assert "embedding" in results

        # EmbeddingResult.model_dump() structure
        emb = results["embedding"]
        assert "vector" in emb
        assert "model_id" in emb
        assert "dimension" in emb
        assert isinstance(emb["vector"], list)
        assert emb["dimension"] == len(emb["vector"])
        assert emb["dimension"] > 0

        print(f"Generated embedding: dim={emb['dimension']}, model={emb['model_id']}")

    def test_embedding_normalized(self, embedder, test_image):
        """Embeddings should be L2-normalized."""
        results, _ = embedder.process(test_image, {})
        vec = np.array(results["embedding"]["vector"])
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-2), f"Embedding norm={norm}, expected ~1.0"

    def test_embedding_batch(self, embedder, test_image):
        """Batch of identical images should produce identical embeddings."""
        batch_results = embedder.process_batch([test_image, test_image])

        assert len(batch_results) == 2
        for r in batch_results:
            assert "embedding" in r
            assert isinstance(r["embedding"]["vector"], list)

        # Same image → same embedding
        v1 = np.array(batch_results[0]["embedding"]["vector"])
        v2 = np.array(batch_results[1]["embedding"]["vector"])
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert cos_sim > 0.99, f"Expected identical embeddings, got cos_sim={cos_sim:.4f}"

    def test_text_encoding(self, embedder):
        """Text encoding should produce a valid embedding vector."""
        vec = embedder.encode_text("a person standing next to a car")
        assert isinstance(vec, (list, np.ndarray))
        vec_np = np.array(vec)
        assert vec_np.ndim == 1
        assert len(vec_np) > 0
