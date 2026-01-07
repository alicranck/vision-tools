import os
import numpy as np
from vision_tools.core.tools.embedder import CLIPEmbedder, SigLIP2Embedder
from vision_tools.utils.image_utils import load_image_opencv
from .test_utils import load_config


def test_clip_embedder():
    # Setup
    test_image_path = os.path.join(os.path.dirname(__file__), "../assets", "test_image.png")
    image = load_image_opencv(test_image_path)
    
    config = load_config("embedding")
    
    embedder = CLIPEmbedder(config["model"], config)
    
    # Run
    results, did_run = embedder.process(image, {})
    
    # Assert
    assert "embedding" in results
    assert isinstance(results["embedding"], list)
    assert len(results["embedding"]) == 512 # ViT-B/32 embedding dimension
    
    # Check normalization
    embedding_np = np.array(results["embedding"])
    norm = np.linalg.norm(embedding_np)
    assert np.isclose(norm, 1.0, atol=1e-3)
    
    print(f"Generated embedding of size {len(results['embedding'])}.")


def test_siglip2_embedder():
    # Setup
    test_image_path = os.path.join(os.path.dirname(__file__), "../assets", "test_image.png")
    image = load_image_opencv(test_image_path)
    
    config = load_config("clip_embedding")
    
    embedder = SigLIP2Embedder(config["model"], config)
    
    # Run
    results, did_run = embedder.process(image, {})
    
    # Assert
    assert "embedding" in results
    assert isinstance(results["embedding"], list)
    assert len(results["embedding"]) == 512 # ViT-B/32 embedding dimension
    
    # Check normalization
    embedding_np = np.array(results["embedding"])
    norm = np.linalg.norm(embedding_np)
    assert np.isclose(norm, 1.0, atol=1e-3)
    
    print(f"Generated embedding of size {len(results['embedding'])}.")


if __name__ == "__main__":
    test_clip_embedder()
    test_siglip2_embedder()
