import clip
import torch
from typing import List, Dict, Any
from vision_core.data.vector_store import VectorStore

class TextEmbedder:
    _instance = None
    _model = None
    _device = 'cpu'

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, model_name="ViT-B/32", device="cpu"):
        if TextEmbedder._model is None:
            model, _ = clip.load(model_name, device=device)
            TextEmbedder._model = model
            TextEmbedder._device = device

    def encode_text(self, text: str) -> List[float]:
        text_tokens = clip.tokenize([text]).to(TextEmbedder._device)
        with torch.no_grad():
            text_features = TextEmbedder._model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0].tolist()

def search_videos(query: str, vector_store: VectorStore, limit: int = 5) -> List[Dict[str, Any]]:
    embedder = TextEmbedder.get_instance()
    query_vector = embedder.encode_text(query)
    
    results = vector_store.search_embeddings(query_vector, n_results=limit)

    formatted_results = []
    if results['ids']:
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i]
            })
            
    return formatted_results
