import chromadb
from chromadb.config import Settings
import uuid
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Abstration layer for ChromaDB to store and retrieve video frame embeddings.
    """
    def __init__(self, collection_name: str = "video_frames", persist_dir: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"VectorStore initialized with collection '{collection_name}' at '{persist_dir}'")

    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any], id: Optional[str] = None):
        """
        Adds a single embedding to the store.
        """
        if id is None:
            id = str(uuid.uuid4())
        
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[id]
        )

    def search_embeddings(self, query_embedding: List[float], n_results: int = 5, where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Searches for the nearest neighbors of the query embedding.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        return results

    def delete_embeddings(self, where: Dict[str, Any]):
        """
        Deletes embeddings based on metadata filter.
        Example: vector_store.delete_embeddings({"video_id": "123"})
        """
        self.collection.delete(where=where)

    def count(self) -> int:
        return self.collection.count()
    
    def reset(self):
        self.client.reset()
