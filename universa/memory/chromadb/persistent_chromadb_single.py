from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from ..embedding_functions.Embedding_AutoModel_single import EmbeddingFn
from ..embedding_functions.base_embedder import BaseEmbeddingFunction
from functools import lru_cache


class ChromaDB:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_embedding_function():
        """Cache the embedding function to avoid reloading the model"""
        return EmbeddingFn()

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "chroma_db",
        embedding_function: Optional[BaseEmbeddingFunction] = None,
    ):
        self.collection_name = collection_name

        # Initialize client with allow_reset=True
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )

        # Use provided embedding function or get default one
        self.embedding_function = embedding_function or self.get_embedding_function()

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )

    def add_data(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Add data to collection"""
        try:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas or [{"source": "agent_description"} for _ in ids],
                embeddings=embeddings,
            )
        except Exception as e:
            print(f"Error adding data to collection: {e}")
            raise

    def query_data(
        self,
        query_text: List[str],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the collection"""
        try:
            return self.collection.query(
                query_texts=query_text, n_results=n_results, where=where
            )
        except Exception as e:
            print(f"Error querying collection: {e}")
            raise

    def get_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting collection count: {e}")
            return 0
