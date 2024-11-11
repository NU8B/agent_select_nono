from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from ..embedding_functions.Embedding_AutoModel import EmbeddingFn
from ..embedding_functions.base_embedder import BaseEmbeddingFunction


class ChromaDB:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "chroma_db",
        embedding_function: Optional[BaseEmbeddingFunction] = None,
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function or EmbeddingFn.get_instance()

        # Initialize client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )

        # Initialize or reset collection
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Initialize collection with proper dimension handling"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name, embedding_function=self.embedding_function
            )

            # Verify dimensions match
            sample_embedding = self.embedding_function.create_embeddings(["test"])[0]
            if self.get_count() > 0:
                existing_dim = len(self.collection.peek()["embeddings"][0])
                if len(sample_embedding) != existing_dim:
                    print(
                        f"Embedding dimensions changed ({existing_dim} -> {len(sample_embedding)}). Resetting collection..."
                    )
                    self.client.delete_collection(self.collection_name)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function,
                    )

        except ValueError:
            # Collection doesn't exist, create new one
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )

    def add_data(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add new data to collection"""
        existing_ids = (
            set(self.collection.get()["ids"]) if self.get_count() > 0 else set()
        )

        # Filter out existing documents
        new_docs = []
        new_ids = []
        new_metadatas = []

        for i, doc_id in enumerate(ids):
            if doc_id not in existing_ids:
                new_docs.append(documents[i])
                new_ids.append(doc_id)
                new_metadatas.append(
                    metadatas[i] if metadatas else {"source": "agent_description"}
                )

        if new_docs:
            self.collection.add(
                documents=new_docs,
                ids=new_ids,
                metadatas=new_metadatas,
            )
            print(f"Added {len(new_docs)} new documents to collection")

    def query_data(
        self,
        query_text: List[str],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the collection"""
        return self.collection.query(
            query_texts=query_text,
            n_results=n_results,
            where=where,
        )

    def get_count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
