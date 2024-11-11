from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from ..embedding_functions.Embedding_AutoModel import EmbeddingFn
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
        # Store the collection name and set the embedding function
        self.collection_name = collection_name

        # Initialize client with allow_reset=True
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )

        # Use provided embedding function or get default one
        self.embedding_function = embedding_function or self.get_embedding_function()

        # Check if dimensions match
        try:
            existing_collection = self.client.get_collection(collection_name)
            sample_embedding = self.embedding_function.create_embeddings(["test"])[0]

            if len(sample_embedding) != len(
                existing_collection.peek()["embeddings"][0]
            ):
                print(f"Model dimensions changed. Resetting collection...")
                self.client.reset()
                self.client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(allow_reset=True, anonymized_telemetry=False),
                )
        except Exception:
            pass

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

    def reset_collection(self, collection_name: str) -> None:
        """Reset specific collection"""
        try:
            self.client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")

    def add_data(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
<<<<<<< Updated upstream
        """Add data to collection if not already present"""
        try:
            # Get existing IDs or empty set if collection is empty
            try:
                existing_ids = set(self.collection.get()["ids"])
            except Exception:
                existing_ids = set()

            # Filter out documents that already exist
            new_docs = []
            new_ids = []
            new_metadatas = []

            for i, doc_id in enumerate(ids):
                if doc_id not in existing_ids:
                    new_docs.append(documents[i])
                    new_ids.append(doc_id)
                    if metadatas:
                        new_metadatas.append(metadatas[i])
                    else:
                        # Add default metadata for each document if none provided
                        new_metadatas.append({"source": "agent_description"})

            if new_docs:
                self.collection.add(
                    documents=new_docs,
                    ids=new_ids,
                    metadatas=new_metadatas,
                    embeddings=embeddings,
=======
        """Add new data to collection"""
        # Get existing document IDs
        existing_ids = (
            set(self.collection.get()["ids"]) if self.get_count() > 0 else set()
        )

        # Prepare lists
        new_docs = []
        new_ids = []
        new_metadatas = []

        # Filter out existing documents
        for i, doc_id in enumerate(ids):
            if doc_id not in existing_ids:
                new_docs.append(documents[i])
                new_ids.append(doc_id)
                new_metadatas.append(
                    metadatas[i] if metadatas else {"source": "agent_description"}
>>>>>>> Stashed changes
                )
                print(f"Added {len(new_docs)} new documents to collection")
            else:
                print("All documents already exist in collection")

<<<<<<< Updated upstream
        except Exception as e:
            print(f"Error adding data to collection: {e}")
            raise
=======
        # Add new documents to collection if there are any
        if new_docs:
            self.collection.add(
                documents=new_docs,
                ids=new_ids,
                metadatas=new_metadatas,
            )
            print(f"Added {len(new_docs)} new documents to collection")
>>>>>>> Stashed changes

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
