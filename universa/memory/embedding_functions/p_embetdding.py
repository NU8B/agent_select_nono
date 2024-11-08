from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from .base_embedder import BaseEmbeddingFunction
import logging
import sys
import os


class GTEEmbeddingFn(BaseEmbeddingFunction):
    """Optimized embedding function using GTE model"""

    _shared_model = None  # Class-level model instance
    _shared_tokenizer = None  # Class-level tokenizer instance

    def __init__(self) -> None:
        """Initialize with minimal setup"""
        self._embedding_cache = {}

    def _ensure_model_loaded(self):
        """Lazy load the model only when needed, using class-level singleton"""
        if GTEEmbeddingFn._shared_model is None:
            # Set UTF-8 encoding for the environment
            os.environ["PYTHONIOENCODING"] = "utf-8"

            # Suppress all logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("tqdm").setLevel(logging.ERROR)

            with open(os.devnull, "w", encoding="utf-8") as null_file:
                old_stdout = sys.stdout
                sys.stdout = null_file
                try:
                    model_name = "BAAI/bge-large-en-v1.5"
                    GTEEmbeddingFn._shared_model = AutoModel.from_pretrained(
                        model_name, trust_remote_code=True
                    )
                    GTEEmbeddingFn._shared_tokenizer = AutoTokenizer.from_pretrained(
                        model_name
                    )

                    if torch.cuda.is_available():
                        GTEEmbeddingFn._shared_model = (
                            GTEEmbeddingFn._shared_model.half()
                        )
                        GTEEmbeddingFn._shared_model.to("cuda")
                finally:
                    sys.stdout = old_stdout

        self.model = GTEEmbeddingFn._shared_model
        self.tokenizer = GTEEmbeddingFn._shared_tokenizer

    def create_embeddings(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        """Create embeddings with lazy loading and caching"""
        try:
            self._ensure_model_loaded()

            # Convert single string to list if necessary
            if isinstance(texts, str):
                texts = [texts]

            # Tokenize and create embeddings
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Convert to list format
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()

            # Convert to list format
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()

            return embeddings.tolist()

        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise
