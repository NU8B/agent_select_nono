from typing import List
import torch
from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbeddingFunction
import logging
import sys
import os
import warnings


class EmbeddingFn(BaseEmbeddingFunction):
    """Optimized embedding function using Sentence Transformer model"""

    _shared_model = None

    def __init__(self) -> None:
        self._embedding_cache = {}

    def _ensure_model_loaded(self):
        if EmbeddingFn._shared_model is None:
            # Suppress all warnings and logging
            warnings.filterwarnings("ignore")
            os.environ["PYTHONIOENCODING"] = "utf-8"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            for logger in [
                logging.getLogger(name)
                for name in ["transformers", "tqdm", "sentence_transformers"]
            ]:
                logger.setLevel(logging.ERROR)

            with open(os.devnull, "w", encoding="utf-8") as null_file:
                old_stdout = sys.stdout
                sys.stdout = null_file
                try:
                    # model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
                    # model_name = "BAAI/bge-large-en-v1.5"  # keep as option
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    EmbeddingFn._shared_model = SentenceTransformer(
                        model_name,
                        trust_remote_code=True,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )

                    # Performance optimizations
                    EmbeddingFn._shared_model.max_seq_length = 512
                    if torch.cuda.is_available():
                        EmbeddingFn._shared_model = EmbeddingFn._shared_model.half()
                        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

                finally:
                    sys.stdout = old_stdout

            self.model = EmbeddingFn._shared_model

    def create_embeddings(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        try:
            self._ensure_model_loaded()

            if isinstance(texts, str):
                texts = [texts]

            # Larger batch size for better throughput
            batch_size = 64  # Increased from 32

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

                if torch.is_tensor(embeddings):
                    embeddings = embeddings.cpu().numpy()

                return embeddings.tolist()

        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise
