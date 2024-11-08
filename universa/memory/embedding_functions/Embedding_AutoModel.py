from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from .base_embedder import BaseEmbeddingFunction
import logging
import os
import warnings
import numpy as np

# Silence the specific warning
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.cuda.amp.autocast"
)


class EmbeddingFn(BaseEmbeddingFunction):
    _shared_model = None
    _shared_tokenizer = None

    def __init__(self) -> None:
        self._embedding_cache = {}
        self.model_name = "dunzhang/stella_en_1.5B_v5"
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        if EmbeddingFn._shared_model is None:
            # Suppress warnings and logging
            os.environ["PYTHONIOENCODING"] = "utf-8"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("tqdm").setLevel(logging.ERROR)

            print(f"Loading model: {self.model_name}")
            EmbeddingFn._shared_model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            EmbeddingFn._shared_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            # Optimize model for inference
            EmbeddingFn._shared_model.eval()
            if torch.cuda.is_available():
                EmbeddingFn._shared_model = EmbeddingFn._shared_model.half().cuda()
                torch.backends.cudnn.benchmark = True

        self.model = EmbeddingFn._shared_model
        self.tokenizer = EmbeddingFn._shared_tokenizer

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings with optimized batch processing"""
        try:
            # Process in batches for better GPU utilization
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad(), torch.inference_mode():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )
                        all_embeddings.append(embeddings.cpu().numpy())

            return np.vstack(all_embeddings).tolist()

        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.create_embeddings(input)