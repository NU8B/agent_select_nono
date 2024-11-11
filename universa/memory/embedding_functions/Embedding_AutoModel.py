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
    _instance = None
    _model = None
    _tokenizer = None

    def __init__(self) -> None:
        self.model_name = "dunzhang/stella_en_1.5B_v5"
        self._ensure_model_loaded()

    @classmethod
    def get_instance(cls):
        """Singleton accessor"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_model_loaded(self):
        if self._model is None:
            # Configure environment
            os.environ["PYTHONIOENCODING"] = "utf-8"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("tqdm").setLevel(logging.ERROR)

            print(f"Loading model: {self.model_name}")
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Optimize model for inference
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.half().cuda()
                torch.backends.cudnn.benchmark = True

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings with optimized batch processing"""
        try:
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self._tokenizer(
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
                        outputs = self._model(**inputs)
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
