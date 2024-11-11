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
<<<<<<< Updated upstream
        self._embedding_cache = {}
        self.model_name = "dunzhang/stella_en_1.5B_v5"
        self._ensure_model_loaded()

=======
        self.model_name = "dunzhang/stella_en_1.5B_v5" # Set the model with any model from Hugging Face
        self._ensure_model_loaded() # Load the model and tokenizer

    @classmethod
    def get_instance(cls):
        """Singleton accessor"""
        # Ensure only one instance of the class is created
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

>>>>>>> Stashed changes
    def _ensure_model_loaded(self):
        if EmbeddingFn._shared_model is None:
            # Suppress warnings and logging
            os.environ["PYTHONIOENCODING"] = "utf-8"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("tqdm").setLevel(logging.ERROR)

            print(f"Loading model: {self.model_name}")
<<<<<<< Updated upstream
            EmbeddingFn._shared_model = AutoModel.from_pretrained(
=======
            # Load the pre-trained model and tokenizer
            self._model = AutoModel.from_pretrained(
>>>>>>> Stashed changes
                self.model_name, trust_remote_code=True
            )
            EmbeddingFn._shared_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            # Optimize model for inference
<<<<<<< Updated upstream
            EmbeddingFn._shared_model.eval()
            if torch.cuda.is_available():
                EmbeddingFn._shared_model = EmbeddingFn._shared_model.half().cuda()
                torch.backends.cudnn.benchmark = True
=======
            self._model.eval() # Set the model to evaluation mode
            if torch.cuda.is_available():
                self._model = self._model.half().cuda() # Use half precision on GPU
                torch.backends.cudnn.benchmark = True # Enable cuDNN auto-tuning
>>>>>>> Stashed changes

        self.model = EmbeddingFn._shared_model
        self.tokenizer = EmbeddingFn._shared_tokenizer

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings with optimized batch processing"""
        try:
            # Process in batches for better GPU utilization
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
<<<<<<< Updated upstream
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
=======
                batch_texts = texts[i : i + batch_size] 
                inputs = self._tokenizer(
>>>>>>> Stashed changes
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Use GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Disable gradient calculation and inference mode
                with torch.no_grad(), torch.inference_mode():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
<<<<<<< Updated upstream
                        outputs = self.model(**inputs)
=======
                        outputs = self._model(**inputs) # Get model outputs
                        # Calculate mean of the last hidden state to get embeddings
>>>>>>> Stashed changes
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                        # Normalize the embeddings
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )
                        # Append the embeddings to the list
                        all_embeddings.append(embeddings.cpu().numpy())

            return np.vstack(all_embeddings).tolist() 

        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Allow the class instance to be called like a function
        return self.create_embeddings(input)
