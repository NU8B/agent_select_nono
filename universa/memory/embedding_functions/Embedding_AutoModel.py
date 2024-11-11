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
    """A singleton class that provides text embedding functionality using the Stella model.
    
    This class loads and manages a pre-trained language model for creating text embeddings,
    optimized for GPU acceleration when available. It implements a singleton pattern to
    ensure efficient memory usage by maintaining only one instance of the model.
    
    Attributes:
        model_name (str): The identifier for the pre-trained model to be used
        _instance (cls): Singleton instance of the class
        _model (AutoModel): The loaded transformer model
        _tokenizer (AutoTokenizer): The tokenizer corresponding to the model
    """
    _instance = None
    _model = None
    _tokenizer = None

    def __init__(self) -> None:
        """Initialize the embedding function with the Stella model."""
        self.model_name = "dunzhang/stella_en_1.5B_v5"
        self._ensure_model_loaded()

    @classmethod
    def get_instance(cls):
        """Singleton pattern implementation to ensure only one model instance exists.
        
        Returns:
            EmbeddingFn: The single instance of the embedding function
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_model_loaded(self):
        """Loads the model and tokenizer if not already loaded.
        
        This method:
        1. Configures environment variables for proper encoding
        2. Suppresses unnecessary warning logs
        3. Loads the model and tokenizer
        4. Optimizes the model for inference:
           - Puts model in evaluation mode
           - Moves to GPU if available
           - Enables half-precision (FP16) for better performance
           - Enables CUDNN benchmarking for faster computation
        """
        if self._model is None:
            # Configure environment for proper UTF-8 handling and disable tokenizer parallelism
            os.environ["PYTHONIOENCODING"] = "utf-8"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Suppress unnecessary logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("tqdm").setLevel(logging.ERROR)

            print(f"Loading model: {self.model_name}")
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Optimize model for inference
            self._model.eval()  # Set to evaluation mode
            if torch.cuda.is_available():
                self._model = self._model.half().cuda()  # Use FP16 and move to GPU
                torch.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts with optimized batch processing.
        
        Args:
            texts (List[str]): List of input texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors, one for each input text
            
        Process:
        1. Process texts in batches of 32 for memory efficiency
        2. Tokenize with padding and truncation to 512 tokens
        3. Use GPU acceleration if available
        4. Apply inference optimizations:
           - No gradient computation
           - Inference mode
           - Automatic mixed precision (AMP)
        5. Generate embeddings by averaging token representations
        6. Normalize embeddings to unit length
        
        Raises:
            Exception: If embedding creation fails, with detailed error message
        """
        try:
            batch_size = 32  # Process in smaller batches to manage memory
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                # Tokenize with padding and length limiting
                inputs = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move inputs to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Generate embeddings with optimizations
                with torch.no_grad(), torch.inference_mode():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self._model(**inputs)
                        # Average token embeddings and normalize
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
        """Make the class callable, delegating to create_embeddings.
        
        Args:
            input (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.create_embeddings(input)
