"""
Unified embedder interface для всех sentence-transformer моделей.
"""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from tqdm import tqdm


class UnifiedEmbedder:
    """Unified interface для всех эмбеддеров."""

    def __init__(
        self,
        model_name: str,
        prefix_passage: str = "",
        prefix_query: str = "",
        max_seq_length: int = 512,
        device: str = "cuda",
        batch_size: int = 64,
    ):
        """
        Args:
            model_name: HuggingFace model name
            prefix_passage: Префикс для документов (например, "passage: " для E5)
            prefix_query: Префикс для запросов (например, "query: " для E5)
            max_seq_length: Максимальная длина последовательности
            device: cuda или cpu
            batch_size: Batch size для encode
        """
        self.model_name = model_name
        self.prefix_passage = prefix_passage
        self.prefix_query = prefix_query
        self.max_seq_length = max_seq_length
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        print(f"Loading model: {model_name}")
        print(f"  Device: {self.device}")

        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = max_seq_length
        except Exception as e:
            print(f"  WARNING: Failed to load {model_name}: {e}")
            print(f"  Trying fallback to paraphrase-multilingual-mpnet-base-v2...")
            self.model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                device=self.device
            )
            self.model.max_seq_length = max_seq_length

    def encode_passages(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode passages/documents.

        Args:
            texts: List of texts
            show_progress: Show progress bar
            normalize: Normalize embeddings (L2 norm)

        Returns:
            Embeddings [n_texts, dim]
        """
        # Add prefix if needed
        if self.prefix_passage:
            texts = [self.prefix_passage + text for text in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings.astype('float32')

    def encode_queries(
        self,
        queries: List[str],
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode queries.

        Args:
            queries: List of queries
            show_progress: Show progress bar
            normalize: Normalize embeddings

        Returns:
            Embeddings [n_queries, dim]
        """
        # Add prefix if needed
        if self.prefix_query:
            queries = [self.prefix_query + q for q in queries]

        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings.astype('float32')

    def get_embedding_dim(self) -> int:
        """Returns embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
