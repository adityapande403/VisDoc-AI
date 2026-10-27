from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from PIL import Image
import torch

class BaseVisualRetriever(ABC):
    """Abstract interface for multi-vector vision-language retrieval models."""

    @abstractmethod
    def encode_pages(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Encodes a list of document page images into multi-vector embeddings.
        Returns:
            List of 2D tensors, each of shape (N_d, D) where N_d is the number of patch/token vectors.
        """
        pass

    @abstractmethod
    def encode_queries(self, queries: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a list of textual queries into multi-vector representations.
        Returns:
            Tuple of:
            - query_embeddings: Tensor of shape (B_q, N_q, D)
            - query_masks: Tensor of shape (B_q, N_q) indicating valid non-padding tokens.
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass
