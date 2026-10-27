from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch

class BaseQuantizer(ABC):
    """Abstract base class for late-interaction multi-vector embedding quantizers."""

    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantizes an uncompressed floating point embedding tensor of shape (N, D).
        Returns:
            quantized_tensor: Compressed tensor
            metadata: Auxiliary metadata needed for reconstruction or scoring (e.g. scales, zero-points)
        """
        pass

    @abstractmethod
    def dequantize(self, quantized_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Reconstructs the float tensor representation.
        """
        pass

    @abstractmethod
    def compute_score(
        self,
        query_embeddings: torch.Tensor,
        doc_quantized: torch.Tensor,
        metadata: Dict[str, Any],
        query_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes late-interaction MaxSim score using the quantized representation.
        """
        pass
