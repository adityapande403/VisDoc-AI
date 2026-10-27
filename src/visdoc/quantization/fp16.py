from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from visdoc.quantization.base import BaseQuantizer
from visdoc.scoring.maxsim import exact_maxsim

class FP16Quantizer(BaseQuantizer):
    """
    FP16 Half-Precision baseline quantizer.
    Stores embeddings in standard IEEE 754 float16 (2 bytes per dimension).
    """

    def __init__(self):
        self.format_name = "fp16"
        self.bytes_per_dim = 2

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Ensure L2 normalization before storing in half precision
        normed = F.normalize(tensor.float(), p=2, dim=-1)
        quantized = normed.to(torch.float16)
        meta = {
            "format": self.format_name,
            "shape": list(quantized.shape),
            "dtype": "float16",
        }
        return quantized, meta

    def dequantize(self, quantized_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        return quantized_tensor.to(torch.float32)

    def compute_score(
        self,
        query_embeddings: torch.Tensor,
        doc_quantized: torch.Tensor,
        metadata: Dict[str, Any],
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Exact reference MaxSim on float32/float16
        return exact_maxsim(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_quantized.to(query_embeddings.device),
            query_mask=query_mask,
            doc_mask=doc_mask,
        )
