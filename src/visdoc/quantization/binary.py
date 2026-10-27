from typing import Dict, Any, Tuple, Optional
import math
import torch
import torch.nn.functional as F
from visdoc.quantization.base import BaseQuantizer
from visdoc.scoring.bitwise import pack_bits_tensor, unpack_bits_tensor, binary_maxsim

class BinaryQuantizer(BaseQuantizer):
    """
    1-Bit Binary Quantizer with bit-packing.
    Maps 128-dimensional float embeddings to 128 bits (16 uint8 bytes).
    16x compression over FP16, 32x over FP32.
    """

    def __init__(self, threshold: float = 0.0, embedding_dim: int = 128):
        self.format_name = "binary"
        self.threshold = threshold
        self.embedding_dim = embedding_dim
        self.bytes_per_dim = 1.0 / 8.0 # 0.125 bytes per dim -> 16 bytes per 128-dim vector

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if tensor.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {tensor.shape[-1]}")

        # Ensure L2 normalization
        normed = F.normalize(tensor.float(), p=2, dim=-1)

        # Sign bit thresholding
        bits = normed > self.threshold

        # Pack bits into uint8
        packed = pack_bits_tensor(bits)

        meta = {
            "format": self.format_name,
            "original_shape": list(tensor.shape),
            "packed_shape": list(packed.shape),
            "embedding_dim": self.embedding_dim,
            "threshold": self.threshold,
            "bytes_per_vector": self.embedding_dim // 8,
        }
        return packed, meta

    def dequantize(self, quantized_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        unpacked_bits = unpack_bits_tensor(quantized_tensor)
        # Convert bool to +1.0 / -1.0 scaled to unit sphere: 1.0 / sqrt(dim)
        scale = 1.0 / math.sqrt(self.embedding_dim)
        dequantized = torch.where(unpacked_bits, scale, -scale).float()
        return dequantized

    def compute_score(
        self,
        query_embeddings: torch.Tensor,
        doc_quantized: torch.Tensor,
        metadata: Dict[str, Any],
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Binarize query with same packing
        q_bits = (query_embeddings > self.threshold)
        q_packed = pack_bits_tensor(q_bits).to(doc_quantized.device)

        return binary_maxsim(
            packed_query=q_packed,
            packed_doc=doc_quantized,
            embedding_dim=self.embedding_dim,
            query_mask=query_mask.to(doc_quantized.device) if query_mask is not None else None,
            doc_mask=doc_mask.to(doc_quantized.device) if doc_mask is not None else None,
        )
