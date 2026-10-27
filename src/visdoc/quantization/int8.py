from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from visdoc.quantization.base import BaseQuantizer
from visdoc.scoring.maxsim import exact_maxsim

class INT8Quantizer(BaseQuantizer):
    """
    Uniform symmetric INT8 scalar quantization for multi-vector embeddings.
    Maps [-max_val, max_val] into [-127, 127] with scale = max_val / 127.0.
    1 byte per dimension (2x compression over FP16).
    """

    def __init__(self, mode: str = "dequant", clip_percentile: float = 99.9, per_vector: bool = True):
        self.format_name = "int8"
        self.mode = mode # "dequant" or "int_dot"
        self.clip_percentile = clip_percentile
        self.per_vector = per_vector
        self.bytes_per_dim = 1

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Ensure L2 normalization
        normed = F.normalize(tensor.float(), p=2, dim=-1)

        if self.per_vector:
            # Per-vector scale: shape (N, 1)
            max_vals = torch.max(torch.abs(normed), dim=-1, keepdim=True)[0]
            # Avoid division by zero
            scales = torch.clamp(max_vals, min=1e-8) / 127.0
        else:
            # Per-tensor scale: scalar
            max_val = torch.max(torch.abs(normed)).item()
            scales = torch.tensor([max(max_val, 1e-8) / 127.0], device=tensor.device)

        # Quantize to int8: clamp to [-127, 127]
        quantized = torch.clamp(torch.round(normed / scales), -127, 127).to(torch.int8)

        meta = {
            "format": self.format_name,
            "shape": list(quantized.shape),
            "scales": scales.cpu().tolist(),
            "per_vector": self.per_vector,
            "mode": self.mode,
        }
        return quantized, meta

    def dequantize(self, quantized_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        scales = torch.tensor(metadata["scales"], device=quantized_tensor.device, dtype=torch.float32)
        dequantized = quantized_tensor.float() * scales
        # Re-normalize reconstructed vectors to unit sphere
        return F.normalize(dequantized, p=2, dim=-1)

    def compute_score(
        self,
        query_embeddings: torch.Tensor,
        doc_quantized: torch.Tensor,
        metadata: Dict[str, Any],
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "dequant":
            doc_dequant = self.dequantize(doc_quantized.to(query_embeddings.device), metadata)
            return exact_maxsim(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_dequant,
                query_mask=query_mask,
                doc_mask=doc_mask,
            )
        else:
            # Fast direct scaled dot product
            # query in float32, doc in int8:
            # sim = (q @ d_int8.T) * scale
            q = F.normalize(query_embeddings.to(doc_quantized.device).float(), p=2, dim=-1)
            scales = torch.tensor(metadata["scales"], device=doc_quantized.device, dtype=torch.float32)

            # Matmul: (N_q, D) @ (D, N_d) -> (N_q, N_d)
            raw_dot = torch.matmul(q, doc_quantized.float().transpose(0, 1))
            sim_matrix = raw_dot * scales.squeeze(-1).unsqueeze(0)

            if doc_mask is not None:
                sim_matrix = sim_matrix.masked_fill(~doc_mask.bool().unsqueeze(0), -1e4)

            max_sims, _ = torch.max(sim_matrix, dim=-1)

            if query_mask is not None:
                max_sims = max_sims.masked_fill(~query_mask.bool(), 0.0)

            return torch.sum(max_sims)
