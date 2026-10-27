from typing import List, Optional
import torch
import numpy as np

# Precomputed 256-element POPCOUNT lookup table for uint8
POPCOUNT_TABLE = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.uint8)

def pack_bits_tensor(bool_tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs a boolean tensor of shape (..., D) where D is divisible by 8 into uint8 tensor of shape (..., D // 8).
    """
    shape = bool_tensor.shape
    dim = shape[-1]
    if dim % 8 != 0:
        raise ValueError(f"Last dimension ({dim}) must be a multiple of 8")

    # Reshape to (..., D // 8, 8)
    reshaped = bool_tensor.reshape(*shape[:-1], dim // 8, 8).to(torch.uint8)

    # Shift factors: [128, 64, 32, 16, 8, 4, 2, 1]
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=bool_tensor.device)
    packed = (reshaped * powers).sum(dim=-1, dtype=torch.uint8)
    return packed

def unpack_bits_tensor(packed_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unpacks a uint8 tensor of shape (..., D // 8) into a boolean tensor of shape (..., D).
    """
    shape = packed_tensor.shape
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=packed_tensor.device)
    unpacked = (packed_tensor.unsqueeze(-1) & powers) > 0
    return unpacked.reshape(*shape[:-1], shape[-1] * 8)

def binary_maxsim(
    packed_query: torch.Tensor,
    packed_doc: torch.Tensor,
    embedding_dim: int = 128,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes late-interaction MaxSim directly on 1-bit binary packed representations.

    Args:
        packed_query: (N_q, D // 8) in torch.uint8
        packed_doc: (N_d, D // 8) in torch.uint8
        embedding_dim: Original vector dimension (e.g. 128)
        query_mask: (N_q,) boolean mask
        doc_mask: (N_d,) boolean mask

    Returns:
        Scalar tensor containing the binary MaxSim score.
    """
    # Bitwise XOR: (N_q, N_d, D // 8)
    xor_result = packed_query.unsqueeze(1) ^ packed_doc.unsqueeze(0)

    # Fast Popcount via table lookup
    table = POPCOUNT_TABLE.to(packed_query.device)
    hamming_matrix = table[xor_result.long()].sum(dim=-1).float() # (N_q, N_d)

    # Convert Hamming distance to equivalent Cosine similarity: (dim - 2 * H) / dim
    sim_matrix = (embedding_dim - 2.0 * hamming_matrix) / float(embedding_dim)

    if doc_mask is not None:
        sim_matrix = sim_matrix.masked_fill(~doc_mask.bool().unsqueeze(0), -1e4)

    max_sims, _ = torch.max(sim_matrix, dim=-1) # (N_q,)

    if query_mask is not None:
        max_sims = max_sims.masked_fill(~query_mask.bool(), 0.0)

    return torch.sum(max_sims)
