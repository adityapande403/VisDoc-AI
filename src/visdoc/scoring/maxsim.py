from typing import List, Optional, Union
import torch
import torch.nn.functional as F

def exact_maxsim(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes reference late-interaction MaxSim score for a single (query, document) pair.

    Args:
        query_embeddings: Tensor of shape (N_q, D)
        doc_embeddings: Tensor of shape (N_d, D)
        query_mask: Optional boolean or 0/1 tensor of shape (N_q,)
        doc_mask: Optional boolean or 0/1 tensor of shape (N_d,)

    Returns:
        Scalar tensor containing the MaxSim score.
    """
    if query_embeddings.dim() != 2 or doc_embeddings.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got Q: {query_embeddings.shape}, D: {doc_embeddings.shape}")

    # Ensure L2 normalization
    q_norm = F.normalize(query_embeddings.float(), p=2, dim=-1)
    d_norm = F.normalize(doc_embeddings.float(), p=2, dim=-1)

    # Similarity matrix: (N_q, N_d)
    sim_matrix = torch.matmul(q_norm, d_norm.transpose(0, 1))

    if doc_mask is not None:
        doc_mask_bool = doc_mask.bool()
        # Set invalid document token similarities to large negative value
        sim_matrix = sim_matrix.masked_fill(~doc_mask_bool.unsqueeze(0), -1e4)

    # Max over document tokens: (N_q,)
    max_sims, _ = torch.max(sim_matrix, dim=-1)

    if query_mask is not None:
        query_mask_bool = query_mask.bool()
        max_sims = max_sims.masked_fill(~query_mask_bool, 0.0)
        score = torch.sum(max_sims)
    else:
        score = torch.sum(max_sims)

    return score

def compute_maxsim_scores(
    query_embeddings: torch.Tensor,
    doc_embeddings_list: List[torch.Tensor],
    query_mask: Optional[torch.Tensor] = None,
    doc_masks_list: Optional[List[torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Batched, memory-safe late-interaction MaxSim scoring over a list of document page embeddings.

    Args:
        query_embeddings: Tensor of shape (N_q, D) or (1, N_q, D)
        doc_embeddings_list: List of document page tensors, each of shape (N_d, D)
        query_mask: Optional mask of shape (N_q,)
        doc_masks_list: Optional list of document masks
        device: Target computation device
        batch_size: Document chunking size to avoid OOM

    Returns:
        1D Tensor of shape (len(doc_embeddings_list),) containing MaxSim scores.
    """
    if len(doc_embeddings_list) == 0:
        return torch.empty(0, dtype=torch.float32)

    if query_embeddings.dim() == 3:
        query_embeddings = query_embeddings.squeeze(0)

    target_dev = device or query_embeddings.device
    q = F.normalize(query_embeddings.to(target_dev).float(), p=2, dim=-1) # (N_q, D)

    if query_mask is not None:
        q_mask = query_mask.to(target_dev).bool() # (N_q,)
    else:
        q_mask = torch.ones(q.size(0), dtype=torch.bool, device=target_dev)

    scores = []
    num_docs = len(doc_embeddings_list)

    for i in range(0, num_docs, batch_size):
        chunk_docs = doc_embeddings_list[i : i + batch_size]
        chunk_masks = doc_masks_list[i : i + batch_size] if doc_masks_list else [None] * len(chunk_docs)

        for d_emb, d_mask in zip(chunk_docs, chunk_masks):
            d = F.normalize(d_emb.to(target_dev).float(), p=2, dim=-1) # (N_d, D)
            sim = torch.matmul(q, d.transpose(0, 1)) # (N_q, N_d)

            if d_mask is not None:
                d_m = d_mask.to(target_dev).bool()
                sim = sim.masked_fill(~d_m.unsqueeze(0), -1e4)

            max_sim, _ = torch.max(sim, dim=-1) # (N_q,)
            score = torch.sum(max_sim.masked_fill(~q_mask, 0.0))
            scores.append(score.item())

    return torch.tensor(scores, dtype=torch.float32)

def batched_maxsim_matrix(
    queries: torch.Tensor,
    docs: torch.Tensor,
    query_masks: Optional[torch.Tensor] = None,
    doc_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fully vectorized MaxSim for uniform batches:
    queries: (B_q, N_q, D)
    docs: (B_d, N_d, D)
    Returns: (B_q, B_d) score matrix.
    """
    # Normalize
    q = F.normalize(queries.float(), p=2, dim=-1) # (B_q, N_q, D)
    d = F.normalize(docs.float(), p=2, dim=-1)    # (B_d, N_d, D)

    # Compute all pairwise dot products: (B_q, B_d, N_q, N_d)
    sims = torch.einsum("qik, bjk -> qbij", q, d)

    if doc_masks is not None:
        # doc_masks: (B_d, N_d) -> (1, B_d, 1, N_d)
        d_mask = doc_masks.bool().unsqueeze(0).unsqueeze(2)
        sims = sims.masked_fill(~d_mask, -1e4)

    # Max over document tokens: (B_q, B_d, N_q)
    max_sims, _ = torch.max(sims, dim=-1)

    if query_masks is not None:
        # query_masks: (B_q, N_q) -> (B_q, 1, N_q)
        q_mask = query_masks.bool().unsqueeze(1)
        max_sims = max_sims.masked_fill(~q_mask, 0.0)

    # Sum over query tokens: (B_q, B_d)
    scores = torch.sum(max_sims, dim=-1)
    return scores
