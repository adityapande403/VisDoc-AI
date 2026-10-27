from typing import List, Dict, Set, Union
import math
import numpy as np
import torch
import torch.nn.functional as F

def compute_ndcg_at_k(retrieved_ids: List[str], positive_ids: Set[str], k: int = 5) -> float:
    """
    Computes NDCG@K for a single query.
    Supports binary and multiple positive labels.
    """
    if not positive_ids or k <= 0:
        return 0.0

    # Calculate Discounted Cumulative Gain (DCG@k)
    dcg = 0.0
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if pid in positive_ids:
            # Binary relevance = 1.0
            dcg += 1.0 / math.log2(rank + 1)

    # Calculate Ideal DCG (IDCG@k)
    ideal_hits = min(len(positive_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg

def compute_mrr_at_k(retrieved_ids: List[str], positive_ids: Set[str], k: int = 10) -> float:
    """Computes Mean Reciprocal Rank (MRR@K)."""
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if pid in positive_ids:
            return 1.0 / rank
    return 0.0

def compute_recall_at_k(retrieved_ids: List[str], positive_ids: Set[str], k: int = 5) -> float:
    """Computes Recall@K: fraction of true positive pages retrieved in top K."""
    if not positive_ids:
        return 0.0
    hits = sum(1 for pid in retrieved_ids[:k] if pid in positive_ids)
    return hits / float(len(positive_ids))

def evaluate_retrieval_run(
    results_per_query: Dict[str, List[str]], # query_id -> list of retrieved page_ids
    ground_truth: Dict[str, Set[str]],       # query_id -> set of positive page_ids
) -> Dict[str, float]:
    """
    Aggregates standard Information Retrieval metrics across all evaluation queries.
    """
    ndcg5_list = []
    mrr10_list = []
    r1_list = []
    r5_list = []
    r10_list = []
    r20_list = []

    for qid, pos_ids in ground_truth.items():
        retrieved = results_per_query.get(qid, [])
        ndcg5_list.append(compute_ndcg_at_k(retrieved, pos_ids, k=5))
        mrr10_list.append(compute_mrr_at_k(retrieved, pos_ids, k=10))
        r1_list.append(compute_recall_at_k(retrieved, pos_ids, k=1))
        r5_list.append(compute_recall_at_k(retrieved, pos_ids, k=5))
        r10_list.append(compute_recall_at_k(retrieved, pos_ids, k=10))
        r20_list.append(compute_recall_at_k(retrieved, pos_ids, k=20))

    return {
        "ndcg@5": float(np.mean(ndcg5_list)) if ndcg5_list else 0.0,
        "mrr@10": float(np.mean(mrr10_list)) if mrr10_list else 0.0,
        "recall@1": float(np.mean(r1_list)) if r1_list else 0.0,
        "recall@5": float(np.mean(r5_list)) if r5_list else 0.0,
        "recall@10": float(np.mean(r10_list)) if r10_list else 0.0,
        "recall@20": float(np.mean(r20_list)) if r20_list else 0.0,
        "evaluated_queries": len(ground_truth),
    }

def compute_quantization_error(
    original_tensor: torch.Tensor,
    reconstructed_tensor: torch.Tensor
) -> Dict[str, float]:
    """
    Computes numerical fidelity between original and reconstructed embeddings:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Cosine Similarity Correlation
    """
    orig = F.normalize(original_tensor.float(), p=2, dim=-1)
    recon = F.normalize(reconstructed_tensor.float(), p=2, dim=-1)

    mse = torch.mean((orig - recon) ** 2).item()
    mae = torch.mean(torch.abs(orig - recon)).item()
    cos_sim = torch.mean(torch.sum(orig * recon, dim=-1)).item()

    return {
        "mse": float(mse),
        "mae": float(mae),
        "mean_cosine_similarity": float(cos_sim),
    }
