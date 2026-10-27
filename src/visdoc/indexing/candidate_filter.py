from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

class CandidateFilter:
    """
    Two-stage candidate generation filter.
    Uses mean-pooled document vectors to rapidly prune large indexes
    before computing exact or quantized multi-vector MaxSim.
    """

    def __init__(self, top_candidates: int = 100, ratio: float = 0.2):
        self.top_candidates = top_candidates
        self.ratio = ratio

    @staticmethod
    def compute_page_summary(page_embeddings: torch.Tensor) -> torch.Tensor:
        """Computes L2-normalized mean-pooled summary vector of a page: shape (D,)."""
        mean_vec = torch.mean(page_embeddings.float(), dim=0)
        return F.normalize(mean_vec, p=2, dim=-1)

    def filter_candidates(
        self,
        query_embeddings: torch.Tensor,
        page_summaries: torch.Tensor, # (N_pages, D)
        query_mask: Optional[torch.Tensor] = None,
        k: Optional[int] = None
    ) -> List[int]:
        """
        Ranks page summaries against query summary and returns top candidate indices.
        """
        if page_summaries.size(0) == 0:
            return []

        # Mean query vector
        if query_mask is not None:
            valid_q = query_embeddings[query_mask.bool()]
        else:
            valid_q = query_embeddings

        if valid_q.size(0) == 0:
            valid_q = query_embeddings

        q_summary = F.normalize(torch.mean(valid_q.float(), dim=0), p=2, dim=-1) # (D,)

        # Dot product with all page summaries: (N_pages,)
        dots = torch.matmul(page_summaries.to(q_summary.device), q_summary)

        total_pages = page_summaries.size(0)
        num_to_keep = k or max(self.top_candidates, int(total_pages * self.ratio))
        num_to_keep = min(total_pages, max(1, num_to_keep))

        top_indices = torch.topk(dots, k=num_to_keep).indices.cpu().tolist()
        return top_indices
