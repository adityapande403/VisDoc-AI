import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from visdoc.scoring.maxsim import exact_maxsim

class ColBERTPairwiseNegativeLoss(nn.Module):
    """
    ColBERT / ColQwen Late-Interaction Contrastive Loss.
    Combines pairwise margin/cross-entropy with hard negatives and in-batch cross-entropy.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        in_batch_term_weight: float = 0.5,
        normalize_scores: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.in_batch_term_weight = in_batch_term_weight
        self.normalize_scores = normalize_scores
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        query_embs: torch.Tensor,       # (B, N_q, D)
        pos_doc_embs: torch.Tensor,     # (B, N_dp, D)
        neg_doc_embs: Optional[torch.Tensor] = None, # (B, N_dn, D)
        query_masks: Optional[torch.Tensor] = None,   # (B, N_q)
        pos_masks: Optional[torch.Tensor] = None,     # (B, N_dp)
        neg_masks: Optional[torch.Tensor] = None,     # (B, N_dn)
    ) -> torch.Tensor:
        """
        Computes the late-interaction contrastive loss.
        """
        batch_size = query_embs.size(0)

        # 1. Compute positive scores: S_pos[i] = MaxSim(Q[i], D_pos[i])
        pos_scores = []
        for i in range(batch_size):
            q_m = query_masks[i] if query_masks is not None else None
            p_m = pos_masks[i] if pos_masks is not None else None
            s_pos = exact_maxsim(query_embs[i], pos_doc_embs[i], q_m, p_m)
            pos_scores.append(s_pos)
        pos_scores_tensor = torch.stack(pos_scores) # (B,)

        total_loss = torch.tensor(0.0, device=query_embs.device, requires_grad=True)

        # 2. Pairwise hard negative loss if hard negatives are provided
        if neg_doc_embs is not None:
            neg_scores = []
            for i in range(batch_size):
                q_m = query_masks[i] if query_masks is not None else None
                n_m = neg_masks[i] if neg_masks is not None else None
                s_neg = exact_maxsim(query_embs[i], neg_doc_embs[i], q_m, n_m)
                neg_scores.append(s_neg)
            neg_scores_tensor = torch.stack(neg_scores) # (B,)

            # Binary cross entropy with logits on (pos - neg) / tau
            diff = (pos_scores_tensor - neg_scores_tensor) / self.temperature
            # We want pos > neg, so target is 1.0: loss = log(1 + exp(-diff)) = softplus(-diff)
            pairwise_loss = F.softplus(-diff).mean()
        else:
            pairwise_loss = torch.tensor(0.0, device=query_embs.device)

        # 3. In-batch negative cross-entropy loss if batch_size > 1
        if batch_size > 1 and self.in_batch_term_weight > 0.0:
            # Full (B, B) score matrix
            in_batch_matrix = torch.zeros((batch_size, batch_size), device=query_embs.device)
            for i in range(batch_size):
                q_m = query_masks[i] if query_masks is not None else None
                for j in range(batch_size):
                    p_m = pos_masks[j] if pos_masks is not None else None
                    in_batch_matrix[i, j] = exact_maxsim(query_embs[i], pos_doc_embs[j], q_m, p_m)

            scaled_matrix = in_batch_matrix / self.temperature
            labels = torch.arange(batch_size, device=query_embs.device, dtype=torch.long)
            in_batch_loss = self.ce_loss(scaled_matrix, labels)
        else:
            in_batch_loss = torch.tensor(0.0, device=query_embs.device)

        # Combine
        if neg_doc_embs is not None and batch_size > 1 and self.in_batch_term_weight > 0.0:
            loss = (1.0 - self.in_batch_term_weight) * pairwise_loss + self.in_batch_term_weight * in_batch_loss
        elif neg_doc_embs is not None:
            loss = pairwise_loss
        else:
            loss = in_batch_loss

        return loss
