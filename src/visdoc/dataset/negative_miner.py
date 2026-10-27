from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
import torch
from visdoc.dataset.schema import RetrievalExample
from visdoc.indexing.page_index import PageIndex
from visdoc.models.base import BaseVisualRetriever
from visdoc.utils.logging import logger

@dataclass
class MinedNegative:
    page_id: str
    document_id: str
    similarity_score: float
    difficulty_level: str # "hard", "medium", "adjacent_page"
    provenance: Dict[str, Any]

class HardNegativeMiner:
    """
    Mines difficult negative page candidates using retrieval near-neighbours
    while preventing false negative contamination and label leakage.
    """

    def __init__(
        self,
        retriever_model: BaseVisualRetriever,
        page_index: PageIndex,
        max_negatives_per_query: int = 5,
        min_score_threshold: float = 0.3,
    ):
        self.model = retriever_model
        self.index = page_index
        self.max_negatives = max_negatives_per_query
        self.min_score = min_score_threshold

    def mine_negatives_for_example(
        self,
        example: RetrievalExample,
        allow_same_document: bool = True
    ) -> List[MinedNegative]:
        """
        Mines hard negative pages for a single query example.
        """
        # Encode query
        query_embs, query_masks = self.model.encode_queries([example.query])
        q_emb = query_embs[0]
        q_mask = query_masks[0]

        # Retrieve top candidates (fetching extra to filter positives)
        top_k = min(len(self.index), self.max_negatives * 4 + 10)
        raw_results = self.index.search(
            query_embeddings=q_emb,
            query_mask=q_mask,
            top_k=top_k,
        )

        pos_set: Set[str] = set(example.positive_page_ids)
        mined: List[MinedNegative] = []

        for res in raw_results:
            page_id = res["page_id"]
            doc_id = res["document_id"]
            score = res["score"]

            # Exclude true positive pages
            if page_id in pos_set:
                continue

            # Check document leakage constraint
            if not allow_same_document and doc_id == example.source_document_id:
                continue

            # Determine negative difficulty
            is_same_doc = (doc_id == example.source_document_id)
            if is_same_doc:
                difficulty = "adjacent_page"
            elif score > self.min_score:
                difficulty = "hard"
            else:
                difficulty = "medium"

            mined.append(
                MinedNegative(
                    page_id=page_id,
                    document_id=doc_id,
                    similarity_score=score,
                    difficulty_level=difficulty,
                    provenance={
                        "retrieved_rank": len(mined) + 1,
                        "query_id": example.query_id,
                        "mining_model": str(type(self.model).__name__),
                    }
                )
            )

            if len(mined) >= self.max_negatives:
                break

        return mined

    def enrich_dataset_with_hard_negatives(
        self,
        examples: List[RetrievalExample],
        allow_same_document: bool = True
    ) -> List[RetrievalExample]:
        """
        Mines hard negatives for an entire dataset split and updates negative_page_ids.
        """
        updated_examples: List[RetrievalExample] = []
        total_mined = 0

        for ex in examples:
            mined = self.mine_negatives_for_example(ex, allow_same_document=allow_same_document)
            mined_ids = [m.page_id for m in mined]

            # Merge with existing negatives without duplicates
            combined_negatives = list(dict.fromkeys(ex.negative_page_ids + mined_ids))

            new_ex = ex.model_copy(update={"negative_page_ids": combined_negatives})
            updated_examples.append(new_ex)
            total_mined += len(mined)

        logger.info(f"Mined {total_mined} total hard negatives across {len(examples)} queries.")
        return updated_examples
