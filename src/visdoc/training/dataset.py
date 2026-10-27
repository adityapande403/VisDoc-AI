from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from visdoc.dataset.schema import RetrievalExample
from visdoc.ingestion.renderer import RenderedPage
from visdoc.utils.logging import logger

class VisualRetrievalDataset(Dataset):
    """
    PyTorch Dataset for visual late-interaction contrastive fine-tuning.
    Yields (query_text, positive_image, negative_image, metadata) triplets.
    """

    def __init__(
        self,
        examples: List[RetrievalExample],
        pages_dict: Dict[str, RenderedPage],
    ):
        self.examples = examples
        self.pages_dict = pages_dict

        # Filter out examples whose positive page is missing
        self.valid_examples = [
            ex for ex in self.examples
            if any(pid in self.pages_dict for pid in ex.positive_page_ids)
        ]
        logger.info(f"Initialized VisualRetrievalDataset with {len(self.valid_examples)} valid examples.")

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.valid_examples[idx]

        # Select first available positive page
        pos_id = next(pid for pid in ex.positive_page_ids if pid in self.pages_dict)
        pos_page = self.pages_dict[pos_id]

        # Select hard negative page if available, else pick another page
        neg_page = None
        if ex.negative_page_ids:
            for nid in ex.negative_page_ids:
                if nid in self.pages_dict and nid != pos_id:
                    neg_page = self.pages_dict[nid]
                    break

        if neg_page is None:
            # Pick a fallback negative from other pages
            for pid, page in self.pages_dict.items():
                if pid not in ex.positive_page_ids:
                    neg_page = page
                    break

        return {
            "query": ex.query,
            "pos_image": pos_page.image,
            "neg_image": neg_page.image if neg_page else pos_page.image,
            "query_id": ex.query_id,
            "pos_page_id": pos_id,
            "neg_page_id": neg_page.page_id if neg_page else "",
        }

    @classmethod
    def from_jsonl(cls, jsonl_path: str | Path, pages: List[RenderedPage]) -> "VisualRetrievalDataset":
        """Loads dataset from JSONL annotations file."""
        pages_dict = {p.page_id: p for p in pages}
        examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(RetrievalExample(**data))
        return cls(examples=examples, pages_dict=pages_dict)
