from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Generator
import torch
from visdoc.ingestion.renderer import RenderedPage
from visdoc.models.base import BaseVisualRetriever
from visdoc.quantization.base import BaseQuantizer
from visdoc.quantization.fp16 import FP16Quantizer
from visdoc.utils.logging import logger

@dataclass
class PageEmbedding:
    """Multi-vector embedding representation of a single document page."""
    page_id: str
    document_id: str
    page_number: int
    num_vectors: int
    embedding_dim: int
    embeddings: torch.Tensor # (N_d, D) unquantized float tensor
    mask: Optional[torch.Tensor] = None # (N_d,) boolean valid mask
    quantized_tensor: Optional[torch.Tensor] = None
    quantization_meta: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validates numerical integrity of page embeddings."""
        if self.embeddings.numel() == 0:
            raise ValueError(f"Page {self.page_id} has empty embedding tensor.")
        if not torch.all(torch.isfinite(self.embeddings)):
            raise ValueError(f"Page {self.page_id} contains NaN or Inf values in embeddings.")
        if self.embeddings.dim() != 2:
            raise ValueError(f"Expected 2D embedding tensor (N_d, D), got shape {self.embeddings.shape}")
        if self.embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {self.embeddings.shape[-1]}")
        return True

class EmbeddingGenerator:
    """Generates multi-vector embeddings for rendered pages."""

    def __init__(
        self,
        retriever: BaseVisualRetriever,
        quantizer: Optional[BaseQuantizer] = None,
        batch_size: int = 4
    ):
        self.retriever = retriever
        self.quantizer = quantizer or FP16Quantizer()
        self.batch_size = batch_size

    def generate_embeddings(
        self,
        pages: List[RenderedPage]
    ) -> List[PageEmbedding]:
        """Encodes a list of rendered pages into PageEmbedding objects."""
        results: List[PageEmbedding] = []
        total = len(pages)

        for i in range(0, total, self.batch_size):
            batch_pages = pages[i : i + self.batch_size]
            images = [p.image for p in batch_pages]

            embs_list = self.retriever.encode_pages(images)

            for page, emb in zip(batch_pages, embs_list):
                # Quantize embedding using active quantizer
                q_tensor, q_meta = self.quantizer.quantize(emb)

                page_emb = PageEmbedding(
                    page_id=page.page_id,
                    document_id=page.document_id,
                    page_number=page.page_number,
                    num_vectors=emb.shape[0],
                    embedding_dim=emb.shape[1],
                    embeddings=emb,
                    quantized_tensor=q_tensor,
                    quantization_meta=q_meta,
                    metadata=page.to_dict(),
                )
                page_emb.validate()
                results.append(page_emb)

        logger.info(f"Generated embeddings for {len(results)} pages (Dim: {self.retriever.embedding_dim})")
        return results
