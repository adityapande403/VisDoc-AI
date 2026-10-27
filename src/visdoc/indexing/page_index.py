from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import time
import torch
import torch.nn.functional as F

from visdoc.embeddings.generator import PageEmbedding
from visdoc.quantization.base import BaseQuantizer
from visdoc.quantization.fp16 import FP16Quantizer
from visdoc.quantization.int8 import INT8Quantizer
from visdoc.quantization.binary import BinaryQuantizer
from visdoc.quantization import get_quantizer
from visdoc.indexing.candidate_filter import CandidateFilter
from visdoc.utils.logging import logger

class PageIndex:
    """
    Production Multi-Vector Persistent Page Index supporting late interaction,
    FP16/INT8/Binary quantization, metadata filtering, and incremental updates.
    """

    def __init__(
        self,
        format_name: str = "fp16",
        index_dir: Optional[str | Path] = None,
        embedding_dim: int = 128,
        use_gpu: bool = True,
        quantizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.format_name = format_name.lower()
        self.index_dir = Path(index_dir) if index_dir else None
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Quantizer instance
        self.quantizer: BaseQuantizer = get_quantizer(self.format_name, **(quantizer_kwargs or {}))

        # In-memory index structures
        self.page_ids: List[str] = []
        self.document_ids: List[str] = []
        self.page_numbers: List[int] = []
        self.quantized_tensors: List[torch.Tensor] = []
        self.quantization_metas: List[Dict[str, Any]] = []
        self.page_metadatas: List[Dict[str, Any]] = []
        self.page_summaries: Optional[torch.Tensor] = None # (N_pages, D) for candidate filtering

        # Checksum registry for duplicate detection: sha256 -> page_id
        self.checksum_map: Dict[str, str] = {}
        # Page ID to index lookup
        self.id_to_idx: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.page_ids)

    def add_pages(
        self,
        page_embeddings: List[PageEmbedding],
        allow_duplicates: bool = False
    ) -> int:
        """
        Adds page embeddings to the index.
        Returns the number of newly added pages.
        """
        added_count = 0
        new_summaries = []

        for pe in page_embeddings:
            # Check for duplicate page by image sha256 if available
            sha256 = pe.metadata.get("image_sha256", pe.page_id)
            if not allow_duplicates and sha256 in self.checksum_map:
                logger.warning(f"Skipping duplicate page {pe.page_id} (checksum {sha256[:8]} already exists)")
                continue

            # Quantize embedding using active quantizer
            q_tensor, q_meta = self.quantizer.quantize(pe.embeddings)

            # Move quantized tensor to target device (e.g. GPU VRAM)
            q_tensor = q_tensor.to(self.device)

            idx = len(self.page_ids)
            self.page_ids.append(pe.page_id)
            self.document_ids.append(pe.document_id)
            self.page_numbers.append(pe.page_number)
            self.quantized_tensors.append(q_tensor)
            self.quantization_metas.append(q_meta)
            self.page_metadatas.append(pe.metadata)

            self.checksum_map[sha256] = pe.page_id
            self.id_to_idx[pe.page_id] = idx

            # Compute summary vector for candidate generation
            summary = CandidateFilter.compute_page_summary(pe.embeddings)
            new_summaries.append(summary)
            added_count += 1

        if new_summaries:
            new_sum_tensor = torch.stack(new_summaries).to(self.device)
            if self.page_summaries is None:
                self.page_summaries = new_sum_tensor
            else:
                self.page_summaries = torch.cat([self.page_summaries, new_sum_tensor], dim=0)

        logger.info(f"Added {added_count} pages to index '{self.format_name}' (Total pages: {len(self.page_ids)})")
        return added_count

    def search(
        self,
        query_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_candidate_filter: bool = False,
        candidate_ratio: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Executes late-interaction MaxSim search over the indexed pages.

        Args:
            query_embeddings: (N_q, D) query vector tensor
            query_mask: (N_q,) boolean mask
            top_k: Number of ranked results to return
            metadata_filter: Key-value metadata criteria (e.g. {"document_id": "doc1"})
            use_candidate_filter: Whether to run fast candidate pruning

        Returns:
            Ranked list of result dicts: page_id, document_id, page_number, score, metadata.
        """
        total_pages = len(self.page_ids)
        if total_pages == 0:
            return []

        # Step 1: Candidate indices selection
        if metadata_filter:
            candidate_indices = [
                i for i in range(total_pages)
                if all(self.page_metadatas[i].get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            candidate_indices = list(range(total_pages))

        if not candidate_indices:
            return []

        # Optional Two-stage candidate filtering
        if use_candidate_filter and len(candidate_indices) > top_k * 2 and self.page_summaries is not None:
            cf = CandidateFilter(top_candidates=max(top_k * 2, 20), ratio=candidate_ratio)
            sub_summaries = self.page_summaries[candidate_indices]
            top_sub = cf.filter_candidates(query_embeddings, sub_summaries, query_mask=query_mask)
            candidate_indices = [candidate_indices[idx] for idx in top_sub]

        # Step 2: Compute MaxSim scores on candidates
        q_emb = query_embeddings.to(self.device)
        q_mask = query_mask.to(self.device) if query_mask is not None else None

        scored_candidates = []
        for idx in candidate_indices:
            doc_q = self.quantized_tensors[idx]
            doc_meta = self.quantization_metas[idx]

            score = self.quantizer.compute_score(
                query_embeddings=q_emb,
                doc_quantized=doc_q,
                metadata=doc_meta,
                query_mask=q_mask,
            )
            score_val = score.item()
            scored_candidates.append((score_val, idx))

        # Step 3: Sort descending by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Step 4: Format top-k results
        results = []
        for score_val, idx in scored_candidates[:top_k]:
            results.append({
                "page_id": self.page_ids[idx],
                "document_id": self.document_ids[idx],
                "page_number": self.page_numbers[idx],
                "score": float(score_val),
                "format": self.format_name,
                "metadata": self.page_metadatas[idx],
            })

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Calculates exact in-memory storage footprint for the index."""
        quantized_bytes = sum(t.element_size() * t.numel() for t in self.quantized_tensors)
        summary_bytes = self.page_summaries.element_size() * self.page_summaries.numel() if self.page_summaries is not None else 0
        total_vectors = sum(t.shape[0] for t in self.quantized_tensors)

        # Baseline FP16 theoretical bytes for the same vectors:
        fp16_bytes = total_vectors * self.embedding_dim * 2
        fp32_bytes = total_vectors * self.embedding_dim * 4

        return {
            "format": self.format_name,
            "total_pages": len(self.page_ids),
            "total_vectors": total_vectors,
            "quantized_embedding_bytes": quantized_bytes,
            "quantized_embedding_mb": quantized_bytes / (1024 * 1024),
            "summary_bytes": summary_bytes,
            "summary_mb": summary_bytes / (1024 * 1024),
            "fp16_baseline_mb": fp16_bytes / (1024 * 1024),
            "compression_ratio_vs_fp16": (fp16_bytes / max(1, quantized_bytes)) if quantized_bytes > 0 else 1.0,
            "compression_ratio_vs_fp32": (fp32_bytes / max(1, quantized_bytes)) if quantized_bytes > 0 else 1.0,
            "device": str(self.device),
        }

    def save(self, directory: str | Path, name: Optional[str] = None) -> Path:
        """Serializes the complete index to disk."""
        target_dir = Path(directory) / (name or f"index_{self.format_name}")
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save quantized tensors
        tensors_dict = {f"doc_{i:06d}": t.cpu() for i, t in enumerate(self.quantized_tensors)}
        if self.page_summaries is not None:
            tensors_dict["summaries"] = self.page_summaries.cpu()

        torch.save(tensors_dict, target_dir / "tensors.pt")

        # Save index metadata
        index_meta = {
            "format_name": self.format_name,
            "embedding_dim": self.embedding_dim,
            "page_ids": self.page_ids,
            "document_ids": self.document_ids,
            "page_numbers": self.page_numbers,
            "quantization_metas": self.quantization_metas,
            "page_metadatas": self.page_metadatas,
            "checksum_map": self.checksum_map,
        }
        with open(target_dir / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(index_meta, f, indent=2)

        logger.info(f"Saved {self.format_name} index with {len(self.page_ids)} pages to {target_dir}")
        return target_dir

    @classmethod
    def load(cls, target_dir: str | Path, use_gpu: bool = True) -> "PageIndex":
        """Loads a serialized index from disk."""
        target_path = Path(target_dir)
        meta_file = target_path / "index_meta.json"
        tensors_file = target_path / "tensors.pt"

        if not meta_file.exists() or not tensors_file.exists():
            raise FileNotFoundError(f"Missing index files in {target_path}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        idx = cls(
            format_name=meta["format_name"],
            embedding_dim=meta["embedding_dim"],
            use_gpu=use_gpu,
        )

        idx.page_ids = meta["page_ids"]
        idx.document_ids = meta["document_ids"]
        idx.page_numbers = meta["page_numbers"]
        idx.quantization_metas = meta["quantization_metas"]
        idx.page_metadatas = meta["page_metadatas"]
        idx.checksum_map = meta["checksum_map"]
        idx.id_to_idx = {pid: i for i, pid in enumerate(idx.page_ids)}

        tensors_dict = torch.load(tensors_file, map_location=idx.device)
        idx.quantized_tensors = [tensors_dict[f"doc_{i:06d}"].to(idx.device) for i in range(len(idx.page_ids))]
        if "summaries" in tensors_dict:
            idx.page_summaries = tensors_dict["summaries"].to(idx.device)

        logger.info(f"Loaded {meta['format_name']} index with {len(idx.page_ids)} pages from {target_path}")
        return idx
