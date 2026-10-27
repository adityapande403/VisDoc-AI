from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import torch
import safetensors.torch
from visdoc.embeddings.generator import PageEmbedding
from visdoc.utils.logging import logger

class EmbeddingStorage:
    """Handles serialization, disk persistence, and memory-mapping of multi-vector embeddings."""

    @staticmethod
    def save_embeddings(
        embeddings: List[PageEmbedding],
        output_dir: str | Path,
        index_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Saves page embeddings and metadata to an index directory.
        Separates high-dimensional tensors (safetensors) from metadata (json).
        """
        out_path = Path(output_dir) / index_name
        out_path.mkdir(parents=True, exist_ok=True)

        tensor_dict: Dict[str, torch.Tensor] = {}
        meta_list: List[Dict[str, Any]] = []

        total_embedding_bytes = 0
        total_quantized_bytes = 0

        for idx, pe in enumerate(embeddings):
            # Key for tensors
            emb_key = f"emb_{idx:06d}"
            tensor_dict[emb_key] = pe.embeddings
            total_embedding_bytes += pe.embeddings.element_size() * pe.embeddings.numel()

            q_key = None
            if pe.quantized_tensor is not None:
                q_key = f"q_{idx:06d}"
                tensor_dict[q_key] = pe.quantized_tensor
                total_quantized_bytes += pe.quantized_tensor.element_size() * pe.quantized_tensor.numel()

            meta_list.append({
                "page_id": pe.page_id,
                "document_id": pe.document_id,
                "page_number": pe.page_number,
                "num_vectors": pe.num_vectors,
                "embedding_dim": pe.embedding_dim,
                "emb_key": emb_key,
                "q_key": q_key,
                "quantization_meta": pe.quantization_meta,
                "metadata": pe.metadata,
            })

        # Save tensors
        tensors_file = out_path / "embeddings.safetensors"
        safetensors.torch.save_file(tensor_dict, str(tensors_file))

        # Save metadata
        metadata_file = out_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2)

        # Record storage manifest
        safetensors_bytes = tensors_file.stat().st_size
        metadata_bytes = metadata_file.stat().st_size
        total_disk_bytes = safetensors_bytes + metadata_bytes

        manifest = {
            "index_name": index_name,
            "total_pages": len(embeddings),
            "safetensors_bytes": safetensors_bytes,
            "metadata_bytes": metadata_bytes,
            "total_disk_bytes": total_disk_bytes,
            "raw_embedding_bytes": total_embedding_bytes,
            "quantized_embedding_bytes": total_quantized_bytes,
            "tensors_path": str(tensors_file),
            "metadata_path": str(metadata_file),
        }

        manifest_file = out_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Saved index '{index_name}' with {len(embeddings)} pages. "
            f"Total on-disk: {total_disk_bytes / (1024*1024):.2f} MB "
            f"(Safetensors: {safetensors_bytes / (1024*1024):.2f} MB, Metadata: {metadata_bytes / 1024:.2f} KB)"
        )
        return manifest

    @staticmethod
    def load_embeddings(
        index_dir: str | Path,
        index_name: str = "default",
        device: str = "cpu"
    ) -> List[PageEmbedding]:
        """Loads embeddings and metadata from an index directory."""
        dir_path = Path(index_dir) / index_name
        metadata_file = dir_path / "metadata.json"
        tensors_file = dir_path / "embeddings.safetensors"

        if not metadata_file.exists() or not tensors_file.exists():
            raise FileNotFoundError(f"Index files not found in {dir_path}")

        with open(metadata_file, "r", encoding="utf-8") as f:
            meta_list = json.load(f)

        tensor_dict = safetensors.torch.load_file(str(tensors_file), device=device)

        page_embeddings: List[PageEmbedding] = []
        for m in meta_list:
            emb = tensor_dict[m["emb_key"]]
            q_tensor = tensor_dict[m["q_key"]] if m.get("q_key") in tensor_dict else None

            pe = PageEmbedding(
                page_id=m["page_id"],
                document_id=m["document_id"],
                page_number=m["page_number"],
                num_vectors=m["num_vectors"],
                embedding_dim=m["embedding_dim"],
                embeddings=emb,
                quantized_tensor=q_tensor,
                quantization_meta=m.get("quantization_meta", {}),
                metadata=m.get("metadata", {}),
            )
            page_embeddings.append(pe)

        logger.info(f"Loaded {len(page_embeddings)} page embeddings from {dir_path}")
        return page_embeddings
