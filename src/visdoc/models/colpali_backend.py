import os
from pathlib import Path
from typing import List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image

from visdoc.models.base import BaseVisualRetriever
from visdoc.utils.logging import logger

class ColQwenModelBackend(BaseVisualRetriever):
    """
    Production backend for ColQwen2.5 (vidore/colqwen2.5-v0.2) multi-vector retrieval.
    """

    def __init__(
        self,
        model_name_or_path: str = "vidore/colqwen2.5-v0.2",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        hf_cache_dir: Optional[str] = "d:/Aditya/hf6/.hf_cache",
        mask_non_image_embeddings: bool = False,
    ):
        # Guarantee HF cache is set on D: drive
        if hf_cache_dir:
            os.environ["HF_HOME"] = str(Path(hf_cache_dir).resolve())

        # Determine target device
        if device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
            if device == "cuda":
                logger.warning("CUDA requested but not available. Falling back to CPU.")

        # Determine dtype
        if self._device.type == "cuda":
            if torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info(f"Loading ColQwen2.5 model from '{model_name_or_path}' on {self._device} ({self.dtype})...")

        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self.processor = ColQwen2_5_Processor.from_pretrained(
            model_name_or_path,
            cache_dir=hf_cache_dir,
        )
        self.model = ColQwen2_5.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            cache_dir=hf_cache_dir,
            mask_non_image_embeddings=mask_non_image_embeddings,
        ).to(self._device)
        self.model.eval()

        self._embedding_dim = getattr(self.model, "dim", 128)
        logger.info(f"ColQwen2.5 model loaded successfully. Embedding dimension: {self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.inference_mode()
    def encode_pages(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Encodes document page images into lists of multi-vector embeddings.
        Each page tensor has shape (N_d, 128) and is L2-normalized.
        """
        if not images:
            return []

        processed = self.processor.process_images(images=images)
        processed = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}

        output_embs = self.model(**processed) # (batch_size, sequence_length, dim)

        # Unbind by batch
        results: List[torch.Tensor] = []
        attn_mask = processed.get("attention_mask", None)

        for i in range(len(images)):
            page_emb = output_embs[i] # (sequence_length, dim)
            if attn_mask is not None:
                valid_mask = attn_mask[i].bool()
                page_emb = page_emb[valid_mask] # Keep only valid tokens
            # Ensure unit L2 normalization
            page_emb = F.normalize(page_emb.float(), p=2, dim=-1)
            results.append(page_emb.cpu())

        return results

    @torch.inference_mode()
    def encode_queries(self, queries: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes queries into (B_q, N_q, D) embeddings and (B_q, N_q) boolean mask.
        """
        if not queries:
            return torch.empty(0, 0, self._embedding_dim), torch.empty(0, 0, dtype=torch.bool)

        processed = self.processor.process_queries(queries=queries)
        processed = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}

        output_embs = self.model(**processed) # (B_q, N_q, dim)
        output_embs = F.normalize(output_embs.float(), p=2, dim=-1)

        mask = processed.get("attention_mask", torch.ones(output_embs.shape[:2], device=self._device)).bool()
        return output_embs.cpu(), mask.cpu()


class MockVisualRetrieverModel(BaseVisualRetriever):
    """
    Deterministic, zero-download mock retriever for fast unit testing and CI.
    Generates reproducible pseudo-embeddings based on visual and textual features.
    """

    def __init__(self, embedding_dim: int = 128, device: str = "cpu"):
        self._embedding_dim = embedding_dim
        self._device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_pages(self, images: List[Image.Image]) -> List[torch.Tensor]:
        results = []
        for img in images:
            # Deterministic seed based on image dimensions and pixel sum
            w, h = img.size
            # Sample small thumbnail to get deterministic hash
            thumb = img.resize((16, 16)).convert("L")
            pixel_sum = sum(thumb.tobytes())
            generator = torch.Generator().manual_seed((w * 1000 + h + pixel_sum) % (2**31 - 1))

            num_tokens = 64 # simulated patch tokens
            embs = torch.randn(num_tokens, self._embedding_dim, generator=generator)
            embs = F.normalize(embs, p=2, dim=-1)
            results.append(embs)
        return results

    def encode_queries(self, queries: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not queries:
            return torch.empty(0, 0, self._embedding_dim), torch.empty(0, 0, dtype=torch.bool)

        batch_embs = []
        masks = []
        max_len = 16

        for q in queries:
            # Deterministic seed based on query string
            q_hash = sum(ord(c) * (i + 1) for i, c in enumerate(q))
            generator = torch.Generator().manual_seed(q_hash % (2**31 - 1))

            tokens = max(1, min(len(q.split()), max_len))
            embs = torch.randn(tokens, self._embedding_dim, generator=generator)
            embs = F.normalize(embs, p=2, dim=-1)

            # Pad to max_len
            padded = torch.zeros(max_len, self._embedding_dim)
            padded[:tokens] = embs
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[:tokens] = True

            batch_embs.append(padded)
            masks.append(mask)

        return torch.stack(batch_embs), torch.stack(masks)


def load_visual_retriever(
    model_name: str = "vidore/colqwen2.5-v0.2",
    use_mock: bool = False,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    hf_cache_dir: Optional[str] = "d:/Aditya/hf6/.hf_cache",
) -> BaseVisualRetriever:
    """Factory function for visual retriever models."""
    if use_mock:
        logger.info("Initializing MockVisualRetrieverModel for testing...")
        return MockVisualRetrieverModel(embedding_dim=128, device=device)
    else:
        return ColQwenModelBackend(
            model_name_or_path=model_name,
            device=device,
            torch_dtype=torch_dtype,
            hf_cache_dir=hf_cache_dir,
        )
