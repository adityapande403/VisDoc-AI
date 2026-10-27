from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import yaml
import torch
from PIL import Image

from visdoc.config import get_default_model_config, get_default_ingestion_config, ModelConfig, IngestionConfig
from visdoc.ingestion.loader import DocumentLoader
from visdoc.ingestion.renderer import RenderedPage
from visdoc.models.base import BaseVisualRetriever
from visdoc.models.colpali_backend import load_visual_retriever, MockVisualRetrieverModel
from visdoc.embeddings.generator import EmbeddingGenerator, PageEmbedding
from visdoc.indexing.page_index import PageIndex
from visdoc.utils.logging import logger

@dataclass
class SearchResult:
    """Standardized search result item returned by VisualDocumentRetriever."""
    page_id: str
    document_id: str
    page_number: int
    score: float
    format: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_image_path: Optional[str] = None

    def __repr__(self) -> str:
        return f"<SearchResult doc='{self.document_id}' page={self.page_number} score={self.score:.4f} format='{self.format}'>"

class VisualDocumentRetriever:
    """
    Main Visual Document Retriever engine with multi-vector late interaction.
    """

    def __init__(
        self,
        retriever_model: Optional[BaseVisualRetriever] = None,
        model_config: Optional[ModelConfig] = None,
        ingestion_config: Optional[IngestionConfig] = None,
        index_format: str = "fp16",
        index_dir: Optional[str | Path] = None,
        use_mock_model: bool = False,
    ):
        self.model_config = model_config or get_default_model_config()
        self.ingestion_config = ingestion_config or get_default_ingestion_config()
        self.index_format = index_format.lower()
        self.index_dir = Path(index_dir) if index_dir else Path("d:/Aditya/hf6/data/indexes")

        # Initialize vision-language retrieval model backend
        if retriever_model is not None:
            self.model = retriever_model
        else:
            self.model = load_visual_retriever(
                model_name=self.model_config.model_name,
                use_mock=use_mock_model,
                device=self.model_config.device,
                torch_dtype=self.model_config.torch_dtype,
                hf_cache_dir=self.model_config.hf_cache_dir,
            )

        # Ingestion loader
        self.loader = DocumentLoader(self.ingestion_config)

        # Multi-vector index
        self.index = PageIndex(
            format_name=self.index_format,
            index_dir=self.index_dir,
            embedding_dim=self.model.embedding_dim,
            use_gpu=(self.model_config.device == "cuda"),
        )

        # Generator
        self.generator = EmbeddingGenerator(
            retriever=self.model,
            quantizer=self.index.quantizer,
            batch_size=4,
        )

    @classmethod
    def from_config(
        cls,
        config_path_or_dict: Union[str, Path, Dict[str, Any]],
        use_mock: bool = False
    ) -> "VisualDocumentRetriever":
        """Factory method to construct retriever from a configuration YAML or dict."""
        if isinstance(config_path_or_dict, (str, Path)):
            path = Path(config_path_or_dict)
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
        else:
            cfg_data = config_path_or_dict

        index_format = cfg_data.get("index_format", cfg_data.get("default_format", "fp16"))
        index_dir = cfg_data.get("index_dir", "d:/Aditya/hf6/data/indexes")

        return cls(
            index_format=index_format,
            index_dir=index_dir,
            use_mock_model=use_mock,
        )

    def index_documents(
        self,
        paths: List[Union[str, Path]],
        batch_size: int = 4,
        allow_duplicates: bool = False,
    ) -> int:
        """
        Ingests and indexes document files (PDFs, images).
        Returns total number of newly indexed pages.
        """
        all_pages: List[RenderedPage] = []
        for p in paths:
            path_obj = Path(p)
            if path_obj.is_dir():
                pages = self.loader.load_directory(path_obj, allow_duplicates=allow_duplicates)
            else:
                pages = self.loader.load_document(path_obj, allow_duplicates=allow_duplicates)
            all_pages.extend(pages)

        if not all_pages:
            logger.warning("No new pages found to index.")
            return 0

        # Generate multi-vector embeddings
        page_embs = self.generator.generate_embeddings(all_pages)

        # Add to index
        added = self.index.add_pages(page_embs, allow_duplicates=allow_duplicates)
        logger.info(f"Successfully indexed {added} pages into {self.index_format.upper()} index.")
        return added

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_candidate_filter: bool = False,
    ) -> List[SearchResult]:
        """
        Encodes the text query and performs multi-vector late-interaction MaxSim retrieval.
        """
        if not query or not query.strip():
            return []

        # Encode query
        query_embs, query_masks = self.model.encode_queries([query.strip()])
        q_emb = query_embs[0] # (N_q, D)
        q_mask = query_masks[0] # (N_q,)

        # Query index
        raw_results = self.index.search(
            query_embeddings=q_emb,
            query_mask=q_mask,
            top_k=top_k,
            metadata_filter=metadata_filter,
            use_candidate_filter=use_candidate_filter,
        )

        # Convert to SearchResult objects
        results = [
            SearchResult(
                page_id=r["page_id"],
                document_id=r["document_id"],
                page_number=r["page_number"],
                score=r["score"],
                format=r["format"],
                metadata=r["metadata"],
                page_image_path=r["metadata"].get("source_path"),
            )
            for r in raw_results
        ]
        return results

    def save_index(self, name: Optional[str] = None) -> Path:
        """Saves current index to disk."""
        return self.index.save(self.index_dir, name=name)

    def load_index(self, directory: str | Path) -> None:
        """Loads an index from disk."""
        self.index = PageIndex.load(directory, use_gpu=(self.model_config.device == "cuda"))
        self.index_format = self.index.format_name
