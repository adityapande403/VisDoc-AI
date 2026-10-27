from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import io
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Response
from PIL import Image
import torch

from visdoc.api.schemas import (
    SearchQueryRequest,
    SearchResponse,
    SearchResultItem,
    DocumentIngestResponse,
    ModelInfoResponse,
    HealthResponse,
)
from visdoc.retrieval.retriever import VisualDocumentRetriever
from visdoc.utils.logging import logger

router = APIRouter()

# Global retriever instance holder
_retriever: Optional[VisualDocumentRetriever] = None

def get_retriever() -> VisualDocumentRetriever:
    global _retriever
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever engine has not been initialized.")
    return _retriever

def set_retriever(instance: VisualDocumentRetriever) -> None:
    global _retriever
    _retriever = instance

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Returns system health and GPU status."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        device=str(_retriever.model.device) if _retriever else "unknown",
        gpu_available=torch.cuda.is_available(),
    )

@router.get("/v1/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Returns metadata about the active visual retriever model and indexes."""
    retriever = get_retriever()
    mem_stats = retriever.index.get_memory_stats()
    return ModelInfoResponse(
        model_name=retriever.model_config.model_name,
        embedding_dim=retriever.model.embedding_dim,
        device=str(retriever.model.device),
        dtype=str(getattr(retriever.model, "dtype", "float32")),
        supported_quantizations=["fp16", "int8", "binary"],
        indexed_pages=len(retriever.index),
        memory_stats=mem_stats,
    )

@router.post("/v1/search", response_model=SearchResponse)
def search_documents(request: SearchQueryRequest) -> SearchResponse:
    """
    Multi-vector visual document retrieval using ColBERT / ColQwen Late-Interaction MaxSim.
    """
    retriever = get_retriever()
    t0 = time.perf_counter()

    results = retriever.search(
        query=request.query,
        top_k=request.top_k,
        metadata_filter=request.metadata_filter,
        use_candidate_filter=request.use_candidate_filter,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    items = [
        SearchResultItem(
            page_id=r.page_id,
            document_id=r.document_id,
            page_number=r.page_number,
            score=r.score,
            format=r.format,
            metadata=r.metadata,
            preview_url=f"/v1/pages/{r.page_id}/preview",
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query,
        total_results=len(items),
        latency_ms=elapsed_ms,
        format=retriever.index_format,
        results=items,
    )

@router.post("/v1/documents", response_model=DocumentIngestResponse)
async def upload_and_index_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
) -> DocumentIngestResponse:
    """
    Uploads a PDF or supported image file, renders pages, generates visual embeddings, and indexes them.
    """
    retriever = get_retriever()

    # Save uploaded file safely to cache
    cache_dir = Path("d:/Aditya/hf6/data/cache/uploads")
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest_file = cache_dir / file.filename
    content = await file.read()
    with open(dest_file, "wb") as f:
        f.write(content)

    doc_id = document_id or Path(file.filename).stem

    # Render pages
    pages = retriever.loader.load_document(dest_file, document_id=doc_id)
    if not pages:
        raise HTTPException(status_code=400, detail="Document could not be rendered or is a duplicate.")

    # Generate embeddings and index
    page_embs = retriever.generator.generate_embeddings(pages)
    retriever.index.add_pages(page_embs, allow_duplicates=True)

    return DocumentIngestResponse(
        document_id=doc_id,
        filename=file.filename,
        total_pages_rendered=len(pages),
        page_ids=[p.page_id for p in pages],
        doc_sha256=pages[0].metadata.get("doc_sha256", ""),
    )

@router.get("/v1/documents/{document_id}")
def get_document_pages(document_id: str) -> Dict[str, Any]:
    """Retrieves all indexed page IDs and metadata for a specific document."""
    retriever = get_retriever()
    matched_indices = [
        i for i, did in enumerate(retriever.index.document_ids)
        if did == document_id
    ]
    if not matched_indices:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found in index.")

    return {
        "document_id": document_id,
        "total_pages": len(matched_indices),
        "page_ids": [retriever.index.page_ids[i] for i in matched_indices],
        "page_numbers": [retriever.index.page_numbers[i] for i in matched_indices],
    }

@router.delete("/v1/documents/{document_id}")
def delete_document(document_id: str) -> Dict[str, Any]:
    """Removes a document from the index."""
    retriever = get_retriever()
    to_keep = [i for i, did in enumerate(retriever.index.document_ids) if did != document_id]
    deleted_count = len(retriever.index.document_ids) - len(to_keep)

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    # Filter index lists
    retriever.index.page_ids = [retriever.index.page_ids[i] for i in to_keep]
    retriever.index.document_ids = [retriever.index.document_ids[i] for i in to_keep]
    retriever.index.page_numbers = [retriever.index.page_numbers[i] for i in to_keep]
    retriever.index.quantized_tensors = [retriever.index.quantized_tensors[i] for i in to_keep]
    retriever.index.quantization_metas = [retriever.index.quantization_metas[i] for i in to_keep]
    retriever.index.page_metadatas = [retriever.index.page_metadatas[i] for i in to_keep]
    if retriever.index.page_summaries is not None and len(to_keep) > 0:
        retriever.index.page_summaries = retriever.index.page_summaries[to_keep]

    return {"status": "deleted", "document_id": document_id, "deleted_pages": deleted_count}
