from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of ranked page results to return")
    format: Optional[str] = Field(default=None, description="Target index format: fp16, int8, or binary")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filtering criteria")
    use_candidate_filter: bool = Field(default=False, description="Enable two-stage candidate pruning")

class SearchResultItem(BaseModel):
    page_id: str
    document_id: str
    page_number: int
    score: float
    format: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    preview_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    total_results: int
    latency_ms: float
    format: str
    results: List[SearchResultItem]

class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    total_pages_rendered: int
    page_ids: List[str]
    doc_sha256: str

class ModelInfoResponse(BaseModel):
    model_name: str
    embedding_dim: int
    device: str
    dtype: str
    supported_quantizations: List[str]
    indexed_pages: int
    memory_stats: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    gpu_available: bool
