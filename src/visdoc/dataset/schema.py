from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

class RetrievalExample(BaseModel):
    """
    Standardized schema for visual retrieval queries, positive targets, and hard negatives.
    """
    query_id: str = Field(..., description="Unique query identifier (e.g. query_000001)")
    query: str = Field(..., min_length=2, description="Natural language query string")
    positive_page_ids: List[str] = Field(..., min_length=1, description="List of ground truth relevant page IDs")
    negative_page_ids: List[str] = Field(default_factory=list, description="Mined or assigned hard negative page IDs")
    query_type: str = Field(
        default="factual_lookup",
        description="Query classification category (e.g. pinout_lookup, financial_table, schematic_wiring)"
    )
    language: str = Field(default="en", description="ISO 639-1 language code")
    source_document_id: str = Field(..., description="ID of the primary source document")
    split: Literal["train", "val", "test"] = Field(default="train", description="Dataset split")
    source_license: str = Field(default="MIT", description="Data provenance licence identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary domain metadata")

    @field_validator("positive_page_ids")
    @classmethod
    def validate_positive_pages(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("positive_page_ids must contain at least one page ID")
        return v

    @field_validator("query")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        clean = v.strip()
        if len(clean) < 3:
            raise ValueError("Query text must be at least 3 characters long")
        return clean

class DatasetManifest(BaseModel):
    """Dataset summary manifest with provenance and statistics."""
    dataset_version: str = "1.0.0"
    created_at: str
    total_queries: int
    total_documents: int
    total_pages: int
    categories: Dict[str, int]
    splits: Dict[str, int]
    provenance: Dict[str, Any]
