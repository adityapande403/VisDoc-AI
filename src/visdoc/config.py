from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field

class RenderingConfig(BaseModel):
    dpi: int = Field(default=150, ge=72, le=600)
    max_dimension: int = Field(default=1536, ge=256, le=4096)
    color_mode: str = Field(default="RGB")
    auto_rotate: bool = Field(default=True)
    extract_metadata: bool = Field(default=True)

class IngestionConfig(BaseModel):
    supported_formats: List[str] = Field(default=[".pdf", ".png", ".jpg", ".jpeg", ".tiff"])
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    cache_dir: str = Field(default="d:/Aditya/hf6/data/cache")
    page_image_format: str = Field(default="png")
    store_previews: bool = Field(default=True)

class ModelConfig(BaseModel):
    model_name: str = Field(default="vidore/colqwen2.5-v0.2")
    base_model: str = Field(default="vidore/colqwen2.5-base")
    fallback_model: str = Field(default="vidore/colpali-v1.2")
    embedding_dim: int = Field(default=128)
    max_query_length: int = Field(default=50)
    device: str = Field(default="cuda")
    torch_dtype: str = Field(default="bfloat16")
    hf_cache_dir: str = Field(default="d:/Aditya/hf6/.hf_cache")
    mask_non_image_embeddings: bool = Field(default=False)
    normalize_embeddings: bool = Field(default=True)

class QuantizationConfig(BaseModel):
    format: str = Field(default="fp16") # "fp16", "int8", "binary"
    int8_mode: str = Field(default="dequant")
    int8_clip_percentile: float = Field(default=99.9)
    binary_threshold: float = Field(default=0.0)

class IndexConfig(BaseModel):
    index_dir: str = Field(default="d:/Aditya/hf6/data/indexes")
    default_format: str = Field(default="fp16")
    batch_size: int = Field(default=16)
    use_gpu_scoring: bool = Field(default=True)
    top_k: int = Field(default=10)

def load_yaml(filepath: str | Path) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_default_model_config() -> ModelConfig:
    cfg_path = Path("d:/Aditya/hf6/configs/model.yaml")
    if cfg_path.exists():
        return ModelConfig(**load_yaml(cfg_path))
    return ModelConfig()

def get_default_ingestion_config() -> IngestionConfig:
    cfg_path = Path("d:/Aditya/hf6/configs/ingestion.yaml")
    if cfg_path.exists():
        data = load_yaml(cfg_path)
        storage = data.get("storage", {})
        return IngestionConfig(
            supported_formats=data.get("supported_formats", [".pdf", ".png", ".jpg", ".jpeg"]),
            rendering=RenderingConfig(**data.get("rendering", {})),
            cache_dir=storage.get("cache_dir", "d:/Aditya/hf6/data/cache"),
            page_image_format=storage.get("page_image_format", "png"),
            store_previews=storage.get("store_previews", True),
        )
    return IngestionConfig()
