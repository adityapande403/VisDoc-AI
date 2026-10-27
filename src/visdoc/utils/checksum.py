import hashlib
from pathlib import Path
from typing import Union
import io
from PIL import Image

def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA256 checksum of raw bytes."""
    return hashlib.sha256(data).hexdigest()

def compute_image_sha256(image: Image.Image) -> str:
    """Compute deterministic SHA256 of a PIL Image."""
    buffer = io.BytesIO()
    # Save as PNG to ensure lossless deterministic bytes
    image.save(buffer, format="PNG")
    return compute_bytes_sha256(buffer.getvalue())
