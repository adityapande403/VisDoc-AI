from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import pymupdf  # PyMuPDF
from PIL import Image
from visdoc.config import RenderingConfig
from visdoc.utils.checksum import compute_image_sha256, compute_file_sha256
from visdoc.utils.logging import logger

@dataclass
class RenderedPage:
    """Represents a rendered document page with image, metadata, and identifiers."""
    page_id: str
    document_id: str
    page_number: int # 1-indexed
    image: Image.Image
    width: int
    height: int
    image_sha256: str
    dpi: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "image_sha256": self.image_sha256,
            "dpi": self.dpi,
            "metadata": self.metadata,
        }

class PDFRenderer:
    """High-quality PDF page rendering using PyMuPDF."""

    def __init__(self, config: Optional[RenderingConfig] = None):
        self.config = config or RenderingConfig()

    def render_pdf(
        self,
        pdf_path: Union[str, Path],
        document_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RenderedPage]:
        """
        Renders all pages of a PDF into RenderedPage objects.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc_sha256 = compute_file_sha256(pdf_path)
        doc_id = document_id or pdf_path.stem

        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            raise

        doc_meta = doc.metadata or {}
        pages: List[RenderedPage] = []

        # PyMuPDF 72 DPI is standard 1.0 zoom. Calculate zoom factor for target DPI.
        zoom = self.config.dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Auto rotation handling
            if self.config.auto_rotate and page.rotation != 0:
                # pyMuPDF automatically applies rotation in get_pixmap unless matrix explicitly counters it
                pass

            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                # Convert directly to PIL Image in RGB
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Resize if exceeding max_dimension while preserving aspect ratio
                max_dim = max(img.width, img.height)
                if max_dim > self.config.max_dimension:
                    scale = self.config.max_dimension / float(max_dim)
                    new_w = int(img.width * scale)
                    new_h = int(img.height * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                page_sha256 = compute_image_sha256(img)
                page_id = f"{doc_id}_p{page_num:04d}"

                page_meta = {
                    "source_path": str(pdf_path.resolve()),
                    "doc_sha256": doc_sha256,
                    "title": doc_meta.get("title", ""),
                    "author": doc_meta.get("author", ""),
                    "total_pages": len(doc),
                    **(extra_metadata or {})
                }

                pages.append(
                    RenderedPage(
                        page_id=page_id,
                        document_id=doc_id,
                        page_number=page_num,
                        image=img,
                        width=img.width,
                        height=img.height,
                        image_sha256=page_sha256,
                        dpi=self.config.dpi,
                        metadata=page_meta
                    )
                )
            except Exception as e:
                logger.error(f"Error rendering page {page_num} of {pdf_path}: {e}")
                raise

        doc.close()
        logger.info(f"Rendered {len(pages)} pages from {pdf_path.name} at {self.config.dpi} DPI")
        return pages
