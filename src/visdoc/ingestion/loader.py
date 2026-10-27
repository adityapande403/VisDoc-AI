from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image
from visdoc.config import IngestionConfig, get_default_ingestion_config
from visdoc.ingestion.renderer import PDFRenderer, RenderedPage
from visdoc.utils.checksum import compute_file_sha256, compute_image_sha256
from visdoc.utils.logging import logger

class DocumentLoader:
    """Document Ingestion Engine supporting PDFs and standard image formats."""

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or get_default_ingestion_config()
        self.pdf_renderer = PDFRenderer(self.config.rendering)
        self.seen_doc_checksums: Dict[str, str] = {} # sha256 -> document_id

    def load_document(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        allow_duplicates: bool = False
    ) -> List[RenderedPage]:
        """
        Loads a document file (PDF, PNG, JPG, JPEG, TIFF) and returns list of RenderedPage objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.config.supported_formats:
            raise ValueError(f"Unsupported format '{suffix}'. Supported formats: {self.config.supported_formats}")

        file_sha256 = compute_file_sha256(path)
        if not allow_duplicates and file_sha256 in self.seen_doc_checksums:
            prev_id = self.seen_doc_checksums[file_sha256]
            logger.warning(f"Skipping duplicate document {path.name} (already ingested as {prev_id})")
            return []

        doc_id = document_id or path.stem

        if suffix == ".pdf":
            pages = self.pdf_renderer.render_pdf(
                path,
                document_id=doc_id,
                extra_metadata=extra_metadata
            )
        else:
            # Image file ingestion (PNG, JPG, TIFF)
            try:
                img = Image.open(path).convert(self.config.rendering.color_mode)
                max_dim = max(img.width, img.height)
                if max_dim > self.config.rendering.max_dimension:
                    scale = self.config.rendering.max_dimension / float(max_dim)
                    new_w = int(img.width * scale)
                    new_h = int(img.height * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                img_sha256 = compute_image_sha256(img)
                page_id = f"{doc_id}_p0001"
                page_meta = {
                    "source_path": str(path.resolve()),
                    "doc_sha256": file_sha256,
                    "total_pages": 1,
                    **(extra_metadata or {})
                }
                pages = [
                    RenderedPage(
                        page_id=page_id,
                        document_id=doc_id,
                        page_number=1,
                        image=img,
                        width=img.width,
                        height=img.height,
                        image_sha256=img_sha256,
                        dpi=self.config.rendering.dpi,
                        metadata=page_meta
                    )
                ]
            except Exception as e:
                logger.error(f"Error loading image document {path}: {e}")
                raise

        self.seen_doc_checksums[file_sha256] = doc_id
        return pages

    def load_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        allow_duplicates: bool = False
    ) -> List[RenderedPage]:
        """Loads all supported documents from a directory."""
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        pattern = "**/*" if recursive else "*"
        all_pages: List[RenderedPage] = []

        files = sorted([p for p in dir_path.glob(pattern) if p.is_file() and p.suffix.lower() in self.config.supported_formats])
        logger.info(f"Discovered {len(files)} document files in {dir_path}")

        for f in files:
            try:
                pages = self.load_document(f, allow_duplicates=allow_duplicates)
                all_pages.extend(pages)
            except Exception as e:
                logger.error(f"Skipping failed file {f}: {e}")

        logger.info(f"Successfully ingested {len(all_pages)} total pages from {len(files)} files")
        return all_pages
