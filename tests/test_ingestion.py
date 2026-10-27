import pytest
from pathlib import Path
from PIL import Image
from visdoc.ingestion.renderer import PDFRenderer
from visdoc.ingestion.loader import DocumentLoader
from visdoc.config import RenderingConfig, IngestionConfig

def test_pdf_rendering(tmp_pdf_path):
    renderer = PDFRenderer(RenderingConfig(dpi=150))
    pages = renderer.render_pdf(tmp_pdf_path)

    assert len(pages) == 1
    page = pages[0]
    assert page.page_number == 1
    assert page.document_id == tmp_pdf_path.stem
    assert page.image.mode == "RGB"
    assert page.width > 0 and page.height > 0
    assert len(page.image_sha256) == 64

def test_image_ingestion(tmp_path, sample_pil_image):
    img_path = tmp_path / "sample.png"
    sample_pil_image.save(img_path)

    loader = DocumentLoader()
    pages = loader.load_document(img_path)

    assert len(pages) == 1
    p = pages[0]
    assert p.document_id == "sample"
    assert p.page_number == 1
    assert p.width == sample_pil_image.width

def test_duplicate_document_rejection(tmp_path, sample_pil_image):
    img_path = tmp_path / "dup.png"
    sample_pil_image.save(img_path)

    loader = DocumentLoader()
    pages1 = loader.load_document(img_path)
    assert len(pages1) == 1

    # Second ingestion of identical file without allow_duplicates
    pages2 = loader.load_document(img_path, allow_duplicates=False)
    assert len(pages2) == 0 # Rejected as duplicate
