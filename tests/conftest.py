import sys
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_pil_image():
    """Creates a synthetic PIL RGB image with text and shapes."""
    img = Image.new("RGB", (300, 200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 100, 80], fill=(50, 100, 200), outline=(0, 0, 0))
    draw.text((30, 40), "PIN 1: 3.3V", fill=(255, 255, 255))
    draw.text((30, 60), "PIN 2: GND", fill=(255, 255, 255))
    return img

@pytest.fixture
def tmp_pdf_path(tmp_path, sample_pil_image):
    """Creates a temporary PDF file."""
    import pymupdf
    pdf_path = tmp_path / "test_doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 50), "Test Document Page 1")
    page.insert_text((50, 100), "Pin VCC: 3.3V Supply")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
