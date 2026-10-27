"""
VisDoc AI - Multimodal Visual Document Retrieval with ColPali, ColQwen and Late Interaction.
"""

__version__ = "0.1.0"

from visdoc.ingestion.loader import DocumentLoader
from visdoc.ingestion.renderer import PDFRenderer
from visdoc.scoring.maxsim import compute_maxsim_scores, exact_maxsim
from visdoc.quantization.base import BaseQuantizer
from visdoc.quantization.fp16 import FP16Quantizer
from visdoc.quantization.int8 import INT8Quantizer
from visdoc.quantization.binary import BinaryQuantizer
from visdoc.indexing.page_index import PageIndex
from visdoc.retrieval.retriever import VisualDocumentRetriever

__all__ = [
    "DocumentLoader",
    "PDFRenderer",
    "compute_maxsim_scores",
    "exact_maxsim",
    "BaseQuantizer",
    "FP16Quantizer",
    "INT8Quantizer",
    "BinaryQuantizer",
    "PageIndex",
    "VisualDocumentRetriever",
]
