import pytest
from pathlib import Path
from visdoc.retrieval.retriever import VisualDocumentRetriever

def test_visual_document_retriever_pipeline(tmp_pdf_path):
    retriever = VisualDocumentRetriever(
        index_format="fp16",
        use_mock_model=True
    )

    added = retriever.index_documents([tmp_pdf_path])
    assert added == 1

    results = retriever.search("Find 3.3V supply pin", top_k=5)
    assert len(results) == 1
    top = results[0]
    assert top.page_number == 1
    assert top.document_id == tmp_pdf_path.stem
    assert isinstance(top.score, float)
