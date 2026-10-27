import pytest
import torch
from visdoc.indexing.page_index import PageIndex
from visdoc.embeddings.generator import PageEmbedding

def test_page_index_operations(tmp_path):
    idx = PageIndex(format_name="fp16", embedding_dim=128, use_gpu=False)

    # Create dummy embeddings
    p1 = PageEmbedding(
        page_id="doc1_p0001",
        document_id="doc1",
        page_number=1,
        num_vectors=10,
        embedding_dim=128,
        embeddings=torch.randn(10, 128),
        metadata={"category": "datasheet"}
    )
    p2 = PageEmbedding(
        page_id="doc2_p0001",
        document_id="doc2",
        page_number=1,
        num_vectors=12,
        embedding_dim=128,
        embeddings=torch.randn(12, 128),
        metadata={"category": "invoice"}
    )

    idx.add_pages([p1, p2])
    assert len(idx) == 2

    # Search without filter
    q = torch.randn(4, 128)
    res = idx.search(q, top_k=2)
    assert len(res) == 2

    # Search with metadata filter
    res_filtered = idx.search(q, top_k=2, metadata_filter={"category": "datasheet"})
    assert len(res_filtered) == 1
    assert res_filtered[0]["page_id"] == "doc1_p0001"

    # Save and reload index
    save_dir = idx.save(tmp_path, name="test_idx")
    loaded_idx = PageIndex.load(save_dir, use_gpu=False)
    assert len(loaded_idx) == 2
    assert loaded_idx.page_ids == idx.page_ids
