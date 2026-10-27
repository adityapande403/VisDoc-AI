import pytest
import torch
from visdoc.dataset.schema import RetrievalExample
from visdoc.dataset.negative_miner import HardNegativeMiner
from visdoc.indexing.page_index import PageIndex
from visdoc.embeddings.generator import PageEmbedding
from visdoc.models.colpali_backend import MockVisualRetrieverModel

def test_hard_negative_miner():
    model = MockVisualRetrieverModel(embedding_dim=128)
    idx = PageIndex(format_name="fp16", embedding_dim=128, use_gpu=False)

    # Index 3 pages
    for i in range(3):
        pe = PageEmbedding(
            page_id=f"doc_{i}_p0001",
            document_id=f"doc_{i}",
            page_number=1,
            num_vectors=16,
            embedding_dim=128,
            embeddings=torch.randn(16, 128),
        )
        idx.add_pages([pe])

    miner = HardNegativeMiner(retriever_model=model, page_index=idx, max_negatives_per_query=2)

    ex = RetrievalExample(
        query_id="q1",
        query="Test query",
        positive_page_ids=["doc_0_p0001"],
        source_document_id="doc_0",
    )

    mined = miner.mine_negatives_for_example(ex)
    assert len(mined) <= 2
    # Ensure known positive is NEVER in mined negatives
    for m in mined:
        assert m.page_id != "doc_0_p0001"
