import pytest
from fastapi.testclient import TestClient
from visdoc.api.app import create_app
from visdoc.retrieval.retriever import VisualDocumentRetriever

@pytest.fixture
def client(tmp_pdf_path):
    retriever = VisualDocumentRetriever(index_format="fp16", use_mock_model=True)
    retriever.index_documents([tmp_pdf_path])
    app = create_app(retriever=retriever)
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_model_info_endpoint(client):
    response = client.get("/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["embedding_dim"] == 128
    assert data["indexed_pages"] == 1

def test_search_endpoint(client):
    response = client.post("/v1/search", json={
        "query": "Which pin provides the 3.3V supply?",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 1
    assert data["results"][0]["page_number"] == 1
    assert "latency_ms" in data
