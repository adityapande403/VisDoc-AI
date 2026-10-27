# Deployment & Security Guide

## 1. Localhost-First Privacy Architecture

VisDoc AI is designed for strictly air-gapped, zero-egress visual document search:
- **Zero Cloud Leakage**: All inference runs locally on the host machine.
- **Localhost Default**: REST API binds exclusively to `127.0.0.1:8000`.
- **Cache Isolation**: All models and temporary artifacts are anchored to `D:\Aditya\hf6\.hf_cache` via `HF_HOME`.

---

## 2. Python SDK Usage

```python
from visdoc import VisualDocumentRetriever

# Initialize retriever (FP16, INT8, or Binary)
retriever = VisualDocumentRetriever(
    index_format="int8", # 2.0x compressed index
)

# Index PDF documents or image files
retriever.index_documents(
    paths=["./data/documents/doc_eng_datasheet_001.pdf"]
)

# Execute visual late-interaction search
results = retriever.search(
    query="Which pin provides the 5.0 V analog sensor supply?",
    top_k=5,
)

for result in results:
    print(f"Document: {result.document_id} | Page: {result.page_number} | Score: {result.score:.4f}")
```

---

## 3. REST API Reference

Launch the server:
```bash
uvicorn visdoc.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

### Endpoints
- `GET /health`: Health and GPU status.
- `GET /v1/model-info`: Active model backend, embedding dimension, index stats.
- `POST /v1/search`: Visual late-interaction search.
  ```json
  {
    "query": "Find turbine inspection certificate with approved stamp",
    "top_k": 5,
    "use_candidate_filter": true
  }
  ```
- `POST /v1/documents`: Upload and index a PDF or image file (multipart form data).
- `GET /v1/documents/{document_id}`: Inspect indexed page IDs for a document.
- `DELETE /v1/documents/{document_id}`: Remove a document from the index.
