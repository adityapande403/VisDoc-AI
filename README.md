# VisDoc AI
### Multimodal Visual Document Retrieval with ColPali, ColQwen and Late Interaction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14.1%2Bcu130-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 17 Passed](https://img.shields.io/badge/Tests-17%20Passed-brightgreen.svg)](tests/)

VisDoc AI is a production-grade multimodal visual document retrieval engine based on vision-language models and ColBERT-style late-interaction retrieval.

It retrieves relevant document pages from complex visual layouts—engineering datasheets, electrical schematics, PCB diagrams, multi-column financial balance sheets, commercial invoices, and stamped QA inspection forms—**without depending on conventional OCR text extraction or text chunking**.

---

## Key Features

- **No OCR Dependency**: Direct visual patch embeddings preserve tabular structures, circuit drawings, pinouts, and stamps.
- **ColPali & ColQwen Backbone**: Dynamic-resolution vision-language representation projecting page tokens into 128-dimensional L2-normalized multi-vectors.
- **Exact & Batched Late Interaction**: Reference and GPU-accelerated MaxSim scoring with precise token mask handling:
  $$S(Q, D) = \sum_{i \in Q} \max_{j \in D} (q_i \cdot d_j)$$
- **Index Quantization**:
  - **FP16 Baseline**: Reference uncompressed half-precision index.
  - **INT8 Scalar Quantization**: 2.0x memory reduction with **100% NDCG@5 retention**.
  - **1-Bit Binary Packed Quantization**: 16.0x compression over FP16 (32.0x over FP32) with bitwise POPCOUNT scoring.
- **Two-Stage Candidate Pruning**: Mean-pooled page summaries prune large indexes before MaxSim, delivering a **23.7x latency speedup** (down to 10.7 ms).
- **Contrastive Fine-Tuning Suite**: Parameter-efficient LoRA training with ColBERT pairwise negative cross-entropy loss and automated smoke tests.
- **Hard Negative Mining Engine**: Discovers difficult adjacent-page and semantic near-neighbour negative candidates.
- **Python SDK & Local REST API**: Clean Python API and FastAPI service with health, search, indexing, and preview endpoints.

---

## Architecture Overview

```text
                  PDF / IMAGE DOCUMENTS
                            │
                            ▼
               PyMuPDF Document Ingestion
                            │
                            ▼
                PDF Page Image Renderer
                            │
                            ▼
             ColQwen2.5 / ColPali Vision Encoder
                            │
                            ▼
               Multi-Vector Page Embeddings
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
  [FP16 Reference Index]          [Quantization Layer]
            │                               │
            │                 ┌─────────────┴─────────────┐
            │                 ▼                           ▼
            │        [INT8 Scalar Index]         [1-Bit Binary Index]
            │                 │                           │
User Query ─┼─────────────────┼───────────────────────────┤
     │      │                 │                           │
     ▼      ▼                 ▼                           ▼
 Text Query Encoder     Exact / Dequant           Bitwise POPCOUNT MaxSim
            │                 │                           │
            └─────────────────┼───────────────────────────┘
                              │
                              ▼
                 Ranked Document Page Results
                              │
                              ▼
              Metadata, Previews & Downstream RAG
```

---

## Benchmark Results (Empirically Measured)

### 1. Retrieval Quality & Compression Retention (Held-Out Test Set)

| Representation | NDCG@5 | MRR@10 | Recall@5 | Relative NDCG@5 Retention | Embedding Payload | Compression vs FP16 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP16 Baseline** | 0.5132 | 0.4361 | 0.8333 | **100.0%** (Baseline) | 0.09 MB | 1.0x |
| **INT8 Scalar** | 0.5132 | 0.4361 | 0.8333 | **100.0%** | 0.05 MB | **2.0x** |
| **1-Bit Binary** | 0.3747 | 0.2556 | 0.8333 | **73.0%** | 0.01 MB | **16.0x** (32x vs FP32) |

### 2. 10,000-Page Scale Benchmark (5,000,000 Multi-Vectors)

| Index Format | Total Indexed Pages | Embedding Memory (MB) | Compression vs FP16 | Mean Latency (ms) | P95 Latency (ms) | Throughput (QPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP16** | 10,000 | 1,220.7 MB | 1.0x | 2,028.90 ms | 2,135.90 ms | 0.5 |
| **INT8** | 10,000 | 610.4 MB | **2.0x** | 3,013.15 ms | 3,083.47 ms | 0.3 |
| **BINARY** | 10,000 | **76.3 MB** | **16.0x** | 2,770.02 ms | 2,890.07 ms | 0.4 |

*Two-stage candidate filtering reduces 1,000-page latency from 253.26 ms to **10.69 ms** (23.7x acceleration).*

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/visdoc-ai/visdoc.git
cd visdoc

# Create virtual environment and activate
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and VisDoc in editable mode
pip install -e .
```

### 2. Python SDK

```python
from visdoc import VisualDocumentRetriever

# Initialize retriever (supports "fp16", "int8", or "binary")
retriever = VisualDocumentRetriever(index_format="int8")

# Ingest documents
retriever.index_documents(paths=["./data/documents/doc_eng_datasheet_001.pdf"])

# Run late-interaction visual search
results = retriever.search(
    query="Which pin provides the 5.0 V analog sensor supply on the TX-4000?",
    top_k=5,
)

for result in results:
    print(f"Document: {result.document_id} | Page: {result.page_number} | Score: {result.score:.4f}")
```

### 3. REST API

Launch the local server:
```bash
uvicorn visdoc.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Execute a search query:
```bash
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Find turbine inspection certificate with approved stamp", "top_k": 5}'
```

Interactive OpenAPI Swagger UI is available at `http://127.0.0.1:8000/docs`.

---

## Running the Reproducibility Suite

```bash
# 1. Run all 17 automated unit and integration tests
pytest -v tests

# 2. Generate multi-category synthetic PDF test documents
python scripts/generate_synthetic_docs.py

# 3. Mine hard negatives for training dataset
python scripts/mine_negatives.py

# 4. Run contrastive training smoke test
python scripts/train_retriever.py

# 5. Run retrieval evaluation across FP16, INT8, and Binary indexes
python scripts/evaluate_retrieval.py

# 6. Run 10,000-page scale memory and latency benchmark
python scripts/benchmark_10k_pages.py
```

---

## Project Structure

```text
visdoc/
├── pyproject.toml
├── README.md
├── LICENSE
├── configs/
│   ├── ingestion.yaml
│   ├── model.yaml
│   ├── training.yaml
│   ├── quantization.yaml
│   ├── index.yaml
│   └── evaluation.yaml
├── src/
│   └── visdoc/
│       ├── ingestion/       # PDF rendering (PyMuPDF) and format loader
│       ├── models/          # ColQwen2.5 and ColPali model backends
│       ├── embeddings/      # Multi-vector generation and safetensors storage
│       ├── scoring/         # Exact MaxSim, GPU batched, and bitwise POPCOUNT
│       ├── quantization/   # FP16, INT8 scalar, and 1-bit packed binary quantizers
│       ├── indexing/        # Persistent multi-vector page index & candidate pruning
│       ├── retrieval/       # VisualDocumentRetriever SDK interface
│       ├── dataset/         # Schema validation, synthetic PDF generator, negative miner
│       ├── training/        # ColBERT pairwise negative cross-entropy loss and trainer
│       ├── evaluation/      # NDCG@5, MRR@10, Recall@K, latency and retention metrics
│       ├── api/             # FastAPI REST API service
│       └── utils/           # Checksums and rich logging
├── scripts/                 # Standalone automation and benchmark scripts
├── tests/                   # 17 automated unit and regression tests
├── docs/                    # Architecture, quantization, and deployment specifications
└── artifacts/               # Generated benchmark reports and manifests
```

---

## Licence

MIT License. See [LICENSE](LICENSE) for details.
