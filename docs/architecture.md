# VisDoc AI Architecture Guide

VisDoc AI is a production-oriented multimodal visual document retrieval engine built on Vision-Language Models (VLMs) and ColBERT-style late interaction.

## 1. System Overview

Conventional RAG pipelines rely on Optical Character Recognition (OCR), parsing text chunks into single dense vector representations. This approach loses crucial spatial, geometric, and visual context in complex technical and business documents, such as:
- Pinout tables and circuit schematics in engineering datasheets.
- Multi-column financial statements, balance sheets, and tabular invoices.
- Stamped approval certificates, handwritten notes, and official signatures.
- Multi-column scientific papers and figures.

VisDoc AI replaces OCR-based chunking with direct multi-vector late-interaction representations:
1. Every document page is rendered at high fidelity (DPI 150/200) into an image.
2. The vision-language model (`ColQwen2.5` / `ColPali`) encodes the page into a matrix of visual patch embeddings: $D \in \mathbb{R}^{N_d \times 128}$.
3. Text queries are tokenized and projected into query embeddings: $Q \in \mathbb{R}^{N_q \times 128}$.
4. Retrieval is evaluated using the MaxSim late-interaction operator:
   $$S(Q, D) = \sum_{i=1}^{N_q} M_{q, i} \max_{j: M_{d, j}=1} (q_i \cdot d_j)$$

## 2. Ingestion and Rendering Engine

- **Engine**: PyMuPDF (`pymupdf`)
- **Resolution**: Configurable DPI (default 150), scaling to max dimension (default 1536 px) while preserving native aspect ratios.
- **Identity & Checksums**: SHA256 checksums computed for both source documents and page images to guarantee idempotent ingestion and deduplication.
- **Supported Formats**: PDF, PNG, JPEG, TIFF.

## 3. Embedding & Quantization Architecture

Embeddings are unit L2-normalized 128-dimensional vectors. Three storage representations are supported:
1. **FP16 Baseline**: IEEE 754 half-precision (256 bytes per token).
2. **INT8 Uniform Scalar Quantization**: Per-vector symmetric scaling ($s = \max(|x|) / 127$), reducing payload by 2.0x (128 bytes per token).
3. **1-Bit Binary Packed Quantization**: Sign-bit thresholding ($x > 0$) packed into 16 `uint8` bytes per token (16x reduction over FP16, 32x over FP32), scored using bitwise POPCOUNT.

## 4. Two-Stage Candidate Retrieval

For large collections (e.g. $\ge 1,000$ pages), exact full-index MaxSim scanning can be accelerated using mean-pooled page summary vectors:
- Stage 1: Fast dot product between mean query vector $\bar{q}$ and page summary vectors $\bar{d}$ to filter top candidates (e.g. top 5-10%).
- Stage 2: Exact late-interaction MaxSim scoring on the filtered candidates.
- **Measured speedup**: 23.7x latency reduction (from ~253 ms down to ~10.7 ms on a 1,000-page collection).
