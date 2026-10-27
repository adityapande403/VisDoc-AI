# Limitations and Trade-Offs

## 1. Visual Retrieval vs Structured Extraction
- **Scope**: VisDoc AI is an information retrieval engine designed to find and rank relevant document pages based on visual-spatial features. It retrieves the exact relevant page image and metadata.
- **Downstream RAG**: Downstream multimodal LLMs (e.g., Qwen2.5-VL, GPT-4o, Claude 3.5 Sonnet) should be used to parse exact tabular figures or answer conversational questions conditioned on the retrieved page image.

## 2. Binary Quantization Trade-off
- 1-Bit Binary quantization delivers dramatic storage savings (16.0x over FP16, 32.0x over FP32) and compact memory consumption (76.3 MB for 10,000 pages).
- However, as verified empirically on our held-out test dataset, sign-only binary thresholding without quantization-aware training retains ~73% of baseline NDCG@5.
- For maximum precision (>99.9% cosine correlation and 100% NDCG@5 retention), **INT8 uniform scalar quantization** is strongly recommended as it halves memory footprint with zero loss in ranking accuracy.

## 3. High-Scale Linear Scanning
- While multi-vector late interaction is computationally intensive for full linear scans ($O(N_{pages} \cdot N_q \cdot N_d)$), our two-stage candidate filter (mean-pooled summary pruning) reduces search latency from 253 ms down to 10.7 ms (a 23.7x acceleration) on a 1,000-page collection.
