# Multi-Vector Quantization & Index Compression

## 1. Quantization Techniques

Multi-vector representations produce hundreds of vectors per page, demanding efficient compression to scale to enterprise document corpora. VisDoc AI implements and benchmarks three representations:

### 1.1 FP16 Reference Baseline
- **Storage**: Standard 16-bit half precision (`torch.float16`).
- **Bytes per vector**: 128 dimensions $\times$ 2 bytes = 256 bytes.
- **Reconstruction Error**: Exact baseline (MSE = 0.0, Cosine Similarity = 1.0).

### 1.2 INT8 Uniform Scalar Quantization
- **Storage**: 8-bit signed integer (`torch.int8`).
- **Quantization Formula**:
  $$s = \frac{\max(|x|)}{127}, \quad q = \operatorname{clamp}\left(\left\lfloor \frac{x}{s} \right\rceil, -127, 127\right)$$
- **Bytes per vector**: 128 dimensions $\times$ 1 byte = 128 bytes.
- **Compression Ratio**: 2.0x vs FP16, 4.0x vs FP32.
- **Measured Fidelity**: MSE < 0.0005, Cosine Similarity > 0.998.

### 1.3 1-Bit Binary Quantization (Packed)
- **Storage**: Sign bit packed into `torch.uint8` bytes.
  $$\operatorname{bit}_k = \mathbf{1}(x_k > 0)$$
- **Bytes per vector**: 128 bits / 8 = 16 bytes per vector.
- **Compression Ratio**: 16.0x vs FP16, 32.0x vs FP32.
- **Similarity Scoring**: Bitwise XOR followed by POPCOUNT table lookup:
  $$\operatorname{sim}_{bin}(q, d) = \frac{128 - 2 \cdot \operatorname{popcount}(q \oplus d)}{128}$$

---

## 2. Measured Benchmark Results

### 2.1 Retrieval Quality Retention (Held-Out Test Collection)

| Representation | NDCG@5 | MRR@10 | Recall@5 | Relative NDCG@5 Retention | Embedding Payload | Compression vs FP16 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP16** | 0.5132 | 0.4361 | 0.8333 | **100.0%** (Baseline) | 0.09 MB | 1.0x |
| **INT8** | 0.5132 | 0.4361 | 0.8333 | **100.0%** | 0.05 MB | **2.0x** |
| **BINARY** | 0.3747 | 0.2556 | 0.8333 | **73.0%** | 0.01 MB | **16.0x** |

### 2.2 10,000-Page Scale Benchmark (5,000,000 Multi-Vectors)

| Index Format | Total Indexed Pages | Embedding Memory (MB) | Compression vs FP16 | Mean Latency (ms) | P95 Latency (ms) | Throughput (QPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP16** | 10,000 | 1,220.7 MB | 1.0x | 2,028.90 ms | 2,135.90 ms | 0.5 |
| **INT8** | 10,000 | 610.4 MB | **2.0x** | 3,013.15 ms | 3,083.47 ms | 0.3 |
| **BINARY** | 10,000 | **76.3 MB** | **16.0x** (32x vs FP32) | 2,770.02 ms | 2,890.07 ms | 0.4 |

> [!NOTE]
> Embedding payload compression vs Total index size: The 16x binary compression strictly measures vector payload. Auxiliary metadata (e.g. document titles, SHA256 hashes, page numbers) scales with page count and should be factored into total disk planning.
