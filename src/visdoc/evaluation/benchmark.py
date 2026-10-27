from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import numpy as np
import torch

from visdoc.dataset.schema import RetrievalExample
from visdoc.embeddings.generator import PageEmbedding
from visdoc.indexing.page_index import PageIndex
from visdoc.models.base import BaseVisualRetriever
from visdoc.evaluation.metrics import evaluate_retrieval_run, compute_quantization_error
from visdoc.utils.logging import logger

@dataclass
class BenchmarkReport:
    """Benchmark performance and quality results."""
    format_name: str
    total_pages: int
    metrics: Dict[str, float]
    ndcg_retention_vs_fp16: Optional[float]
    latency_stats: Dict[str, float] # mean_ms, median_ms, p95_ms, p99_ms
    memory_stats: Dict[str, Any]
    throughput_qps: float
    fidelity: Optional[Dict[str, float]] = None

class RetrievalBenchmarkHarness:
    """
    Standardized benchmark harness comparing FP16, INT8, and Binary indexes
    across retrieval metrics, query latency percentiles, and memory consumption.
    """

    def __init__(
        self,
        retriever_model: BaseVisualRetriever,
        warmup_queries: int = 5,
    ):
        self.model = retriever_model
        self.warmup_queries = warmup_queries

    def evaluate_index(
        self,
        index: PageIndex,
        eval_examples: List[RetrievalExample],
        fp16_baseline_ndcg: Optional[float] = None,
        top_k: int = 10,
    ) -> BenchmarkReport:
        """Runs evaluation over the given index using held-out queries."""
        ground_truth: Dict[str, set] = {
            ex.query_id: set(ex.positive_page_ids) for ex in eval_examples
        }

        # Warm-up phase
        if self.warmup_queries > 0 and eval_examples:
            warm_queries = [ex.query for ex in eval_examples[:self.warmup_queries]]
            q_embs, q_masks = self.model.encode_queries(warm_queries)
            for i in range(q_embs.size(0)):
                _ = index.search(q_embs[i], q_masks[i], top_k=top_k)

        # Timed evaluation run
        latencies_ms: List[float] = []
        retrieved_results: Dict[str, List[str]] = {}

        total_start = time.perf_counter()

        for ex in eval_examples:
            # Measure query encoding + search latency
            t0 = time.perf_counter()
            q_emb, q_mask = self.model.encode_queries([ex.query])
            res = index.search(q_emb[0], q_mask[0], top_k=top_k)
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0) # milliseconds
            retrieved_results[ex.query_id] = [r["page_id"] for r in res]

        total_elapsed = time.perf_counter() - total_start
        throughput_qps = len(eval_examples) / max(1e-5, total_elapsed)

        # Compute IR metrics
        ir_metrics = evaluate_retrieval_run(retrieved_results, ground_truth)

        # Calculate Relative NDCG@5 Retention
        ndcg5 = ir_metrics["ndcg@5"]
        if fp16_baseline_ndcg is not None and fp16_baseline_ndcg > 0:
            retention = ndcg5 / fp16_baseline_ndcg
        else:
            retention = 1.0 # Self baseline

        # Latency statistics
        latency_stats = {
            "mean_ms": float(np.mean(latencies_ms)),
            "median_ms": float(np.median(latencies_ms)),
            "p90_ms": float(np.percentile(latencies_ms, 90)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
        }

        # Memory statistics
        mem_stats = index.get_memory_stats()

        logger.info(
            f"Evaluated '{index.format_name.upper()}': NDCG@5={ndcg5:.4f} "
            f"(Retention: {retention*100:.1f}%), Latency P95={latency_stats['p95_ms']:.2f}ms, "
            f"Embeddings={mem_stats['quantized_embedding_mb']:.2f} MB"
        )

        return BenchmarkReport(
            format_name=index.format_name,
            total_pages=len(index),
            metrics=ir_metrics,
            ndcg_retention_vs_fp16=retention,
            latency_stats=latency_stats,
            memory_stats=mem_stats,
            throughput_qps=throughput_qps,
        )

    def benchmark_10k_synthetic(
        self,
        format_names: List[str] = ["fp16", "int8", "binary"],
        num_pages: int = 10000,
        embedding_dim: int = 128,
        vectors_per_page: int = 500,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs high-scale memory and latency benchmark up to 10,000 pages.
        Measures exact search latency in milliseconds per query and memory requirements.
        """
        logger.info(f"--- Starting 10,000-Page Scale Benchmark ({num_pages} pages, {vectors_per_page} vectors/page) ---")
        results = {}

        # Generate representative page embeddings
        torch.manual_seed(42)
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        for fmt in format_names:
            fmt_start = time.time()
            idx = PageIndex(format_name=fmt, embedding_dim=embedding_dim, use_gpu=(dev == "cuda"))

            # Build dummy page embeddings in chunks to measure indexing time and memory
            chunk_size = 1000
            total_chunks = num_pages // chunk_size

            for c in range(total_chunks):
                chunk_embs = []
                for p in range(chunk_size):
                    pid = f"scale_{c:03d}_p{p:04d}"
                    dummy_tensor = torch.randn(vectors_per_page, embedding_dim)
                    pe = PageEmbedding(
                        page_id=pid,
                        document_id=f"doc_{c:03d}",
                        page_number=p + 1,
                        num_vectors=vectors_per_page,
                        embedding_dim=embedding_dim,
                        embeddings=dummy_tensor,
                    )
                    chunk_embs.append(pe)
                idx.add_pages(chunk_embs)

            build_time = time.time() - fmt_start
            mem = idx.get_memory_stats()

            # Measure search latency over 20 representative queries
            q_latencies = []
            for _ in range(20):
                dummy_q = torch.randn(16, embedding_dim, device=idx.device)
                q_mask = torch.ones(16, dtype=torch.bool, device=idx.device)

                t0 = time.perf_counter()
                _ = idx.search(dummy_q, q_mask, top_k=10)
                t1 = time.perf_counter()
                q_latencies.append((t1 - t0) * 1000.0)

            res_entry = {
                "format": fmt,
                "total_pages": len(idx),
                "total_vectors": mem["total_vectors"],
                "index_build_seconds": build_time,
                "embedding_memory_mb": mem["quantized_embedding_mb"],
                "compression_ratio_vs_fp16": mem["compression_ratio_vs_fp16"],
                "mean_latency_ms": float(np.mean(q_latencies)),
                "median_latency_ms": float(np.median(q_latencies)),
                "p95_latency_ms": float(np.percentile(q_latencies, 95)),
                "queries_per_second": 1000.0 / float(np.mean(q_latencies)),
            }
            results[fmt] = res_entry

            logger.info(
                f"[10,000 PAGES] {fmt.upper()}: Memory={res_entry['embedding_memory_mb']:.1f} MB "
                f"({res_entry['compression_ratio_vs_fp16']:.1f}x compression), "
                f"P95 Latency={res_entry['p95_latency_ms']:.2f} ms"
            )

        return results
