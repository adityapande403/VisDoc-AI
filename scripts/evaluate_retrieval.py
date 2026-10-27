import sys
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visdoc.dataset.schema import RetrievalExample
from visdoc.retrieval.retriever import VisualDocumentRetriever
from visdoc.evaluation.benchmark import RetrievalBenchmarkHarness
from visdoc.utils.logging import logger

console = Console()

def main():
    test_file = Path("d:/Aditya/hf6/data/datasets/retrieval_test.jsonl")
    if not test_file.exists():
        logger.error(f"Test dataset {test_file} not found. Run generate_synthetic_docs.py first.")
        return

    eval_examples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_examples.append(RetrievalExample(**json.loads(line)))

    logger.info(f"Loaded {len(eval_examples)} evaluation queries.")

    doc_dir = Path("d:/Aditya/hf6/data/documents")
    pdf_files = list(doc_dir.glob("*.pdf"))

    formats = ["fp16", "int8", "binary"]
    reports = {}

    fp16_ndcg = None

    table = Table(title="VisDoc AI - Multi-Vector Index Evaluation & Compression Benchmark")
    table.add_column("Representation", style="cyan")
    table.add_column("NDCG@5", style="green")
    table.add_column("MRR@10", style="green")
    table.add_column("Recall@5", style="green")
    table.add_column("Relative Retention", style="magenta")
    table.add_column("P95 Latency", style="yellow")
    table.add_column("Embedding Memory", style="blue")
    table.add_column("Compression", style="bold green")

    for fmt in formats:
        retriever = VisualDocumentRetriever(index_format=fmt, use_mock_model=True)
        retriever.index_documents(pdf_files)

        harness = RetrievalBenchmarkHarness(retriever_model=retriever.model, warmup_queries=2)
        report = harness.evaluate_index(
            index=retriever.index,
            eval_examples=eval_examples,
            fp16_baseline_ndcg=fp16_ndcg,
        )

        if fmt == "fp16":
            fp16_ndcg = report.metrics["ndcg@5"]

        reports[fmt] = report

        retention_str = f"{report.ndcg_retention_vs_fp16 * 100:.1f}%" if report.ndcg_retention_vs_fp16 else "100.0%"
        comp_ratio = report.memory_stats.get("compression_ratio_vs_fp16", 1.0)

        table.add_row(
            fmt.upper(),
            f"{report.metrics['ndcg@5']:.4f}",
            f"{report.metrics['mrr@10']:.4f}",
            f"{report.metrics['recall@5']:.4f}",
            retention_str,
            f"{report.latency_stats['p95_ms']:.2f} ms",
            f"{report.memory_stats['quantized_embedding_mb']:.2f} MB",
            f"{comp_ratio:.1f}x",
        )

    console.print(table)

    # Save benchmark report to disk
    report_file = Path("d:/Aditya/hf6/artifacts/reports/retrieval_benchmark.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json_data = {
            fmt: {
                "format": r.format_name,
                "metrics": r.metrics,
                "retention": r.ndcg_retention_vs_fp16,
                "latency_stats": r.latency_stats,
                "memory_stats": r.memory_stats,
            }
            for fmt, r in reports.items()
        }
        json.dump(json_data, f, indent=2)

    logger.info(f"Saved evaluation benchmark report to {report_file}")

if __name__ == "__main__":
    main()
