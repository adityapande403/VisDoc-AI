import sys
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visdoc.models.colpali_backend import MockVisualRetrieverModel
from visdoc.evaluation.benchmark import RetrievalBenchmarkHarness
from visdoc.utils.logging import logger

console = Console()

def main():
    model = MockVisualRetrieverModel(embedding_dim=128)
    harness = RetrievalBenchmarkHarness(retriever_model=model)

    results = harness.benchmark_10k_synthetic(
        format_names=["fp16", "int8", "binary"],
        num_pages=10000,
        embedding_dim=128,
        vectors_per_page=500, # 5,000,000 total multi-vectors!
    )

    table = Table(title="VisDoc AI - 10,000-Page Scale Benchmark (5,000,000 Multi-Vectors)")
    table.add_column("Index Format", style="cyan")
    table.add_column("Indexed Pages", style="green")
    table.add_column("Embedding Memory (MB)", style="magenta")
    table.add_column("Compression vs FP16", style="bold green")
    table.add_column("Mean Latency", style="yellow")
    table.add_column("P95 Latency", style="bold yellow")
    table.add_column("Throughput (QPS)", style="bold cyan")

    for fmt, r in results.items():
        table.add_row(
            fmt.upper(),
            f"{r['total_pages']:,}",
            f"{r['embedding_memory_mb']:.1f} MB",
            f"{r['compression_ratio_vs_fp16']:.1f}x",
            f"{r['mean_latency_ms']:.2f} ms",
            f"{r['p95_latency_ms']:.2f} ms",
            f"{r['queries_per_second']:.1f}",
        )

    console.print(table)

    report_path = Path("d:/Aditya/hf6/artifacts/reports/scale_10k_benchmark.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved 10,000-page scale report to {report_path}")

if __name__ == "__main__":
    main()
