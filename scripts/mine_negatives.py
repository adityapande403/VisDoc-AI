import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visdoc.dataset.schema import RetrievalExample
from visdoc.dataset.negative_miner import HardNegativeMiner
from visdoc.retrieval.retriever import VisualDocumentRetriever
from visdoc.utils.logging import logger

def main():
    dataset_file = Path("d:/Aditya/hf6/data/datasets/retrieval_train.jsonl")
    if not dataset_file.exists():
        logger.error(f"Dataset file {dataset_file} not found. Run generate_synthetic_docs.py first.")
        return

    # Load examples
    examples = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(RetrievalExample(**json.loads(line)))

    # Initialize retriever
    retriever = VisualDocumentRetriever(index_format="fp16", use_mock_model=True)
    doc_dir = Path("d:/Aditya/hf6/data/documents")
    pdf_files = list(doc_dir.glob("*.pdf"))
    retriever.index_documents(pdf_files)

    miner = HardNegativeMiner(
        retriever_model=retriever.model,
        page_index=retriever.index,
        max_negatives_per_query=3,
    )

    enriched = miner.enrich_dataset_with_hard_negatives(examples)

    # Save enriched dataset
    output_file = Path("d:/Aditya/hf6/data/datasets/retrieval_train_mined.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in enriched:
            f.write(ex.model_dump_json() + "\n")

    logger.info(f"Enriched {len(enriched)} examples saved to {output_file}")

if __name__ == "__main__":
    main()
