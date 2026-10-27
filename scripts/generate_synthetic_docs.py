import sys
from pathlib import Path
import json

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visdoc.dataset.generator import SyntheticDocumentGenerator
from visdoc.utils.logging import logger

def main():
    data_dir = Path("d:/Aditya/hf6/data")
    doc_dir = data_dir / "documents"
    ds_dir = data_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    gen = SyntheticDocumentGenerator(output_dir=doc_dir)
    pdf_paths, queries = gen.generate_all()

    # Split into train and test
    train_queries = [q for q in queries if q.split == "train"]
    test_queries = [q for q in queries if q.split == "test"]

    train_file = ds_dir / "retrieval_train.jsonl"
    test_file = ds_dir / "retrieval_test.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for q in train_queries:
            f.write(q.model_dump_json() + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for q in test_queries:
            f.write(q.model_dump_json() + "\n")

    logger.info(f"Generated {len(pdf_paths)} documents.")
    logger.info(f"Saved {len(train_queries)} train queries to {train_file}")
    logger.info(f"Saved {len(test_queries)} test queries to {test_file}")

if __name__ == "__main__":
    main()
