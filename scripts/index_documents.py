import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visdoc.retrieval.retriever import VisualDocumentRetriever
from visdoc.utils.logging import logger

def main():
    parser = argparse.ArgumentParser(description="Index documents using VisDoc AI")
    parser.add_argument("--doc-dir", type=str, default="d:/Aditya/hf6/data/documents")
    parser.add_argument("--format", type=str, default="fp16", choices=["fp16", "int8", "binary"])
    parser.add_argument("--use-mock", action="store_true", help="Use deterministic mock model for testing")
    parser.add_argument("--save-name", type=str, default=None)
    args = parser.parse_args()

    doc_dir = Path(args.doc_dir)
    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {doc_dir}")

    files = list(doc_dir.glob("*.pdf")) + list(doc_dir.glob("*.png")) + list(doc_dir.glob("*.jpg"))
    if not files:
        logger.warning(f"No documents found in {doc_dir}")
        return

    logger.info(f"Indexing {len(files)} files into {args.format.upper()} index...")

    retriever = VisualDocumentRetriever(
        index_format=args.format,
        use_mock_model=args.use_mock,
    )

    added = retriever.index_documents(files)
    save_path = retriever.save_index(name=args.save_name or f"index_{args.format}")

    logger.info(f"Successfully indexed {added} pages into {save_path}")

if __name__ == "__main__":
    main()
