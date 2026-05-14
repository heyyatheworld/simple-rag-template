"""CLI for RAG pipeline: --index, --query, --status, --clear."""
import argparse
from rag_pipeline import RAGPipeline


def main():
    """Parse args and run index, query, status, or clear."""
    parser = argparse.ArgumentParser(description="CLI RAG Service")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", action="store_true", help="Index docs from data dir into the vector DB")
    group.add_argument("--query", type=str, help="Question to answer")
    group.add_argument("--status", action="store_true", help="Read-only diagnostic check of the vector database")
    group.add_argument("--dedupe", action="store_true", help="Remove duplicate chunks (same content_hash) from the vector DB")
    group.add_argument("--clear", action="store_true", help="Clear the vector database")

    args = parser.parse_args()

    rag = RAGPipeline()

    if args.index:
        rag.index()

    elif args.query is not None:
        rag.answer(args.query)

    elif args.status:
        rag.status()

    elif args.dedupe:
        rag.dedupe()

    elif args.clear:
        rag.clear()

    else:
        print("Invalid argument")
        parser.print_help()
        rag.close()

if __name__ == "__main__":
    main()
