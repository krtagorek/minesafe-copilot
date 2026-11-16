"""Compare saved MineSafe retrieval results without modifying the saved index.

Running a question calls OpenAI to embed that question, but never embeds documents
or generates an answer. Chroma operates on a disposable copy of the saved index.
"""

import argparse
import pickle
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data/processed/processed_chunks.pkl"
INDEX_PATH = ROOT / "data/processed/chroma_db"
COLLECTION = "langchain"  # Default used by HybridRetriever's Chroma store.


def print_results(title, documents):
    print(f"\n=== {title} ===")
    if not documents:
        print("(no results)")
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        labels = [
            f"{key}={metadata[key]}"
            for key in ("source_file", "source", "page", "page_label", "total_pages")
            if key in metadata
        ]
        print(f"{rank}. {' | '.join(labels) or '(no source/page metadata)'}")
        text = " ".join(document.page_content.split())
        print(f"   {text[:240]}{'...' if len(text) > 240 else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to compare across retrieval methods")
    parser.add_argument(
        "--index-directory", type=Path, default=INDEX_PATH,
        help="Saved Chroma directory to compare (queried through a temporary copy)",
    )
    args = parser.parse_args()
    index_path = args.index_directory.resolve()
    if not args.question.strip():
        parser.error("question must not be blank")

    # Check saved artifacts before importing retrieval code or creating clients.
    missing = [
        path for path in (CHUNKS_PATH, index_path / "chroma.sqlite3")
        if not path.is_file()
    ]
    if missing:
        parser.exit(1, "Missing saved retrieval data: " + ", ".join(map(str, missing))
                    + "\nStopped; no ingestion or index rebuild was attempted.\n")

    sys.path.insert(0, str(ROOT / "src"))
    from dotenv import load_dotenv
    from chromadb import PersistentClient
    from chromadb.config import Settings
    from langchain_community.retrievers import BM25Retriever
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from retriever import HybridRetriever

    load_dotenv(ROOT / ".env")
    # This is the same trusted local pickle that get_or_create_chunks loads.
    with CHUNKS_PATH.open("rb") as handle:
        chunks = pickle.load(handle)
    if not chunks:
        parser.exit(1, "Saved chunks are empty. Stopped without rebuilding.\n")

    with tempfile.TemporaryDirectory(prefix="minesafe-retrieval-") as scratch:
        snapshot = Path(scratch) / "chroma_db"
        shutil.copytree(index_path, snapshot)
        # Include committed WAL data using SQLite's backup API; source is read-only.
        with sqlite3.connect(
            (index_path / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True
        ) as source, sqlite3.connect(snapshot / "chroma.sqlite3") as target:
            source.backup(target)

        client = PersistentClient(
            path=str(snapshot), settings=Settings(anonymized_telemetry=False)
        )
        # get_collection fails if missing; do not silently create an empty index.
        try:
            collection = client.get_collection(COLLECTION, embedding_function=None)
        except Exception as error:
            parser.exit(1, f"Cannot open saved collection {COLLECTION!r}: {error}\n"
                        "Stopped without rebuilding.\n")
        if collection.count() == 0:
            parser.exit(1, "Saved collection is empty. Stopped without rebuilding.\n")

        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 5
        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            create_collection_if_not_exists=False,
        )
        print(f"Question: {args.question}")
        print(f"Saved chunks: {len(chunks)} | Indexed records: {collection.count()}")
        print("Page metadata is printed as saved (page is normally zero-based).")
        dense_docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(args.question)
        bm25_docs = bm25.invoke(args.question)
        # Bypass __init__: only the stateless RRF method is needed here.
        # The existing RRF method uses no instance state; keep its logic unchanged.
        fusion = HybridRetriever.__new__(HybridRetriever)
        fused_docs = fusion.reciprocal_rank_fusion(dense_docs, bm25_docs, k=5)
        print_results("Vector-only (top 5)", dense_docs)
        print_results("BM25-only (top 5)", bm25_docs)
        print_results("RRF (top 5, c=60)", fused_docs)


if __name__ == "__main__":
    main()
