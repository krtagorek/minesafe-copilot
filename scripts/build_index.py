"""Explicitly create/upsert the index. This command makes embedding API calls."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persist-directory", default=str(ROOT / "data/processed/chroma_db"))
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from ingest import get_or_create_chunks
    from retriever import build_index
    chunks = get_or_create_chunks(
        raw_dir=str(ROOT / "data/raw"),
        cache_path=str(ROOT / "data/processed/processed_chunks.pkl"),
    )
    store = build_index(chunks, args.persist_directory)
    print(f"Indexed collection contains {store._collection.count()} records.")


if __name__ == "__main__":
    main()
