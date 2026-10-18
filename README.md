# Enterprise Safety & Compliance Copilot (Week 2 RAG Project)

A production-grade RAG application designed to query dense technical safety handbooks and compliance guidelines with precise source citations.

## Architecture
- **Ingestion & Splitting:** PyMuPDF extraction with recursive text splitting ($1000$ chunk size, $200$ overlap).
- **Retrieval Engine:** Hybrid Search combining ChromaDB dense vector embeddings (`text-embedding-3-small`) and BM25 keyword matching, unified via Reciprocal Rank Fusion (RRF).
- **Generation:** GPT-4o-mini with strict source grounding and citation tracking.


![alt text](/images/image.png)

## Quickstart
1. Clone the repository.
2. Install dependencies via `uv`:
   ```bash
   uv sync
