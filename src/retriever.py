from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List


BUILD_INSTRUCTION = "Build explicitly with: uv run python scripts/build_index.py"


def document_id(doc: Document) -> str:
    """Stable identity shared by indexing and fusion; metadata is not truncated."""
    identity = [doc.page_content, {
        key: doc.metadata.get(key)
        for key in ("source", "source_file", "page", "page_label")
    }]
    return hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")).hexdigest()


def build_index(chunks: List[Document], persist_directory: str = "data/processed/chroma_db"):
    """Explicit upsert of unique documents using deterministic IDs."""
    documents = {document_id(doc): doc for doc in chunks}
    if not documents:
        raise ValueError("Cannot build an index from empty chunks.")
    return Chroma.from_documents(
        documents=list(documents.values()), ids=list(documents),
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=persist_directory, collection_name="langchain",
    )


class HybridRetriever:
    def __init__(self, chunks: List[Document], persist_directory: str = "data/processed/chroma_db"):
        if not (Path(persist_directory) / "chroma.sqlite3").is_file():
            raise FileNotFoundError(f"Saved Chroma index is missing. {BUILD_INSTRUCTION}")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize Chroma Vector Store
        try:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
                collection_name="langchain", create_collection_if_not_exists=False,
            )
        except Exception as error:
            raise RuntimeError(f"Cannot open saved Chroma collection. {BUILD_INSTRUCTION}") from error
        if self.vectorstore._collection.count() == 0:
            raise ValueError(f"Saved Chroma collection is empty. {BUILD_INSTRUCTION}")
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Initialize BM25 Retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 5

    def reciprocal_rank_fusion(self, dense_docs: List[Document], bm25_docs: List[Document], k: int = 6, c: int = 60) -> List[Document]:
        fusion_scores = {}
        doc_map = {}

        def add_scores(docs):
            seen = set()
            for rank, doc in enumerate(docs):
                doc_id = document_id(doc)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                doc_map[doc_id] = doc
                score = 1.0 / (c + rank + 1)
                fusion_scores[doc_id] = fusion_scores.get(doc_id, 0.0) + score

        add_scores(dense_docs)
        add_scores(bm25_docs)

        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:k]]

    def search(self, query: str) -> List[Document]:
        try:
            dense_docs = self.dense_retriever.invoke(query)
            bm25_docs = self.bm25_retriever.invoke(query)
            return self.reciprocal_rank_fusion(dense_docs, bm25_docs, k=5)
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
