from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List


class HybridRetriever:
    def __init__(self, chunks: List[Document], persist_directory: str = "data/processed/chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize Chroma Vector Store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Initialize BM25 Retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 5

    def reciprocal_rank_fusion(self, dense_docs: List[Document], bm25_docs: List[Document], k: int = 6, c: int = 60) -> List[Document]:
        fusion_scores = {}
        doc_map = {}

        def add_scores(docs):
            for rank, doc in enumerate(docs):
                doc_id = doc.page_content[:100] # Unique content fingerprint
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
