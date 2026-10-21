import os
import pickle
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_clean_pdfs(raw_dir: str = "data/raw") -> list[Document]:
    docs = []
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return docs
    
    for file_path in raw_path.glob("**/*.pdf"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            try:
                loader = PyMuPDFLoader(str(file_path))
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source_file"] = file_path.name
                    doc.page_content = f": " + doc.page_content.replace("\n", " ").strip()
                    docs.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return docs

def get_chunked_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def get_or_create_chunks(raw_dir: str = "data/raw", cache_path: str = "data/processed/processed_chunks.pkl") -> list[Document]:
    if os.path.exists(cache_path):
        print("Loading pre-chunked documents from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print("Processing raw PDF documents and chunking...")
    raw_docs = load_and_clean_pdfs(raw_dir)
    chunks = get_chunked_documents(raw_docs)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
        
    return chunks
