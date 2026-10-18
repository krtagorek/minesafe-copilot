from dotenv import load_dotenv
load_dotenv()

from ingest import get_or_create_chunks
from retriever import HybridRetriever

def run_evaluation():
    print("Initializing evaluation harness...")
    chunks = get_or_create_chunks()
    retriever = HybridRetriever(chunks)
    
    test_cases = [
        ("What are the core protocols for checking electrical voltage drops and hazards?", "PH26-V-1"),
        ("What are the standard guidelines for writing objective safety citations?", "PH25-I-2"),
        ("What are the employer's responsibilities for maintaining general workplace safety?", "OSHA 2209")
    ]
    
    passed = 0
    for query, expected_source in test_cases:
        docs = retriever.search(query)
        sources = [doc.metadata.get("source_file", "") for doc in docs]
        match = any(expected_source in src for src in sources)
        status = "✅ PASS" if match else "❌ FAIL"
        if match:
            passed += 1
        print(f"Query: '{query}' | Expected: {expected_source} | Result: {status}")
        
    print(f"\nEvaluation Complete: {passed}/{len(test_cases)} tests passed.")

if __name__ == "__main__":
    run_evaluation()
