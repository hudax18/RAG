import os
import argparse
from RAG.components.vector_store import VectorStore
from RAG.components.reranker import Reranker

def diagnostic_test(query: str):
    print(f"\n🔍 DIAGNOSTIC TEST FOR QUERY: '{query}'")
    print("-" * 50)
    
    vs = VectorStore()
    reranker = Reranker()
    
    # Step 1: Raw Retrieval
    print("Step 1: Raw Retrieval (Top 10 candidates)...")
    candidates = vs.similarity_search(query, k=10)
    for i, doc in enumerate(candidates):
        print(f"[{i+1}] Snippet: {doc.page_content[:100]}...")

    # Step 2: Re-ranking
    print("\nStep 2: Re-ranking (Cross-Encoder Scoring)...")
    # We use a lower threshold here just to see what the scores look like
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.model.predict(pairs)
    
    scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    
    print("\n--- Top 3 Final Results (Sent to LLM) ---")
    for i, (score, doc) in enumerate(scored_docs[:3]):
        status = "✅ RELEVANT" if score > -1.5 else "⚠️ LOW RELEVANCE"
        print(f"\nRank {i+1} | Score: {score:.4f} | {status}")
        print(f"Content: {doc.page_content[:300]}...")
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    
    diagnostic_test(args.query)
