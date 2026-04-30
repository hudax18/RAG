from sentence_transformers import CrossEncoder
from typing import List
from langchain_core.documents import Document

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_n: int = 5, threshold: float = -2.0) -> List[Document]:
        if not documents:
            return []
        
        # Prepare pairs: (query, doc_text)
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort documents by score
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
        # Filter by threshold (keep only relevant docs)
        relevant_docs = [doc for score, doc in scored_docs if score > threshold]
        
        # Return top N from relevant docs
        return relevant_docs[:top_n]
