from fastembed import TextEmbedding
from typing import List

class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return list(self.model.embed(texts))

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0]
