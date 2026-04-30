import os
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from RAG.components.embedding import EmbeddingModel
import lancedb

class VectorStore:
    def __init__(self, persist_directory: str = "RAG/lance_db", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embedding_model = EmbeddingModel(model_name=model_name)
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        # Wrap EmbeddingModel
        class LangChainEmbeddingWrapper:
            def __init__(self, model):
                self.model = model
            def embed_documents(self, texts):
                return self.model.embed_documents(texts)
            def embed_query(self, text):
                return self.model.embed_query(text)

        self.db = lancedb.connect(self.persist_directory)
        self.table_name = "documents"
        
        # LanceDB in LangChain requires an existing table or it will create one
        self.vector_store = LanceDB(
            connection=self.db,
            embedding=LangChainEmbeddingWrapper(self.embedding_model),
            table_name=self.table_name
        )

    def add_documents(self, documents: list[Document]):
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 20):
        return self.vector_store.similarity_search(query, k=k)
