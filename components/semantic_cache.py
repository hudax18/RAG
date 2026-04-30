import os
import lancedb
import pandas as pd
from RAG.components.embedding import EmbeddingModel

class SemanticCache:
    def __init__(self, persist_directory: str = "RAG/lance_db", threshold: float = 0.95):
        self.persist_directory = persist_directory
        self.table_name = "semantic_cache"
        self.threshold = threshold
        self.embedding_model = EmbeddingModel()
        
        # Connect to LanceDB
        self.db = lancedb.connect(self.persist_directory)
        
        # Open or create table
        if self.table_name not in self.db.table_names():
            # Initial dummy data to define schema
            dummy_data = {
                "vector": [self.embedding_model.embed_query("dummy")],
                "query": ["dummy"],
                "answer": ["dummy"]
            }
            self.table = self.db.create_table(self.table_name, data=pd.DataFrame(dummy_data))
        else:
            self.table = self.db.open_table(self.table_name)

    def check(self, query: str):
        """Mengecek apakah kueri serupa ada di cache."""
        query_vector = self.embedding_model.embed_query(query)
        
        # Cari kueri terdekat
        results = self.table.search(query_vector).limit(1).to_pandas()
        
        if not results.empty:
            # Hitung similarity sederhana (LanceDB returns distance, usually L2)
            # Untuk cosine similarity, kita bisa asumsikan jika vector ternormalisasi.
            # BGE models biasanya memberikan normalized vectors.
            # Distance L2 untuk normalized vectors: distance = 2 * (1 - similarity)
            # Maka similarity = 1 - (distance / 2)
            
            distance = results.iloc[0]['_distance']
            similarity = 1 - (distance / 2)
            
            if similarity >= self.threshold:
                return results.iloc[0]['answer']
        
        return None

    def update(self, query: str, answer: str):
        """Menambahkan kueri dan jawaban baru ke cache."""
        query_vector = self.embedding_model.embed_query(query)
        new_data = {
            "vector": [query_vector],
            "query": [query],
            "answer": [answer]
        }
        self.table.add(pd.DataFrame(new_data))
