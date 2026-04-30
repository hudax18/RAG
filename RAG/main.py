import os
import argparse
import time
from datetime import datetime
from dotenv import load_dotenv
from RAG.components.vector_store import VectorStore

# Load environment variables from .env file
load_dotenv()

from RAG.components.reranker import Reranker
from RAG.components.router import QueryRouter
from RAG.components.llm_tiers import LLMTiers
from RAG.components.history import ChatHistory
from RAG.components.semantic_cache import SemanticCache

class HybridRAG:
    def __init__(self, provider: str = None, gemini_api_key: str = None, local_model_path: str = None):
        # Validasi sederhana
        self._validate_env()
        
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        self.router = QueryRouter()
        self.chat_history = ChatHistory(max_length=5)
        self.semantic_cache = SemanticCache()
        
        # Use provided or from environment
        provider = provider or os.getenv("LLM_PROVIDER", "gemini")
        local_model_path = local_model_path or os.getenv("LOCAL_MODEL_PATH")
        
        self.llm_tiers = LLMTiers(
            provider=provider, 
            api_key=gemini_api_key, 
            local_model_path=local_model_path
        )

    def _validate_env(self):
        provider = os.getenv("LLM_PROVIDER")
        if not provider:
            print("⚠️ Warning: LLM_PROVIDER tidak diset di .env. Menggunakan default: gemini.")
            
        if provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            print("❌ Error: GOOGLE_API_KEY tidak ditemukan. Jalankan 'python3 cli.py setup' untuk konfigurasi.")
            
        if provider == "local" and not os.getenv("LOCAL_MODEL_PATH"):
            print("⚠️ Warning: LOCAL_MODEL_PATH tidak ditemukan. Sistem akan mencoba memaksa Cloud route jika gagal.")

    def query(self, user_query: str):
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] 🔍 Memproses Kueri: '{user_query}'")
        
        # Step 0: Semantic Cache Check
        cached_answer = self.semantic_cache.check(user_query)
        if cached_answer:
            print(f"[{now}] ⚡ Cache Hit! Mengembalikan jawaban instan...")
            return cached_answer

        # Ambil konteks percakapan sebelumnya
        history_context = self.chat_history.get_context()
        
        # Step 1: Retrieval
        print(f"[{now}] 📥 Tahap 1: Mencari dokumen relevan...")
        candidates = self.vector_store.similarity_search(user_query, k=20)
        
        if not candidates:
            return "Maaf, saya tidak menemukan informasi yang relevan di database dokumen saya."
        
        # Step 2: Re-ranking
        print(f"[{now}] 🎯 Tahap 2: Menyaring hasil (Re-ranking)...")
        top_docs = self.reranker.rerank(user_query, candidates, top_n=3, threshold=-1.5)
        context = "\n\n".join([doc.page_content for doc in top_docs])
        
        # Step 3: Execution (Exclusive Mode)
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        # Tentukan rute berdasarkan provider yang diset secara eksklusif
        if provider == "local":
            print(f"[{now}] 🏠 Mode: LOCAL ONLY")
            if not self.llm_tiers.local_llm:
                return "❌ Error: Model Lokal tidak dimuat. Jalankan 'python3 cli.py setup' untuk memperbaiki."
            response = self.llm_tiers.call_local(user_query, context, history=history_context)
        else:
            print(f"[{now}] ☁️ Mode: CLOUD ONLY ({provider.upper()})")
            if not self.llm_tiers.cloud_llm:
                return f"❌ Error: Model Cloud ({provider}) tidak tersedia. Cek API Key Anda."
            response = self.llm_tiers.call_cloud(user_query, context, history=history_context)
            
        # Simpan ke cache untuk kueri mendatang
        self.semantic_cache.update(user_query, response)

        # Simpan ke history
        self.chat_history.add_message("user", user_query)
        self.chat_history.add_message("assistant", response)
            
        return response

if __name__ == "__main__":
    # Tetap biarkan main.py bisa dijalankan langsung jika dibutuhkan, 
    # tapi rekomendasikan cli.py
    parser = argparse.ArgumentParser(description="Hybrid RAG System")
    parser.add_argument("--query", type=str, help="Query to ask the RAG")
    parser.add_argument("--local_model", type=str, help="Path to local GGUF model")
    
    args = parser.parse_args()
    
    rag = HybridRAG(local_model_path=args.local_model)
    
    if args.query:
        ans = rag.query(args.query)
        print("\n--- Response ---")
        print(ans)
    else:
        print("Gunakan 'python3 cli.py chat' untuk pengalaman yang lebih baik.")
        # ... fallback ke interactive jika user memaksa
