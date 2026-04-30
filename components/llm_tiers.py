import os
import re
from RAG.components.llm_factory import LLMFactory
from llama_cpp import Llama

class LLMTiers:
    def __init__(self, provider: str = "gemini", model_name: str = None, api_key: str = None, local_model_path: str = None):
        self.cloud_llm = None
        self.local_llm = None
        self.local_model_path = local_model_path

        # Tentukan pilihan secara eksklusif
        is_local_mode = provider.lower() == "local"

        if is_local_mode:
            # Hanya inisialisasi Lokal
            if self.local_model_path and os.path.exists(self.local_model_path):
                try:
                    self.local_llm = Llama(model_path=self.local_model_path, n_ctx=2048, verbose=False)
                except Exception as e:
                    print(f"⚠️ Warning: Gagal memuat model lokal: {e}")
        else:
            # Hanya inisialisasi Cloud
            try:
                self.cloud_llm = LLMFactory.get_llm(provider, model_name, api_key)
            except Exception as e:
                print(f"⚠️ Warning: Inisialisasi Cloud ({provider}) gagal: {e}")

    def _clean_output(self, text: str) -> str:
        """Menghapus tag reasoning seperti <think>...</think> dari output."""
        # Menghapus apa pun di antara <think> dan </think> termasuk tag-nya
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    def call_cloud(self, prompt: str, context: str, history: str = "") -> str:
        if not self.cloud_llm:
            return "Cloud model not available."

        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{now}] Calling Cloud LLM API...")

        system_prompt = (
            "Anda adalah asisten cerdas penganalisis laporan tahunan. "
            "Gunakan konteks dokumen dan percakapan sebelumnya untuk menjawab.\n\n"
            "ATURAN KONVERSI MATA UANG:\n"
            "1. Jika di dokumen tertulis 'dalam jutaan rupiah', konversikan ke angka yang lebih mudah dibaca manusia.\n"
            "   Contoh: 20.491.015 (dalam jutaan) -> Rp 20,49 Triliun atau Rp 20.491 Miliar.\n"
            "2. Selalu sertakan satuan yang jelas (Triliun/Miliar/Juta).\n\n"
            "ATURAN FORMAT:\n"
            "1. Jika jawaban berupa nilai tunggal/fakta singkat, jawablah LANGSUNG TO-THE-POINT.\n"
            "2. Jika jawaban melibatkan daftar atau perbandingan, GUNAKAN BULLET POINTS.\n"
            "3. JAWABLAH SELALU DALAM BAHASA INDONESIA.\n"
            "4. Jangan berikan alasan teknis kecuali diminta."
        )

        full_prompt = f"{history}\n\nKonteks Dokumen:\n{context}\n\nPertanyaan Baru: {prompt}"

        messages = [
            ("system", system_prompt),
            ("human", full_prompt),
        ]

        response = self.cloud_llm.invoke(messages)
        return self._clean_output(response.content)

    def call_local(self, prompt: str, context: str, history: str = "") -> str:
        if not self.local_llm:
            return "Local model not available. Fallback to Cloud if possible."

        full_prompt = (
            "### System:\n"
            "Anda adalah asisten analisis data. JAWABLAH DALAM BAHASA INDONESIA.\n"
            "ATURAN: Jawab singkat untuk fakta tunggal. Gunakan bullet points untuk daftar.\n"
            "PENTING: Konversi angka 'jutaan' ke format yang mudah dibaca seperti 'Triliun' atau 'Miliar'.\n"
            "Contoh: 20.000.000 (jutaan) = 20 Triliun.\n\n"
            f"### Chat History:\n{history}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Question:\n{prompt}\n\n"
            "### Answer:\n"
        )

        output = self.local_llm(
            full_prompt, 
            max_tokens=512, 
            stop=["###", "Question:", "Q:"], 
            echo=False,
            temperature=0.1,
            repeat_penalty=1.2
        )
        return self._clean_output(output['choices'][0]['text'])

