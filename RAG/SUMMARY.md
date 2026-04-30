# Technical Summary: Hybrid RAG System

Dokumen ini merinci arsitektur, komponen, dan batasan operasional dari sistem **Retrieval-Augmented Generation (RAG)** yang telah diimplementasikan.

---

## 1. Arsitektur Sistem (Pipeline)

Sistem ini menggunakan pendekatan **Retrieve & Re-rank** yang dikombinasikan dengan **Dual-Tier Generation** untuk efisiensi biaya dan akurasi.

### A. Tahap Ingesti (Ingestion Pipeline)
1.  **Universal Loading:** Menggunakan `LangChain` loaders untuk mendukung format `.pdf`, `.txt`, `.md`, `.csv`, dan `.docx`.
2.  **Text Splitting:** Dokumen dipecah menjadi bagian kecil (chunks) sebesar 1000 karakter dengan overlap 100 karakter menggunakan `RecursiveCharacterTextSplitter`.
3.  **Embedding:** Menggunakan model ultra-ringan `BAAI/bge-small-en-v1.5` melalui library `FastEmbed`. Proses ini berjalan sangat cepat di CPU.
4.  **Vector Store:** Menggunakan `LanceDB`, database vektor berbasis kolom yang efisien dan serverless, untuk menyimpan representasi vektor dokumen.

### B. Tahap Inferensi (Retrieval & Generation Pipeline)
1.  **Retrieval (Top-20):** Saat user bertanya, sistem mengambil 20 kandidat dokumen paling relevan menggunakan similarity search.
2.  **Re-Ranking (Top-5):** Ke-20 kandidat diproses oleh model **Cross-Encoder** (`BAAI/bge-reranker-base`). Model ini memberikan skor relevansi yang lebih akurat daripada embedding standar.
3.  **Intelligent Routing:** 
    *   Sistem menganalisis kompleksitas kueri (panjang teks dan kata kunci seperti "jelaskan", "bandingkan").
    *   **Jalur Hijau (LOCAL):** Untuk kueri sederhana (jika model GGUF tersedia).
    *   **Jalur Merah (CLOUD):** Untuk kueri kompleks menggunakan `Gemini 2.5 flash`.
    *   **Fallback:** Jika model lokal tidak ditemukan, sistem otomatis beralih ke Cloud.
4.  **Generation:** Jawaban dihasilkan berdasarkan konteks dari 5 dokumen terbaik hasil reranking.

---

## 2. Detail Komponen Teknologi

| Komponen | Teknologi | Alasan Pemilihan |
| :--- | :--- | :--- |
| **Orchestrator** | LangChain | Ekosistem luas dan mudah dikembangkan. |
| **Embedding** | FastEmbed (BGE-Small) | Sangat cepat di CPU, memori rendah (<100MB). |
| **Vector DB** | LanceDB | Menghindari dependensi SQLite eksternal (masalah pada Chroma). |
| **Reranker** | Sentence-Transformers | Meningkatkan akurasi secara signifikan dibanding retrieval biasa. |
| **LLM Cloud** | Google Gemini 2.5 flash | Context window besar dan performa penalaran tinggi. |
| **LLM Local** | Llama-cpp-python | Mendukung format GGUF untuk eksekusi di hardware standar. |

---

## 3. Keterbatasan Sistem (Current Status)

Sistem ini telah berkembang dari prototipe awal. Berikut status keterbatasannya:

1.  **Rate Limits:**
    *   Kecepatan respon tetap dibatasi oleh kuota gratis API Cloud jika menggunakan mode Cloud.
2.  **Kualitas Data Tabel (CSV):**
    *   Pencarian pada file CSV masih berbasis teks, belum mendukung analisis relasional kompleks.
3.  **Performa CPU:**
    *   Proses Re-ranking membebani CPU, namun kini lebih informatif dengan adanya progress bar.

---

## 4. Rencana Pengembangan (Roadmap)
- [x] Penambahan **Semantic Caching** menggunakan LanceDB (Selesai).
- [x] Implementasi **Chat History** menggunakan Buffer Memory (Selesai).
- [x] Integrasi **FastAPI** untuk mendukung akses multi-user (Selesai via `api.py`).
- [x] Antarmuka Grafis (Selesai via `app.py` Streamlit).
- [ ] Optimasi **OCR** untuk membaca teks di dalam gambar/scan PDF.
