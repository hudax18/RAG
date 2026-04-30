# 📦 RAG in a Box: Hybrid & Exclusive RAG System

Sistem **Retrieval-Augmented Generation (RAG)** yang modular, dan mudah pakai. Mendukung penggunaan model Cloud (Gemini, OpenAI, Groq) dan model Lokal (GGUF via llama-cpp).

---

## 🏗️ Pemetaan Arsitektur

Proyek ini dipisahkan menjadi dua bagian utama: **Orkestrasi (Root)** untuk interaksi pengguna, dan **Core Logic (Folder RAG)** untuk mesin AI.

### 📍 Di Luar Folder (`Root/`)
Digunakan untuk manajemen, konfigurasi, dan akses eksternal:
*   **`cli.py`**: **Pusat Kontrol Utama.** Digunakan untuk menjalankan chat, ingest data, dan memulai server.
*   **`setup_rag.py`**: **Wizard Konfigurasi.** Alat interaktif untuk setting API Key, unduh model lokal, dan deteksi model yang ada.
*   **`api.py`**: **REST API (FastAPI).** Membungkus sistem RAG agar bisa diintegrasikan ke aplikasi chat eksternal (Web/Mobile).
*   **`.env`**: Menyimpan kredensial dan preferensi model (Cloud vs Local).

### 📂 Di Dalam Folder (`RAG/`)
Berisi logika inti dan penyimpanan data:
*   **`main.py`**: Logika alur kerja RAG (Retrieval -> Re-rank -> Generation).
*   **`ingest.py`**: Script untuk memproses dokumen (PDF, TXT, CSV, dll) ke database vektor.
*   **`app.py`**: Antarmuka Web berbasis **Streamlit**.
*   **`components/`**: Modul modular (LLM Factory, Vector Store, Query Router, Reranker).
*   **`lance_db/`**: Tempat penyimpanan database vektor (LanceDB).
*   **`models/`**: Tempat penyimpanan file model lokal (`.gguf`).

---

## 🚀 Panduan Memulai

### 1. Persiapan Lingkungan
Instal dependensi yang diperlukan:
```bash
pip install -r RAG/requirements.txt
```

### 2. Konfigurasi (Setup)
Jalankan wizard interaktif untuk memilih LLM (Cloud atau Lokal) dan mengatur folder data:
```bash
python3 cli.py setup
```
*Wizard akan memvalidasi API Key Anda atau menawarkan model lokal yang sudah ada.*

### 3. Ingest Data
Masukkan dokumen Anda (PDF/TXT/CSV) ke folder `data/`, lalu proses ke database:
```bash
python3 cli.py ingest --path data
```

---

## 🛠️ Cara Penggunaan

### 👾 Mode Terminal (CLI)
Untuk chat langsung di terminal:
```bash
python3 cli.py chat
```

### 🌐 Mode Web (Streamlit)
Untuk antarmuka grafis yang ramah pengguna:
```bash
streamlit run RAG/app.py
```

### 🔌 Mode Integrasi (REST API)
Untuk menghubungkan RAG ke aplikasi chat buatan Anda sendiri:
```bash
python3 cli.py serve
```
*Akses dokumentasi API di: `http://localhost:8000/docs`*

---

## ⚙️ Fitur-Fitur
*   **Mode Eksklusif:** Pilih Cloud untuk performa tinggi atau Local untuk privasi total.
*   **Semantic Caching:** Respon instan (< 0.1 detik) untuk pertanyaan yang sudah pernah ditanyakan sebelumnya menggunakan LanceDB.
*   **Custom API Support:** Mendukung provider kustom (OpenAI-compatible) seperti DeepSeek, OpenRouter, atau vLLM dengan Custom Base URL.
*   **Retrieve & Re-rank:** Menggunakan BGE-Reranker untuk akurasi jawaban yang jauh lebih baik.
*   **Universal Ingest:** Mendukung PDF, TXT, Markdown, CSV, dan Docx dengan progress bar visual.

## Keterbatasan
* Pengujian belum dilakukan menyeluruh, hanya sebagain saja, di antaranya: api gemini, groq dan llama.
* Custom API Support seperti openAI compatible dan provider lain belum diuji.
* Hanya pernah diuji pada file PDF.

---
*Dibuat sebagai proyek percobaan, perlu pengembangan / pengujian lebih lanjut*
