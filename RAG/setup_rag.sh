#!/bin/bash

echo "=== Memulai Setup RAG System ==="

# 1. Cek apakah Python terinstal
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 tidak ditemukan. Silakan instal Python terlebih dahulu."
    exit
fi

# 2. Membuat Virtual Environment (opsional tapi disarankan)
echo "Membuat Virtual Environment (env_rag)..."
python3 -m venv env_rag
source env_rag/bin/activate

# 3. Upgrade pip
echo "Mengupgrade pip..."
pip install --upgrade pip

# 4. Instal library dari requirements.txt
echo "Menginstal library (ini mungkin memakan waktu beberapa menit)..."
pip install -r RAG/requirements.txt

echo ""
echo "=== Instalasi Selesai! ==="
echo "Untuk mulai menggunakan, jalankan:"
echo "source env_rag/bin/activate"
echo "export PYTHONPATH=\$PYTHONPATH:."
echo "python3 RAG/main.py"
