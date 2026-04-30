import os
import argparse
from tqdm import tqdm
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader, 
    CSVLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG.components.vector_store import VectorStore

def get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return PyPDFLoader(file_path)
    elif ext == '.txt':
        return TextLoader(file_path)
    elif ext == '.md':
        return UnstructuredMarkdownLoader(file_path)
    elif ext == '.csv':
        return CSVLoader(file_path)
    elif ext == '.docx':
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Format file {ext} tidak didukung.")

def ingest_file(file_path: str, vector_store: VectorStore):
    if not os.path.exists(file_path):
        print(f"File {file_path} tidak ditemukan.")
        return

    try:
        print(f"\n📄 Memproses: {os.path.basename(file_path)}")
        loader = get_loader(file_path)
        pages = loader.load()
        
        # Batasi jika terlalu besar untuk demo, tapi biarkan fleksibel
        if len(pages) > 500:
            print(f"⚠️ File besar terdeteksi ({len(pages)} halaman). Membatasi ke 500 halaman pertama.")
            pages = pages[:500]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(pages)

        print(f"🧩 Memecah menjadi {len(docs)} chunk...")
        
        # Gunakan tqdm untuk simulasi progress jika add_documents tidak mendukung batch progress natively
        # Atau jika add_documents lambat, kita bisa membagi docs menjadi batches
        batch_size = 50
        for i in tqdm(range(0, len(docs), batch_size), desc="Mengunggah ke Vector Store"):
            batch = docs[i:i + batch_size]
            vector_store.add_documents(batch)
            
        print(f"✅ Ingestion {os.path.basename(file_path)} sukses!")
    except Exception as e:
        print(f"❌ Error saat ingest {file_path}: {e}")

def ingest_folder(folder_path: str, vector_store: VectorStore):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for f in filenames:
            if f.lower().endswith(('.pdf', '.txt', '.md', '.csv', '.docx')):
                files.append(os.path.join(root, f))
    
    if not files:
        print(f"Tidak ada file yang didukung di {folder_path}")
        return

    print(f"📂 Menemukan {len(files)} file di {folder_path}")
    for f in files:
        ingest_file(f, vector_store)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal RAG Ingest")
    parser.add_argument("--file", type=str, help="Path ke file yang ingin di-ingest")
    args = parser.parse_args()

    vs = VectorStore()
    
    if args.file:
        ingest_file(args.file, vs)
    else:
        # Default fallback jika tidak ada argumen
        pdf_path = "Annual Report 2024.pdf"
        if os.path.exists(pdf_path):
            ingest_file(pdf_path, vs)
        else:
            print("Gunakan flag --file untuk memasukkan dokumen, misal:")
            print("python3 RAG/ingest.py --file data.txt")
