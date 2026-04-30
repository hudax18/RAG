import os
import sys
import typer
import warnings
import logging
from typing import Optional
from dotenv import load_dotenv

# Sembunyikan warning dari library (LangChain, Tokenizer, dll)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Sembunyikan log TensorFlow/Hardware
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Sembunyikan warning tokenizer

# Konfigurasi logging dasar untuk hanya menampilkan ERROR
logging.basicConfig(level=logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

# Menambahkan root directory ke sys.path agar modul RAG bisa diimpor tanpa export PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = typer.Typer(help="RAG in a Box - CLI Orchestrator")

def check_env():
    if not os.path.exists(".env"):
        typer.secho("⚠️ File .env tidak ditemukan! Menjalankan setup wizard...", fg=typer.colors.YELLOW)
        from setup_rag import setup_wizard
        setup_wizard()
        load_dotenv()
        return True
    load_dotenv()
    return True

@app.command()
def setup():
    """Menjalankan Wizard Konfigurasi untuk LLM dan Data."""
    from setup_rag import setup_wizard
    setup_wizard()

@app.command()
def ingest(
    path: str = typer.Option("data", help="Path ke file atau folder dokumen"),
    is_file: bool = typer.Option(False, "--file", help="Set jika path adalah file tunggal")
):
    """Memproses dokumen ke dalam Vector Database."""
    check_env()
    from RAG.ingest import ingest_file, ingest_folder
    from RAG.components.vector_store import VectorStore
    
    vs = VectorStore()
    if is_file or os.path.isfile(path):
        ingest_file(path, vs)
    else:
        ingest_folder(path, vs)

@app.command()
def chat(
    model: Optional[str] = typer.Option(None, help="Path ke model lokal (opsional)")
):
    """Memulai mode chat interaktif."""
    check_env()
    from RAG.main import HybridRAG
    
    rag = HybridRAG(local_model_path=model)
    typer.secho("\n=== Hybrid RAG Interactive Chat ===", fg=typer.colors.CYAN, bold=True)
    typer.echo("Ketik 'exit' untuk keluar.\n")
    
    while True:
        q = typer.prompt("User")
        if q.lower() in ['exit', 'quit']:
            break
        
        try:
            ans = rag.query(q)
            typer.secho(f"\nAI: {ans}", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED)

@app.command()
def query(
    text: str = typer.Argument(..., help="Pertanyaan untuk RAG"),
    model: Optional[str] = typer.Option(None, help="Path ke model lokal (opsional)")
):
    """Menanyakan satu pertanyaan ke RAG dan mendapatkan jawaban."""
    check_env()
    from RAG.main import HybridRAG
    
    rag = HybridRAG(local_model_path=model)
    ans = rag.query(text)
    typer.echo("\n--- Response ---")
    typer.secho(ans, fg=typer.colors.GREEN)

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host untuk API server"),
    port: int = typer.Option(8000, help="Port untuk API server"),
    reload: bool = typer.Option(False, "--reload", help="Aktifkan auto-reload untuk development")
):
    """Menjalankan REST API Server (FastAPI) untuk integrasi eksternal."""
    check_env()
    import uvicorn
    typer.secho(f"🚀 Memulai RAG API Server di http://{host}:{port}", fg=typer.colors.MAGENTA, bold=True)
    uvicorn.run("api:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    app()
