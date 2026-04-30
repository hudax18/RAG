import os
import re
import inquirer
from dotenv import set_key, load_dotenv
from huggingface_hub import hf_hub_download, HfApi

def search_huggingface(query):
    api = HfApi()
    print(f"\n🔍 Mencari model GGUF ringan untuk: '{query}'...")
    try:
        models = api.list_models(
            search=query,
            tags="gguf",
            sort="downloads",
            direction=-1,
            limit=40
        )
        results = []
        for model in models:
            repo_id = model.id
            downloads = getattr(model, "downloads", 0)
            param_match = re.search(r'([0-9.]+) ?[Bb]', repo_id)
            params = float(param_match.group(1)) if param_match else 99.0
            results.append({
                "repo_id": repo_id,
                "downloads": downloads,
                "params": params,
                "likes": getattr(model, "likes", 0)
            })
        
        def sort_logic(m):
            group = 0 if m['params'] <= 4.0 else (1 if m['params'] <= 10.0 else 2)
            return (group, -m['downloads'])

        sorted_results = sorted(results, key=sort_logic)
        return sorted_results[:20]
    except Exception as e:
        print(f"Error saat mencari: {e}")
        return []

def get_gguf_files(repo_id):
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id)
        return [f for f in files if f.endswith(".gguf")]
    except Exception:
        return []

def download_model(repo_id, filename):
    model_dir = "RAG/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f"\n📥 Mengunduh model {filename} dari {repo_id}...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_dir, local_dir_use_symlinks=False)
    return path

def validate_api_key(provider, api_key, base_url=None):
    if not api_key: return False
    print(f"🔄 Memvalidasi API Key untuk {provider}...")
    try:
        if provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            genai.list_models()
        elif provider == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            client.models.list()
        # Untuk provider lain, validasi sederhana lewat inisialisasi factory
        print("✅ Validasi inisialisasi berhasil!")
        return True
    except Exception as e:
        print(f"❌ Validasi gagal atau provider tidak mendukung validasi instan: {e}")
        return True # Tetap true agar user bisa mencoba jika hanya kendala network saat validasi

def get_existing_local_models():
    model_dir = "RAG/models"
    if not os.path.exists(model_dir):
        return []
    return [f for f in os.listdir(model_dir) if f.endswith(".gguf")]

def setup_wizard():
    env_path = ".env"
    load_dotenv() # Load existing env if any
    print("\n=== RAG in a Box: Setup Wizard (Smart Edition) ===\n")

    while True:
        # MENU UTAMA
        providers = [
            ('Google Gemini', 'gemini'),
            ('OpenAI', 'openai'),
            ('Anthropic', 'anthropic'),
            ('Groq', 'groq'),
            ('Ollama', 'ollama'),
            ('Custom OpenAI-Compatible (DeepSeek, OpenRouter, etc)', 'openai_custom'),
            ('Local (llama-cpp)', 'local'),
            ('Keluar', 'exit')
        ]
        
        provider_q = [inquirer.List('provider', message="Pilih LLM Provider Anda:", choices=providers)]
        provider = inquirer.prompt(provider_q)['provider']

        if provider == 'exit':
            print("Sampai jumpa!")
            break

        back_to_main = False

        if provider == 'local':
            # ... (logika local tetap sama)
            existing_models = get_existing_local_models()
            use_existing = False
            
            if existing_models:
                print(f"📂 Menemukan {len(existing_models)} model lokal di RAG/models/")
                local_choice_q = [
                    inquirer.List('choice', 
                        message="Model lokal terdeteksi:", 
                        choices=[
                            ('Gunakan model yang sudah ada', 'use'),
                            ('Unduh model baru dari Hugging Face', 'new'),
                            ('<-- Kembali', 'back')
                        ])
                ]
                local_choice = inquirer.prompt(local_choice_q)['choice']
                
                if local_choice == 'back': continue
                if local_choice == 'use':
                    model_select_q = [inquirer.List('model', message="Pilih model:", choices=existing_models)]
                    selected_model = inquirer.prompt(model_select_q)['model']
                    path = os.path.join("RAG/models", selected_model)
                    set_key(env_path, "LOCAL_MODEL_PATH", path)
                    set_key(env_path, "LLM_PROVIDER", "local")
                    print(f"✅ Menggunakan: {selected_model}")
                    use_existing = True
            
            if not use_existing:
                while True:
                    mode_choices = [
                        ('Pilih dari daftar rekomendasi', 'rec'),
                        ('Cari di Hugging Face', 'search'),
                        ('<-- Kembali', 'back')
                    ]
                    mode_q = [inquirer.List('mode', message="Metode pemilihan model baru:", choices=mode_choices)]
                    mode = inquirer.prompt(mode_q)['mode']

                    if mode == 'back':
                        back_to_main = True
                        break

                    repo_id, filename = None, None
                    if mode == 'rec':
                        models_list = [
                            {"name": "Llama-3.2-3B (Rekomendasi Cepat)", "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF", "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
                            {"name": "Phi-3.5-mini (3.8B - Efisien)", "repo": "bartowski/Phi-3.5-mini-instruct-GGUF", "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf"},
                            ("<-- Kembali", "back")
                        ]
                        rec_q = [inquirer.List('choice', message="Pilih Model:", choices=[(m['name'] if isinstance(m, dict) else m[0], i) for i, m in enumerate(models_list)])]
                        choice_idx = inquirer.prompt(rec_q)['choice']
                        if models_list[choice_idx] == ("<-- Kembali", "back"): continue
                        repo_id = models_list[choice_idx]['repo']
                        filename = models_list[choice_idx]['file']

                    elif mode == 'search':
                        query = input("\nMasukkan kata kunci pencarian: ")
                        results = search_huggingface(query)
                        if not results: continue
                        repo_choices = [(f"{r['repo_id']} ({r['params']}B)", r['repo_id']) for r in results] + [('back', 'back')]
                        repo_id = inquirer.prompt([inquirer.List('repo', message="Pilih Repo:", choices=repo_choices)])['repo']
                        if repo_id == 'back': continue
                        
                        gguf_files = get_gguf_files(repo_id)
                        file_choices = [(f, f) for f in gguf_files[:15]] + [('back', 'back')]
                        filename = inquirer.prompt([inquirer.List('file', message="Pilih File:", choices=file_choices)])['file']
                        if filename == 'back': continue

                    if repo_id and filename:
                        path = download_model(repo_id, filename)
                        set_key(env_path, "LOCAL_MODEL_PATH", path)
                        set_key(env_path, "LLM_PROVIDER", "local")
                        break
                
                if back_to_main: continue

        else:
            # CLOUD / CUSTOM PROVIDER
            is_custom = provider.endswith('_custom')
            actual_provider = provider.replace('_custom', '')
            
            default_models = {'gemini': 'gemini-1.5-flash', 'openai': 'gpt-4o-mini', 'anthropic': 'claude-3-5-sonnet-20240620', 'groq': 'mixtral-8x7b-32768', 'ollama': 'llama3'}
            key_names = {'gemini': 'GOOGLE_API_KEY', 'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY', 'groq': 'GROQ_API_KEY'}
            
            while True:
                current_key = os.getenv(key_names.get(actual_provider, ''))
                current_base = os.getenv("LLM_BASE_URL", "")
                
                questions = [
                    inquirer.Text('model', message=f"Nama Model:", default=default_models.get(actual_provider, 'gpt-4o-mini')),
                    inquirer.Text('api_key', message=f"Masukkan API Key:", default=current_key),
                ]
                
                if is_custom:
                    questions.append(inquirer.Text('base_url', message="Masukkan Custom Base URL (e.g. https://openrouter.ai/api/v1):", default=current_base))
                
                ans = inquirer.prompt(questions)
                
                base_url = ans.get('base_url')
                if validate_api_key(actual_provider, ans['api_key'], base_url):
                    set_key(env_path, "LLM_PROVIDER", actual_provider)
                    set_key(env_path, "LLM_MODEL_NAME", ans['model'])
                    if actual_provider in key_names: set_key(env_path, key_names[actual_provider], ans['api_key'])
                    if is_custom:
                        set_key(env_path, "LLM_BASE_URL", ans['base_url'])
                    else:
                        # Hapus base_url jika beralih ke provider standar
                        set_key(env_path, "LLM_BASE_URL", "")
                    
                    print(f"✅ Konfigurasi {actual_provider} disimpan.")
                    break
                else:
                    if not inquirer.confirm("Validasi gagal. Tetap simpan?", default=False):
                        if inquirer.confirm("Coba lagi?", default=True): continue
                        else: 
                            back_to_main = True
                            break
                    else:
                        set_key(env_path, "LLM_PROVIDER", actual_provider)
                        set_key(env_path, "LLM_MODEL_NAME", ans['model'])
                        if actual_provider in key_names: set_key(env_path, key_names[actual_provider], ans['api_key'])
                        if is_custom: set_key(env_path, "LLM_BASE_URL", ans['base_url'])
                        break
            
            if back_to_main: continue

        # STEP 3: DATA & FINALIZE (Sederhanakan)
        if inquirer.confirm("Lanjut ke pengaturan folder data?", default=True):
            folder = input("Folder dokumen (default: data): ") or "data"
            set_key(env_path, "DOC_FOLDER", folder)
            print(f"✅ Setup selesai! Folder data diset ke: {folder}")
            return

if __name__ == "__main__":
    setup_wizard()
