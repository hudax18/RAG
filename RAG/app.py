import streamlit as st
import os
from dotenv import load_dotenv
from RAG.main import HybridRAG

load_dotenv()

st.set_page_config(page_title="RAG in a Box", layout="wide")

st.title("📦 RAG in a Box")
st.markdown("Sistem Tanya Jawab Dokumen Pintar (Hybrid RAG)")

# Sidebar untuk konfigurasi
with st.sidebar:
    st.header("Konfigurasi")
    provider = st.selectbox("LLM Provider", ["gemini", "openai", "groq", "ollama"], 
                            index=["gemini", "openai", "groq", "ollama"].index(os.getenv("LLM_PROVIDER", "gemini")))
    
    if st.button("Simpan Konfigurasi"):
        os.environ["LLM_PROVIDER"] = provider
        st.success(f"Provider diganti ke {provider}")

# Inisialisasi RAG
@st.cache_resource
def get_rag():
    return HybridRAG()

rag = get_rag()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            # Untuk demo, kita pakai method query dari HybridRAG
            # Tapi kita modifikasi sedikit di main.py agar bisa mengembalikan source juga
            response = rag.query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
