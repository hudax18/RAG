from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
import os

class LLMFactory:
    @staticmethod
    def get_llm(provider: str, model_name: str = None, api_key: str = None):
        # Gunakan model_name dari parameter atau dari environment, atau default provider
        final_model = model_name or os.getenv("LLM_MODEL_NAME")
        base_url = os.getenv("LLM_BASE_URL")
        # Pastikan base_url adalah None jika string kosong agar library menggunakan default official URL
        if not base_url:
            base_url = None
        
        if provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=final_model or "gemini-1.5-flash",
                google_api_key=api_key or os.getenv("GOOGLE_API_KEY")
            )
        elif provider == "openai":
            return ChatOpenAI(
                model=final_model or "gpt-4o-mini",
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url
            )
        elif provider == "groq":
            return ChatGroq(
                model=final_model or "mixtral-8x7b-32768",
                groq_api_key=api_key or os.getenv("GROQ_API_KEY"),
                base_url=base_url
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=final_model or "claude-3-5-sonnet-20240620",
                anthropic_api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                base_url=base_url
            )
        elif provider == "ollama":
            return ChatOllama(
                model=final_model or "llama3",
                base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )
        else:
            raise ValueError(f"Provider {provider} tidak didukung.")
