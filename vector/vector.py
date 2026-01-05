

import warnings
from dotenv import load_dotenv
import os
from typing import List

# Suprimir warning deprecado de OllamaEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.tools import Tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# ==============
# Config & setup
# ==============
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Selección dinámica de embeddings
if EMBEDDINGS_PROVIDER == "openai":
    EMB = OpenAIEmbeddings()
    print("[INFO] Usando OpenAIEmbeddings para embeddings.")
elif EMBEDDINGS_PROVIDER == "gemini":
    EMB = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    print("[INFO] Usando GoogleGenerativeAIEmbeddings (Gemini) para embeddings.")
else:
    EMB = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="nomic-embed-text")
    print(f"[INFO] Usando OllamaEmbeddings para embeddings (modelo: nomic-embed-text, url: {OLLAMA_BASE_URL}).")

def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# FAQ Tools
def get_qdrant_faq_collection(name: str = "faq_kb") -> Qdrant:
    client = _client()
    return Qdrant(
        client=client,
        collection_name=name,
        embeddings=EMB,
    )

def search_faq(query: str, k: int = 3) -> str:
    """Busca en la base de conocimientos de FAQs."""
    if isinstance(query, dict):
        query = str(query.get('input', query))
    vectorstore = get_qdrant_faq_collection()
    docs = vectorstore.similarity_search(query, k=k)
    
    if not docs:
        return "No encontré información relevante en nuestras preguntas frecuentes."
    
    results = []
    for doc in docs:
        metadata = doc.metadata
        results.append(f"**Pregunta:** {metadata['pregunta']}\n**Respuesta:** {metadata['respuesta']}\n**Categoría:** {metadata['categoria']}\n")
    
    return "\n".join(results)

faq_tool = Tool(
    name="faq_search_tool",
    func=search_faq,
    description="Herramienta para buscar respuestas en las preguntas frecuentes de la tienda. Úsala para consultas sobre entregas, devoluciones, pagos, etc."
)
