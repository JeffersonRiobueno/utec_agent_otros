

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

# ==============
# Config & setup
# ==============
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

# Use OpenAI embeddings only to simplify dependencies
EMB = OpenAIEmbeddings()
print("[INFO] Usando OpenAIEmbeddings para embeddings.")

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
