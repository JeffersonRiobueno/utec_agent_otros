import uuid
import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

# Use OpenAI embeddings only to simplify local builds
EMB = OpenAIEmbeddings()

# Cargar .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

# EMB is already set to OpenAIEmbeddings above
print("[INFO] Usando OpenAIEmbeddings para embeddings.")

def ingest_faq_csv(csv_path, collection_name="faq_kb"):
    """Ingest FAQ data from CSV to Qdrant."""
    print(f"[INFO] Leyendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Eliminar colección si existe para recrear con dims correctas
    # QdrantClient 1.6.9 usa get_collection para verificar existencia o get_collections
    try:
        client.delete_collection(collection_name)
        print(f"[INFO] Colección '{collection_name}' eliminada para recrear.")
    except Exception:
        pass
    
    # Crear colección
    # QdrantClient 1.6.9 usa models.VectorParams
    from qdrant_client.http import models
    # OpenAI embeddings use 1536 dims by default (adjust if you change provider)
    vector_size = 1536
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )
    print(f"[INFO] Colección '{collection_name}' creada con {vector_size} dims.")
    
    documents = []
    ids = []
    for _, row in df.iterrows():
        content = f"Pregunta: {row['pregunta']}\nRespuesta: {row['respuesta']}\nCategoría: {row['categoria']}"
        doc = Document(
            page_content=content,
            metadata={
                "id": str(row['id']),
                "pregunta": row['pregunta'],
                "respuesta": row['respuesta'],
                "categoria": row['categoria']
            }
        )
        documents.append(doc)
        ids.append(str(uuid.uuid4()))  # Usar UUID como id único
    
    print(f"[INFO] Insertando/actualizando {len(documents)} documentos...")
    
    Qdrant.from_documents(
        documents,
        EMB,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=True
    )
    print("[INFO] Ingestión completada.")

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'faq_sample.csv')
    ingest_faq_csv(csv_path)