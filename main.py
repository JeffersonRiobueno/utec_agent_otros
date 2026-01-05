import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor

from vector.vector import faq_tool, get_qdrant_faq_collection
from langchain_core.documents import Document
from typing import List, Dict

load_dotenv()

SYSTEM_PROMPT = (
    "Eres un asistente de soporte especializado en preguntas frecuentes de la tienda. "
    "Tu objetivo es responder consultas comunes sobre la tienda usando la base de conocimientos de FAQs. "
    "Tienes acceso a una herramienta: `faq_search_tool`. Úsala SIEMPRE para buscar información relevante. "
    "IMPORTANTE: La herramienta te devolverá respuestas de FAQs. TU TRABAJO es adaptar esa información a la consulta del usuario de forma amable y útil. "
    "Si la herramienta no encuentra nada relevante, sugiere contactar soporte humano."
)

app = FastAPI()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # openai | ollama | gemini
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")          # por proveedor
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

# (Opcional) URLs/keys por proveedor
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # requerido si usas gemini

class KnowledgeAgentRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    context_summary: Optional[str] = None
    provider: Optional[str] = DEFAULT_PROVIDER
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = DEFAULT_TEMPERATURE

class KnowledgeAgentResponse(BaseModel):
    result: str



# =========================
# Fábrica de LLMs
# =========================
def make_llm(
    provider: str,
    model: str,
    temperature: float
):
    provider = (provider or DEFAULT_PROVIDER).lower()

    if provider == "openai":
        # Requiere: OPENAI_API_KEY
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "ollama":
        # Requiere: Ollama corriendo localmente o remoto
        # Modelos típicos: "llama3.1", "qwen2.5", "phi3", etc.
        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    if provider == "gemini":
        # Requiere: GOOGLE_API_KEY
        if not GOOGLE_API_KEY:
            raise RuntimeError("Falta GOOGLE_API_KEY para usar Gemini.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=GOOGLE_API_KEY)

    raise ValueError(f"Proveedor LLM no soportado: {provider}. Usa: openai | ollama | gemini")

@app.post("/knowledge_agent_search", response_model=KnowledgeAgentResponse)
def knowledge_agent_endpoint(req: KnowledgeAgentRequest):
    print(f"[API] Nueva consulta recibida: '{req.text}' (session_id: {req.session_id})")
    # Ignorar context_summary, solo usar el mensaje del usuario

    llm = make_llm(req.provider, req.model, req.temperature)
    tools = [faq_tool]

    system_prompt = SYSTEM_PROMPT
    # No incluir context_summary

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # Ejecuta el agente de forma completamente automática
    result = executor.invoke({"input": req.text, "session_id": req.session_id})
    # El resultado puede estar en diferentes campos según el modelo
    if isinstance(result, dict) and "output" in result:
        return KnowledgeAgentResponse(result=str(result["output"]))

@app.get("/list_faqs")
def list_faqs():
    """Endpoint para listar todas las FAQs en Qdrant."""
    try:
        vectorstore = get_qdrant_faq_collection()
        # Obtener todos los puntos de la colección
        points = vectorstore.client.scroll(collection_name="faq_kb", limit=100)[0]  # scroll devuelve (points, next_page_offset)
        faqs = []
        for point in points:
            payload = point.payload
            metadata = payload.get("metadata", {})
            faqs.append({
                "id": point.id,
                "pregunta": metadata.get("pregunta"),
                "respuesta": metadata.get("respuesta"),
                "categoria": metadata.get("categoria")
            })
        return {"faqs": faqs, "total": len(faqs)}
    except Exception as e:
        return {"error": str(e)}


class FAQUpdateRequest(BaseModel):
    faqs: List[Dict[str, str]]  # Lista de {"id": str, "pregunta": str, "respuesta": str, "categoria": str}

@app.post("/update_faqs")
def update_faqs_endpoint(req: FAQUpdateRequest):
    """Endpoint para actualizar FAQs en batch. Si id existe, actualiza; si no, inserta."""
    try:
        vectorstore = get_qdrant_faq_collection()
        
        documents = []
        ids = []
        for faq in req.faqs:
            content = f"Pregunta: {faq['pregunta']}\nRespuesta: {faq['respuesta']}\nCategoría: {faq['categoria']}"
            doc = Document(
                page_content=content,
                metadata={
                    "id": faq['id'],
                    "pregunta": faq['pregunta'],
                    "respuesta": faq['respuesta'],
                    "categoria": faq['categoria']
                }
            )
            documents.append(doc)
            ids.append(faq['id'])
        
        # Upsert
        vectorstore.add_documents(documents, ids=ids)
        return {"message": f"Actualizado {len(documents)} FAQs exitosamente."}
    except Exception as e:
        return {"error": str(e)}
