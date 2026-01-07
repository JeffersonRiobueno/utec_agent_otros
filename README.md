# Agent Knowledge

Agente especializado en responder preguntas frecuentes (FAQs) de la tienda usando búsqueda semántica con Qdrant. Parte del sistema de agentes del e-commerce WhatsApp bot.

---

## Descripción

El Agent Knowledge maneja consultas generales sobre la tienda como:
- Tiempos de entrega
- Políticas de devolución
- Métodos de pago
- Información de contacto
- Ubicación de tienda física
- Etc.

Utiliza embeddings para búsqueda semántica en una base de conocimientos de FAQs almacenada en Qdrant.

---

## Comandos recomendados para levantar todo el stack

1. **Instala podman-compose si no lo tienes:**
   ```
   pip install podman-compose
   ```

2. **Levanta Qdrant (versión compatible) con Podman:**
   ```
   podman-compose -f docker-compose-qdrant.yml up -d
   ```

3. **Crea y activa el entorno virtual de Python:**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Instala dependencias del proyecto:**
   ```
   pip install -r requirements.txt
   ```

5. **Configura el archivo `.env`:**
  - Usa `EMBEDDINGS_PROVIDER=openai` para embeddings con OpenAI (Gemini/Ollama eliminados).

6. **Asegúrate de tener el modelo de embeddings disponible:**
  - Configura las API keys necesarias para OpenAI (`OPENAI_API_KEY`).

7. **Ingesta FAQs en Qdrant:**
   ```
   python3 scripts_ingesta/ingest_faq.py
   ```

8. **(Opcional) Verifica las FAQs ingresadas:**
   ```
   python3 scripts_ingesta/list_qdrant_products.py
   ```

9. **Levanta el Agent Knowledge (API FastAPI):**
   ```
   python3 main.py
   ```

---

## API Endpoints

### POST `/products_agent_search`
Busca respuestas relevantes a una consulta usando el agente con herramientas.

**Request:**
```json
{
  "text": "¿Cuáles son los tiempos de entrega?",
  "session_id": "optional-session-id",
  "context_summary": "optional-context",
  "provider": "openai",
  "model": "model-name",
  "temperature": 0.2
}
```

**Response:**
```json
{
  "result": "Respuesta adaptada a la consulta..."
}
```

### GET `/list_faqs`
Lista todas las FAQs almacenadas en Qdrant.

**Response:**
```json
{
  "faqs": [
    {
      "id": "faq-1",
      "pregunta": "¿Cuál es el tiempo de entrega?",
      "respuesta": "Los pedidos se entregan en 1-3 días hábiles...",
      "categoria": "entregas"
    }
  ],
  "total": 10
}
```

### POST `/update_faqs`
Actualiza FAQs en batch (inserta o actualiza existentes).

**Request:**
```json
{
  "faqs": [
    {
      "id": "faq-1",
      "pregunta": "¿Cuál es el tiempo de entrega?",
      "respuesta": "Los pedidos se entregan en 1-3 días hábiles...",
      "categoria": "entregas"
    }
  ]
}
```

**Response:**
```json
{
  "message": "Actualizado 1 FAQs exitosamente."
}
```

---

## Notas importantes

- Qdrant debe estar corriendo en la versión 1.6.9 para compatibilidad con langchain-community.
- El archivo de FAQs de ejemplo está en `data/faq_sample.csv`.
- El agente usa LangChain AgentExecutor con herramientas para búsqueda automática.
- Compatible con OpenAI (Gemini y Ollama eliminados para simplificar builds).

---

## Estructura del proyecto

- `main.py`: API FastAPI del Agent Knowledge con endpoints para búsqueda y gestión de FAQs.
- `vector/vector.py`: Lógica de búsqueda semántica y herramientas de FAQ.
- `scripts_ingesta/`: Scripts para ingestar y gestionar FAQs en Qdrant.
  - `ingest_faq.py`: Script para cargar FAQs desde CSV a Qdrant.
  - `list_qdrant_products.py`: Script para listar FAQs en Qdrant.
- `data/faq_sample.csv`: Archivo CSV con FAQs de ejemplo.
- `docker-compose-qdrant.yml`: Compose para Qdrant.
- `requirements.txt`: Dependencias Python.

---

## Formato del CSV de FAQs

El archivo CSV debe tener las columnas:
- `id`: Identificador único de la FAQ
- `pregunta`: La pregunta frecuente
- `respuesta`: La respuesta correspondiente
- `categoria`: Categoría de la FAQ (ej: entregas, pagos, devoluciones)

Ejemplo:
```csv
id,pregunta,respuesta,categoria
faq-1,¿Cuáles son los tiempos de entrega?,Los pedidos en Lima se entregan entre 1 y 3 días hábiles.,entregas
```

---

## Troubleshooting

- **Qdrant connection error:** Verifica que Qdrant esté corriendo en `http://localhost:6333`
- **API key errors:** Asegúrate de configurar `OPENAI_API_KEY`
- **Empty search results:** Verifica que las FAQs hayan sido ingestadas correctamente con `ingest_faq.py`

---

Sigue estos pasos para tener el Agent Knowledge funcionando correctamente como parte del sistema de agentes del e-commerce.
