# IoT SLM RAG

An IoT domain expert chatbot powered by Retrieval-Augmented Generation (RAG). Uses OpenAI embeddings + Pinecone vector database + GPT-4o-mini to answer IoT questions with context from a curated knowledge base.

## Architecture

```
iot_docs/ (5 txt files)
    ↓ ingest.py
OpenAI text-embedding-3-small → Pinecone (serverless, aws us-east-1)
                                     ↓ rag.py retrieves top-3 chunks
                                 GPT-4o-mini generates answer
                                     ↓ api.py (FastAPI)
                              frontend/index.html (browser UI)
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd iot-slm-rag
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=iot-knowledge
```

### 3. Set up Pinecone (free tier)

1. Go to [https://www.pinecone.io](https://www.pinecone.io) and create a free account
2. In the Pinecone console, copy your **API Key** from the API Keys section
3. Paste it as `PINECONE_API_KEY` in your `.env` file
4. The index (`iot-knowledge`) will be created automatically by `ingest.py`

> Free tier includes 1 serverless index with 2GB storage — more than enough for this project.

### 4. Ingest documents into Pinecone

```bash
python src/ingest.py
```

Expected output:
```
============================================================
  IoT SLM RAG - Document Ingestion Pipeline
============================================================
[1/5] Loading documents from: .../iot_docs
      Loaded 5 document(s)
[2/5] Splitting documents into chunks...
      Created 187 chunks from 5 documents
[3/5] Initializing OpenAI embeddings (text-embedding-3-small)...
[4/5] Setting up Pinecone index...
      Creating new index 'iot-knowledge'...
[5/5] Upserting 187 chunks to Pinecone...
============================================================
  Ingestion complete! Your RAG knowledge base is ready.
============================================================
```

### 5. Start the API server

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://localhost:8000`

- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 6. Open the frontend

Open `frontend/index.html` directly in your browser (no build step needed):

```bash
# Windows
start frontend/index.html

# macOS
open frontend/index.html
```

## Example Questions to Test

```
How do I fix the WL_CONNECT_FAILED error on ESP32?
What are the differences between MQTT QoS 0, 1, and 2?
How do I wire a DHT22 sensor to an ESP32?
Show me Python code to publish MQTT messages with paho-mqtt.
How does LoRaWAN compare to Zigbee for IoT?
My ESP32 keeps getting brownout detector triggered. How do I fix it?
What is the maximum range of HC-SR04 ultrasonic sensor?
How do I implement deep sleep on ESP32 to save battery?
What MQTT port should I use for TLS connections?
How do I calibrate an LM35 temperature sensor with ESP32 ADC?
```

## API Reference

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I connect ESP32 to WiFi?"}'
```

Response:
```json
{
  "answer": "According to your knowledge base, to connect ESP32 to WiFi...",
  "sources": [
    {
      "content": "Station Mode (connect to existing router)...",
      "source": "esp32.txt"
    }
  ]
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "rag_loaded": true,
  "model": "gpt-4o-mini",
  "embeddings": "text-embedding-3-small",
  "index": "iot-knowledge"
}
```

## Project Structure

```
iot-slm-rag/
├── iot_docs/               # Knowledge base documents
│   ├── esp32.txt           # ESP32 specs, pinout, WiFi, errors
│   ├── mqtt.txt            # MQTT protocol, broker setup, code examples
│   ├── sensors.txt         # DHT11/22, LM35, PIR, ultrasonic wiring + code
│   ├── troubleshooting.txt # 28 Q&A IoT problems and solutions
│   └── protocols.txt       # MQTT vs CoAP vs HTTP, LoRaWAN, BLE, Zigbee
├── src/
│   ├── ingest.py           # One-time document ingestion pipeline
│   ├── rag.py              # RAG chain (retriever + LLM)
│   └── api.py              # FastAPI REST API
├── frontend/
│   └── index.html          # Single-file chatbot UI
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Costs

| Service | Usage | Estimated Cost |
|---------|-------|----------------|
| OpenAI Embeddings | ~187 chunks × 500 tokens ≈ 93K tokens | ~$0.01 (one-time) |
| GPT-4o-mini | Per query, ~1-2K tokens | ~$0.001 per question |
| Pinecone | Serverless free tier | Free |

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run uvicorn from the project root: `uvicorn src.api:app --reload`

**`Pinecone index not found`**
Run `python src/ingest.py` first to create and populate the index.

**Frontend shows "Cannot connect to API"**
Make sure the API server is running: `uvicorn src.api:app --reload`

**`WL_CONNECT_FAILED` type questions not answered well**
Re-run ingestion to ensure all documents are indexed: `python src/ingest.py`
