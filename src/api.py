import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

rag_chain = None
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("Starting IoT RAG API...")
    try:
        from src.rag import build_rag_chain
        rag_chain = build_rag_chain()
        print("RAG chain loaded successfully")
    except Exception as e:
        print(f"Failed to load RAG chain: {e}")
        raise
    yield
    print("Shutting down IoT RAG API")


app = FastAPI(
    title="IoT SLM RAG API",
    description="IoT domain expert assistant powered by RAG + GPT-4o-mini",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    content: str
    source: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_loaded": rag_chain is not None,
        "model": "gpt-4o-mini",
        "embeddings": "text-embedding-3-small",
        "index": os.getenv("PINECONE_INDEX_NAME", "iot-knowledge"),
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    try:
        from src.rag import ask
        result = ask(rag_chain, request.question.strip())
        return AnswerResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))
