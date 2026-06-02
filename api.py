"""
api.py — FastAPI Web Server for InnovLabs RAG Chatbot
=====================================================
Endpoints:
  GET  /          → Serves the chat UI (frontend/index.html)
  POST /chat      → {"question": "..."} → {"answer": "...", "sources": [...]}
  GET  /health    → {"status": "ok", "provider": {...}}
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from rag_engine import get_rag_chain, ask


# ==============================================================================
# APP LIFECYCLE
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG chain once at startup."""
    print("=" * 60)
    print("  Scientific RAG")
    print("=" * 60)
    get_rag_chain()   # warm-up: loads embeddings / chain
    yield
    print("Shutting down RAG API...")


app = FastAPI(
    title="Scientific RAG",
    description="Retrieval-Augmented Generation API for scientific knowledge base",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

class ChatRequest(BaseModel):
    question: str
    provider_model: str | None = None # Format: "provider:model"
    user_id: str | None = "public"    # Default to public access


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    image_url: str | None = None           # first image (backward compat)
    image_urls: list[str] = []             # all images (multi-protein queries)
    provenance_type: str | None = None
    provider: dict | None = None


# ==============================================================================
# ROUTES
# ==============================================================================

@app.get("/health")
def health():
    """Kubernetes liveness/readiness probe."""
    _, _, provider_info = get_rag_chain()
    return {"status": "ok", "provider": provider_info}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question and get an answer grounded in your documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        # Parse provider:model if provided
        provider_id, model_id = None, None
        if request.provider_model and ":" in request.provider_model:
            provider_id, model_id = request.provider_model.split(":", 1)

        chain, retriever, provider_info = get_rag_chain(provider_id, model_id)
        result = ask(chain, retriever, request.question, user_id=request.user_id)

        # Convert list of image paths to browser-accessible URLs
        resolved_urls = []
        for path in result.get("image_paths", []):
            if path.startswith("http"):
                resolved_urls.append(path)
            else:
                resolved_urls.append(f"/images/{os.path.basename(path)}")

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            image_url=resolved_urls[0] if resolved_urls else None,   # backward compat
            image_urls=resolved_urls,
            provenance_type=result.get("provenance_type", "Unknown"),
            provider=provider_info
        )
    except Exception as e:
        err = str(e)
        if "RESOURCE_EXHAUSTED" in err:
            raise HTTPException(status_code=429, detail="AI quota exceeded. Please wait and retry.")
        raise HTTPException(status_code=500, detail=err)


# ==============================================================================
# STATIC FRONTEND + IMAGES
# ==============================================================================

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

# Serve the chat UI
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_ui():
        """Serve the chat UI."""
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
