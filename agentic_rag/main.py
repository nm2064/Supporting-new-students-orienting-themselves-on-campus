"""FastAPI backend for the Agentic RAG system."""
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import validate_config, KNOWLEDGE_FILE
from rag import get_rag

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="Agentic RAG system with ChromaDB and OpenAI-powered multilingual generation",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class IngestRequest(BaseModel):
    force: bool = Field(default=False, description="Force re-ingestion even if file unchanged")


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks: int = 0
    file_hash: Optional[str] = None


class ChatRequest(BaseModel):
    # New canonical fields
    message: Optional[str] = Field(default=None, description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    ui_language: str = Field(default="en", description="Requested UI language")
    auto_detect: bool = Field(default=True, description="Enable language auto-detection")
    campus: Optional[str] = Field(default=None, description="Optional campus filter")
    location_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional location/category context"
    )

    # Legacy fields for compatibility
    question: Optional[str] = Field(default=None, description="Legacy: user's question")
    session_id: Optional[str] = Field(default=None, description="Legacy: session ID")


class Source(BaseModel):
    source: str
    chunk_id: str
    excerpt: str
    score: float


class Citation(BaseModel):
    chunk_id: str
    source_url: str


class ChatResponse(BaseModel):
    # Canonical response fields
    effective_language: str
    answer_markdown: str
    citations: list[Citation] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)
    needs_human_handoff: bool = False
    confidence: float = 0.0
    conversation_id: Optional[str] = None

    # Legacy compatibility fields
    answer: str
    sources: list[Source] = Field(default_factory=list)
    used_retrieval: bool
    clarifying_question: Optional[str] = None
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    knowledge_file_exists: bool
    chroma_initialized: bool
    timestamp: str


# Startup validation
@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    try:
        validate_config()
        print("Configuration validated successfully")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set the required environment variables in .env file")
        # Don't raise here to allow server to start, but endpoints will fail


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API."""
    rag = get_rag()
    
    return HealthResponse(
        status="healthy",
        knowledge_file_exists=os.path.exists(KNOWLEDGE_FILE),
        chroma_initialized=rag.vector_store.collection is not None,
        timestamp=datetime.utcnow().isoformat()
    )


# Ingest endpoint
@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest the knowledge file into the vector store.
    
    - **force**: If true, re-ingest even if the file hasn't changed
    """
    try:
        validate_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        rag = get_rag()
        result = rag.ingest(force=request.force)
        return IngestResponse(**result)
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# RAG Chat endpoint
@app.post("/rag-chat", response_model=ChatResponse)
async def rag_chat(request: ChatRequest):
    """
    Process a chat query through the Agentic RAG system.
    
    - **question**: The user's question
    - **session_id**: Optional session ID to maintain chat history
    
    Returns an answer with optional sources and clarifying questions.
    """
    try:
        validate_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    normalized_message = (request.message or request.question or "").strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conversation_id = request.conversation_id or request.session_id

    try:
        rag = get_rag()
        response = rag.chat(
            query=normalized_message,
            session_id=conversation_id,
            ui_language=request.ui_language,
            auto_detect=request.auto_detect,
            campus=request.campus,
            location_context=request.location_context,
        )

        return ChatResponse(
            effective_language=response.effective_language,
            answer_markdown=response.answer_markdown,
            citations=[Citation(**c) for c in response.citations],
            follow_up_suggestions=response.follow_up_suggestions,
            needs_human_handoff=response.needs_human_handoff,
            confidence=response.confidence,
            conversation_id=response.conversation_id,
            answer=response.answer,
            sources=[Source(**s) for s in response.sources],
            used_retrieval=response.used_retrieval,
            clarifying_question=response.clarifying_question,
            session_id=response.session_id,
        )
    
    except Exception as e:
        print(f"ERROR in /rag-chat: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


# Get session history endpoint
@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """Get the chat history for a specific session."""
    rag = get_rag()
    history = rag.get_session_history(session_id)
    return {"session_id": session_id, "history": history}


# Clear session endpoint
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear the chat history for a specific session."""
    rag = get_rag()
    rag.clear_session(session_id)
    return {"session_id": session_id, "status": "cleared"}


# Run the server
if __name__ == "__main__":
    import uvicorn
    
    print("Starting Agentic RAG API server...")
    print("Knowledge file:", KNOWLEDGE_FILE)
    print("Chroma store:", os.path.dirname(os.path.abspath(__file__)) + "/../chroma_store")
    print("")
    print("Available endpoints:")
    print("  - GET  /health")
    print("  - POST /ingest")
    print("  - POST /rag-chat")
    print("  - GET  /session/{session_id}")
    print("  - DELETE /session/{session_id}")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
