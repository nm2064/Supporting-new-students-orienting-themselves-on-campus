"""HTTP API routes for the backend."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from .config import validate_config
from .rag import AgenticRAG, get_rag
from .schemas import (
    Citation,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    Source,
)
from .settings import AppSettings, ConfigurationError, get_settings

router = APIRouter()

SettingsDep = Annotated[AppSettings, Depends(get_settings)]
RagDep = Annotated[AgenticRAG, Depends(get_rag)]


def _raise_config_error(error: Exception) -> None:
    raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: SettingsDep, rag: RagDep) -> HealthResponse:
    try:
        settings.validate_required()
        configuration_valid = True
        status = "healthy"
    except ConfigurationError:
        configuration_valid = False
        status = "degraded"

    return HealthResponse(
        status=status,
        knowledge_file_exists=settings.knowledge_file.exists(),
        chroma_initialized=rag.vector_store.collection is not None,
        timestamp=datetime.now(UTC).isoformat(),
        configuration_valid=configuration_valid,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, rag: RagDep) -> IngestResponse:
    try:
        validate_config()
        result = rag.ingest(force=request.force)
        return IngestResponse(**result)
    except ConfigurationError as error:
        _raise_config_error(error)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {error}") from error


@router.post("/rag-chat", response_model=ChatResponse)
async def rag_chat(request: ChatRequest, rag: RagDep) -> ChatResponse:
    try:
        validate_config()
    except ConfigurationError as error:
        _raise_config_error(error)

    normalized_message = (request.message or request.question or "").strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conversation_id = request.conversation_id or request.session_id

    try:
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
            citations=[Citation(**citation) for citation in response.citations],
            follow_up_suggestions=response.follow_up_suggestions,
            needs_human_handoff=response.needs_human_handoff,
            confidence=response.confidence,
            conversation_id=response.conversation_id,
            answer=response.answer,
            sources=[Source(**source) for source in response.sources],
            used_retrieval=response.used_retrieval,
            clarifying_question=response.clarifying_question,
            session_id=response.session_id,
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {error}") from error


@router.get("/session/{session_id}")
async def get_session_history(session_id: str, rag: RagDep) -> dict[str, object]:
    return {"session_id": session_id, "history": rag.get_session_history(session_id)}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str, rag: RagDep) -> dict[str, str]:
    rag.clear_session(session_id)
    return {"session_id": session_id, "status": "cleared"}
