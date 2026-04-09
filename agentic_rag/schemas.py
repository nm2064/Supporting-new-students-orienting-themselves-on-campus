"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    force: bool = Field(default=False, description="Force re-ingestion even if file unchanged")


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks: int = 0
    file_hash: str | None = None


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, description="User message")
    conversation_id: str | None = Field(default=None, description="Conversation ID")
    ui_language: str = Field(default="en", description="Requested UI language")
    auto_detect: bool = Field(default=True, description="Enable language auto-detection")
    campus: str | None = Field(default=None, description="Optional campus filter")
    location_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional location or category context",
    )

    question: str | None = Field(default=None, description="Legacy: user question")
    session_id: str | None = Field(default=None, description="Legacy: session ID")


class Source(BaseModel):
    source: str
    chunk_id: str
    excerpt: str
    score: float


class Citation(BaseModel):
    chunk_id: str
    source_url: str


class ChatResponse(BaseModel):
    effective_language: str
    answer_markdown: str
    citations: list[Citation] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)
    needs_human_handoff: bool = False
    confidence: float = 0.0
    conversation_id: str | None = None

    answer: str
    sources: list[Source] = Field(default_factory=list)
    used_retrieval: bool
    clarifying_question: str | None = None
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    knowledge_file_exists: bool
    chroma_initialized: bool
    timestamp: str
    configuration_valid: bool
