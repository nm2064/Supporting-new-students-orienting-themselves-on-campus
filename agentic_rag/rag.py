"""Agentic RAG implementation with structured runtime dependencies."""

from __future__ import annotations

from .streaming import streamed_answer

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, LangDetectException, detect_langs

from .llm import LLMGateway
from .signposting import signpost_for
from .logging_utils import log_event
from .settings import AppSettings, get_settings

logger = logging.getLogger("agentic_rag.rag")

DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en", "zh-Hans", "hi", "ar"}
LANGUAGE_LABELS = {
    "en": "English",
    "zh-Hans": "Simplified Chinese",
    "hi": "Hindi",
    "ar": "Modern Standard Arabic",
}
DEFAULT_CAMPUS = "Edinburgh"

GROUNDED_ANSWER_INSTRUCTIONS = """\
You are UniBot, the Heriot-Watt University assistant for new and current students.
Answer using ONLY the numbered evidence passages, which come from official university pages.

How to answer:
- Answer the student's actual question directly in the first sentence.
- Then be thorough. Include every relevant detail the evidence gives: steps in order, \
deadlines, dates and opening times, costs, eligibility, locations, contact details, and \
the pages the student should visit. Use short headed sections or bullet lists when there \
are several points.
- Put a citation marker such as [1] or [2][3] straight after each factual claim, using \
the evidence numbers. Never cite a number that is not in the evidence. Never invent \
facts, links, phone numbers or email addresses.
- If the evidence only partly answers the question, answer that part, say clearly what it \
does not cover, and point to the most relevant cited page or service.
- If the evidence does not answer the question at all, say so plainly and suggest who to contact.
- Assume the {campus} campus unless the student names another. If the evidence is about a \
different campus, say so.
- Write answer_markdown and follow_up_suggestions in {language}. Keep citation markers, \
URLs, email addresses and acronyms such as GRID unchanged. Translate headings,
facility/service names, days, times, labels and follow-up questions into {language}.
When answering in Chinese, use 学习共享空间 for Learning Commons and 爱丁堡校区 for
Edinburgh Campus; do not leave ordinary English descriptions in the answer.
Use local names or transliteration for proper names when appropriate.

Return a JSON object with keys:
answer_markdown (string, Markdown),
follow_up_suggestions (array of up to 3 short questions the student might ask next),
needs_human_handoff (boolean, true when the student should contact university staff),
confidence (number from 0 to 1: how fully the evidence answers the question),
source_titles (object mapping the exact source URLs in the evidence to their page titles
translated into {language}; translate descriptive words, keep the URLs unchanged).
"""


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str
    start_pos: int = 0
    end_pos: int = 0
    title: str = ""


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RouterDecision:
    action: str
    reasoning: str = ""
    clarifying_question: str | None = None


@dataclass
class RagResponse:
    answer_markdown: str
    effective_language: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    follow_up_suggestions: list[str] = field(default_factory=list)
    needs_human_handoff: bool = False
    confidence: float = 0.0
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    used_retrieval: bool = False
    clarifying_question: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None
    detection_confidence: float = 0.0


def _normalize_ui_language(language: str | None) -> str:
    if not language:
        return "en"
    mapping = {
        "en": "en",
        "en-us": "en",
        "en-gb": "en",
        "zh": "zh-Hans",
        "zh-cn": "zh-Hans",
        "zh-hans": "zh-Hans",
        "zh-tw": "zh-Hans",
        "cn": "zh-Hans",
        "hi": "hi",
        "ar": "ar",
    }
    return mapping.get(language.strip().lower(), "en")


def _normalize_detected_language(language: str) -> str:
    mapping = {
        "en": "en",
        "zh": "zh-Hans",
        "zh-cn": "zh-Hans",
        "zh-tw": "zh-Hans",
        "hi": "hi",
        "ar": "ar",
    }
    return mapping.get(language.lower(), "en")


def _normalize_campus(campus: str | None) -> str:
    if not campus or not campus.strip():
        return DEFAULT_CAMPUS
    value = campus.strip()
    lowered = value.lower()
    mapping = {
        "edinburgh": "Edinburgh",
        "edinburgh campus": "Edinburgh",
        "dubai": "Dubai",
        "dubai campus": "Dubai",
        "malaysia": "Malaysia",
        "malaysia campus": "Malaysia",
        "orkney": "Orkney",
        "orkney campus": "Orkney",
        "scottish borders": "Scottish Borders",
        "borders": "Scottish Borders",
        "scottish borders campus": "Scottish Borders",
    }
    return mapping.get(lowered, value)


# Backwards-compatible name: the gateway lives in llm.py.
OpenAIBackend = LLMGateway


class GoogleTranslatorAdapter:
    """Machine-translation adapter backed by Google Translate (deep-translator).

    ``translate`` raises on failure; callers decide how to degrade.
    """

    _LANGUAGE_CODES: dict[str, str] = {"zh-Hans": "zh-CN", "hi": "hi", "ar": "ar", "en": "en"}

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        return GoogleTranslator(
            source=self._LANGUAGE_CODES.get(source_language, source_language),
            target=self._LANGUAGE_CODES.get(target_language, target_language),
        ).translate(text)


class DocumentProcessor:
    """Reads and chunks the knowledge base file."""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.knowledge_file = settings.knowledge_file
        self.ingest_state_file = settings.ingest_state_file

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        digest = hashlib.md5()
        with filepath.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4096), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_ingest_state(self) -> dict[str, Any]:
        if self.ingest_state_file.exists():
            with self.ingest_state_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def save_ingest_state(self, state: dict[str, Any]) -> None:
        self.ingest_state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.ingest_state_file.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def needs_reingestion(self) -> bool:
        if not self.knowledge_file.exists():
            raise FileNotFoundError(f"Knowledge file not found: {self.knowledge_file}")

        current_hash = self.compute_file_hash(self.knowledge_file)
        state = self.load_ingest_state()
        return state.get("file_hash") != current_hash

    def recursive_chunk(self, text: str) -> list[tuple[str, int, int]]:
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap

        # Split large paragraphs by sentence so no single paragraph exceeds chunk_size
        def split_paragraph(para: str) -> list[str]:
            if len(para) <= chunk_size:
                return [para]
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_chunks: list[str] = []
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 > chunk_size and current:
                    sub_chunks.append(current.strip())
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip() if current else sentence
            if current:
                sub_chunks.append(current.strip())
            return sub_chunks if sub_chunks else [para]

        chunks: list[tuple[str, int, int]] = []
        paragraphs = re.split(r"\n\s*\n", text)
        current_chunk = ""
        current_start = 0
        position = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                position += 1
                continue

            sub_paragraphs = split_paragraph(paragraph)
            for sub_para in sub_paragraphs:
                sub_len = len(sub_para)
                if len(current_chunk) + sub_len + 2 > chunk_size and current_chunk:
                    chunks.append((current_chunk.strip(), current_start, position - 1))
                    overlap_text = (
                        current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    )
                    current_chunk = overlap_text + "\n\n" + sub_para
                    current_start = max(0, position - len(overlap_text.split("\n\n")))
                else:
                    current_chunk = f"{current_chunk}\n\n{sub_para}".strip() if current_chunk else sub_para
                    if not chunks:
                        current_start = position

                position += sub_len + 2

        if current_chunk:
            chunks.append((current_chunk.strip(), current_start, position))

        return chunks

    def process_file(self, filepath: Path | None = None) -> list[Chunk]:
        source_file = filepath or self.knowledge_file
        if source_file.suffix.lower() == ".json":
            return self._process_json_knowledge(source_file)
        with source_file.open("r", encoding="utf-8") as handle:
            text = handle.read()

        processed_chunks: list[Chunk] = []
        for index, (chunk_text, start, end) in enumerate(self.recursive_chunk(text)):
            processed_chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source_file.name,
                    chunk_id=f"chunk_{index:04d}",
                    start_pos=start,
                    end_pos=end,
                )
            )
        return processed_chunks

    def _process_json_knowledge(self, source_file: Path) -> list[Chunk]:
        with source_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        pages = payload.get("pages", [])
        processed_chunks: list[Chunk] = []
        chunk_index = 0

        for page_index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue

            title = str(page.get("title", "")).strip()
            category = str(page.get("category", "")).strip()
            site_label = str(page.get("site_label", "")).strip()
            url = str(page.get("url", "")).strip() or source_file.name
            summary = str(page.get("summary", "")).strip()
            headings = page.get("headings", [])
            extracted_text = str(page.get("extracted_text", "")).strip()
            if not extracted_text:
                continue

            page_sections: list[str] = []
            if title:
                page_sections.append(f"Title: {title}")
            if category:
                page_sections.append(f"Category: {category}")
            if site_label:
                page_sections.append(f"Site: {site_label}")
            page_sections.append(f"URL: {url}")
            if summary:
                page_sections.append(f"Summary: {summary}")
            if isinstance(headings, list):
                cleaned_headings = [str(item).strip() for item in headings if str(item).strip()]
                if cleaned_headings:
                    page_sections.append(f"Headings: {' | '.join(cleaned_headings)}")
            page_sections.append("Content:")
            page_sections.append(extracted_text)
            page_text = "\n\n".join(page_sections)

            for part_index, (chunk_text, start, end) in enumerate(self.recursive_chunk(page_text)):
                processed_chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=url,
                        chunk_id=f"page_{page_index:04d}_chunk_{part_index:02d}_{chunk_index:04d}",
                        start_pos=start,
                        end_pos=end,
                        title=title,
                    )
                )
                chunk_index += 1

        return processed_chunks


class VectorStore:
    COLLECTION_NAME = "knowledge_base"

    def __init__(self, settings: AppSettings, backend: OpenAIBackend):
        self.settings = settings
        self.backend = backend
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(allow_reset=True),
        )
        self.collection: Any | None = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.collection = self.client.get_collection(name=self.COLLECTION_NAME)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
        except Exception:
            logger.debug("Vector collection did not exist before refresh.")

        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.backend.embed_texts(texts)
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                        "title": chunk.title,
                    }
                    for chunk in batch
                ],
                ids=[chunk.chunk_id for chunk in batch],
            )

    def search(
        self,
        query: str,
        top_k: int,
        campus: str | None = None,
        category: str | None = None,
    ) -> list[RetrievedChunk]:
        _ = category

        if self.collection is None:
            self._ensure_collection()
        if self.collection is None:
            return []

        count = self.collection.count()
        if count == 0:
            return []
        query_embedding = self.backend.embed_texts([query], input_type="query")[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(count, top_k * 3),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        distances = results.get("distances") or []
        if not documents or not documents[0]:
            return retrieved

        for document, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
            source = metadata.get("source", "knowledge.txt")
            requested_campus = (campus or "").lower().replace(" ", "-")
            source_campus = next((name for name in ("edinburgh", "dubai", "malaysia", "orkney", "scottish-borders")
                                  if f"/campuses/{name}/" in source or f"/{name}-campus/" in source), None)
            if requested_campus and source_campus and source_campus != requested_campus:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk=Chunk(
                        text=document,
                        source=source,
                        chunk_id=metadata.get("chunk_id", "unknown_chunk"),
                        start_pos=metadata.get("start_pos", 0),
                        end_pos=metadata.get("end_pos", 0),
                        title=metadata.get("title", "") or "",
                    ),
                    score=1 - distance,
                )
            )

        return retrieved[:top_k]


@dataclass
class SessionState:
    history: list[dict[str, str]] = field(default_factory=list)
    language_lock: str | None = None
    last_used: float = 0.0


class SessionStore:
    """Thread-safe, short-lived in-memory conversation storage.

    Conversations are kept only while in use: one idle for ``idle_ttl_s`` is
    forgotten, and at most ``max_sessions`` are held (least recently used are
    dropped first). Nothing is written to disk.
    """

    def __init__(
        self,
        max_messages: int = 20,
        idle_ttl_s: float = 30 * 60,
        max_sessions: int = 2000,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_messages = max_messages
        self.idle_ttl_s = idle_ttl_s
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    # Callers hold self._lock.
    def _live(self, session_id: str) -> SessionState | None:
        state = self._sessions.get(session_id)
        if state is not None and self._clock() - state.last_used > self.idle_ttl_s:
            del self._sessions[session_id]
            return None
        return state

    def _touch(self, session_id: str) -> SessionState:
        state = self._live(session_id)
        if state is None:
            state = self._sessions[session_id] = SessionState()
        state.last_used = self._clock()
        if len(self._sessions) > self.max_sessions:
            now = self._clock()
            for stale in [sid for sid, s in self._sessions.items() if now - s.last_used > self.idle_ttl_s]:
                del self._sessions[stale]
            while len(self._sessions) > self.max_sessions:
                oldest = min(self._sessions, key=lambda sid: self._sessions[sid].last_used)
                del self._sessions[oldest]
        return state

    def history_for(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            state = self._live(session_id)
            return [dict(message) for message in state.history] if state else []

    def get_language_lock(self, session_id: str) -> str | None:
        with self._lock:
            state = self._live(session_id)
            return state.language_lock if state else None

    def set_language_lock(self, session_id: str, language: str) -> None:
        with self._lock:
            self._touch(session_id).language_lock = language

    def clear_language_lock(self, session_id: str) -> None:
        with self._lock:
            state = self._live(session_id)
            if state is not None:
                state.language_lock = None

    def append_exchange(self, session_id: str, user_message: str, assistant_message: str) -> None:
        with self._lock:
            state = self._touch(session_id)
            state.history.append({"role": "user", "content": user_message})
            state.history.append({"role": "assistant", "content": assistant_message})
            if len(state.history) > self.max_messages:
                state.history = state.history[-self.max_messages :]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


class AgenticRAG:
    RETRIEVAL_KEYWORDS = [
        "library",
        "parking",
        "housing",
        "dining",
        "deadline",
        "exam",
        "schedule",
        "tuition",
        "fee",
        "department",
        "hours",
        "location",
        "building",
        "health",
        "admission",
        "enrollment",
        "course",
        "enrol",
        "timetable",
        "wellbeing",
        "well-being",
        "counselling",
        "counseling",
        "accommodation",
        "hall",
        "society",
        "societies",
        "student group",
        "sports",
        "sport",
        "visa",
        "funding",
        "scholarship",
        "bursary",
        "support",
        "services",
        "graduate",
        "careers",
        "career",
    ]

    BROAD_RETRIEVAL_PATTERNS = [
        r"\bhow do i\b",
        r"\bwhere can i find\b",
        r"\bwhere do i\b",
        r"\bwhat (support|services|options|clubs|societies|fees|funding)\b",
        r"\bhow can i\b",
        r"\bcan i\b.*\b(enrol|apply|join|transfer|book|find)\b",
    ]

    DIRECT_ANSWER_PATTERNS = [
        r"^(hi|hello|hey|greetings)\b",
        r"how are you",
        r"what('s| is) your name",
        r"who are you",
        r"thank",
        r"^(ok|okay|great|cool|nice)\b",
        r"^(bye|goodbye)\b",
    ]

    def __init__(
        self,
        settings: AppSettings,
        backend: OpenAIBackend,
        vector_store: VectorStore,
        document_processor: DocumentProcessor,
        session_store: SessionStore | None = None,
        translator: Any | None = None,
    ):
        self.settings = settings
        self.backend = backend
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.session_store = session_store or SessionStore()
        self.translator = translator or GoogleTranslatorAdapter()

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        llm: Any | None = None,
        translator: Any | None = None,
        session_store: SessionStore | None = None,
    ) -> "AgenticRAG":
        """Build the assistant. ``llm`` and ``translator`` default to the real adapters."""
        backend = llm or LLMGateway(settings)
        return cls(
            settings=settings,
            backend=backend,
            vector_store=VectorStore(settings, backend),
            document_processor=DocumentProcessor(settings),
            session_store=session_store,
            translator=translator,
        )

    def ingest(self, force: bool = False) -> dict[str, Any]:
        if not self.settings.knowledge_file.exists():
            raise FileNotFoundError(f"Knowledge file not found: {self.settings.knowledge_file}")

        needs_reingest = force or self.document_processor.needs_reingestion()
        if not needs_reingest:
            state = self.document_processor.load_ingest_state()
            return {
                "status": "skipped",
                "message": "Knowledge file unchanged, no re-ingestion needed",
                "chunks": state.get("chunk_count", 0),
                "file_hash": state.get("file_hash", ""),
            }

        chunks = self.document_processor.process_file()
        self.vector_store.add_chunks(chunks)

        file_hash = self.document_processor.compute_file_hash(self.settings.knowledge_file)
        self.document_processor.save_ingest_state(
            {
                "file_hash": file_hash,
                "chunk_count": len(chunks),
                "last_ingested": self.settings.knowledge_file.stat().st_mtime,
            }
        )

        log_event(
            logger,
            "ingest_completed",
            chunks=len(chunks),
            force=force,
            knowledge_file=str(self.settings.knowledge_file),
        )

        return {
            "status": "success",
            "message": f"Ingested {len(chunks)} chunks from {self.settings.knowledge_file}",
            "chunks": len(chunks),
            "file_hash": file_hash,
        }

    def _detect_language(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return "en", 0.0
        # Script evidence is more reliable than statistical detection for short
        # questions containing campus names such as "GRID" or "Learning Commons".
        sample = re.sub(r"https?://\S+|`[^`]*`", "", text)
        latin_words = re.findall(r"[A-Za-z]+", sample)
        scripts = [("zh-Hans", r"[\u3400-\u9fff]"), ("ar", r"[\u0600-\u06ff]"), ("hi", r"[\u0900-\u097f]")]
        for language, pattern in scripts:
            count = len(re.findall(pattern, sample))
            if language == "zh-Hans" and re.search(r"[\u3040-\u30ff]", sample):
                continue
            if count >= 2 and (len(latin_words) <= 2 or count > sum(map(len, latin_words))):
                return language, 0.99
        if len(latin_words) <= 2 and not re.search(r"[^\x00-\x7f]", sample):
            return "en", 0.0
        try:
            candidates = detect_langs(text)
            if not candidates:
                return "en", 0.0
            best = candidates[0]
            return _normalize_detected_language(best.lang), float(best.prob)
        except LangDetectException:
            return "en", 0.0
        except Exception as error:
            logger.debug("Language detection failed: %s", error)
            return "en", 0.0

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Best-effort machine translation; returns the original text on failure."""
        if not text or source_lang == target_lang:
            return text
        try:
            translated = self.translator.translate(text, source_lang, target_lang)
            return translated if translated and translated.strip() else text
        except Exception as error:
            logger.warning(
                "Machine translation failed (%s→%s): %s", source_lang, target_lang, type(error).__name__
            )
            return text

    def _llm_router(self, query_en: str, chat_history: list[dict[str, Any]]) -> RouterDecision:
        recent = chat_history[-4:] if chat_history else []
        history_text = "\n".join(f"{message['role']}: {message['content']}" for message in recent)
        result = self.backend.chat_json(
            system_prompt=(
                "You are a routing agent for a university assistant. Return JSON with "
                "action (DIRECT_ANSWER|RETRIEVE|CLARIFY), reasoning, clarifying_question."
            ),
            user_prompt=(
                f"Chat history:\n{history_text}\n\n"
                f"Query: {query_en}\n"
                "Default to RETRIEVE for student-facing university questions, processes, "
                "services, links, or campus information even when broad. "
                "Use CLARIFY only when the query is too underspecified to answer helpfully "
                "after retrieval. Use DIRECT_ANSWER only for greeting/social/general intent."
            ),
            model=self.settings.chat_model,
        )
        action = str(result.get("action", "RETRIEVE")).upper()
        if action not in {"DIRECT_ANSWER", "RETRIEVE", "CLARIFY"}:
            action = "RETRIEVE"
        return RouterDecision(
            action=action,
            reasoning=str(result.get("reasoning", "LLM fallback decision")),
            clarifying_question=result.get("clarifying_question"),
        )

    def _router_decision(self, query_en: str, chat_history: list[dict[str, Any]], was_translated: bool = False) -> RouterDecision:
        query_lower = query_en.lower().strip()
        for pattern in self.DIRECT_ANSWER_PATTERNS:
            if re.search(pattern, query_lower):
                return RouterDecision(
                    action="DIRECT_ANSWER",
                    reasoning="Greeting or social pattern matched.",
                )

        has_university_keyword = any(keyword in query_lower for keyword in self.RETRIEVAL_KEYWORDS)
        has_broad_retrieval_intent = any(
            re.search(pattern, query_lower) for pattern in self.BROAD_RETRIEVAL_PATTERNS
        )
        # Skip the short-query vagueness check when the query was translated from another
        # language — translations can be imperfect and produce short results even for
        # valid questions, so go straight to retrieval instead of asking for clarification.
        if not was_translated and len(query_en.split()) < 3 and not has_university_keyword:
            return RouterDecision(
                action="CLARIFY",
                reasoning="Query too vague.",
                clarifying_question="Could you share more details about what you need help with on campus?",
            )

        if has_university_keyword or has_broad_retrieval_intent:
            return RouterDecision(
                action="RETRIEVE",
                reasoning="Student-facing retrieval intent detected.",
            )
        # For translated queries that don't match keywords, default to RETRIEVE
        if was_translated:
            return RouterDecision(
                action="RETRIEVE",
                reasoning="Translated query routed to retrieval.",
            )
        return self._llm_router(query_en, chat_history)

    def _direct_answer_en(
        self,
        query_en: str,
        chat_history: list[dict[str, Any]],
        campus: str,
    ) -> dict[str, Any]:
        query_lower = query_en.lower().strip()
        if re.search(r"^(hi|hello|hey|greetings)\b", query_lower):
            return {
                "answer_markdown": (
                    f"Hi. I'm UniBot, a Heriot-Watt student assistant. I assume {campus} campus "
                    "unless you tell me otherwise. I can help with "
                    "academics, enrolment, timetables, wellbeing, accommodation, campus services, "
                    "societies, sport, and fees."
                ),
                "follow_up_suggestions": [
                    "Ask about enrolment or timetables.",
                    "Ask about wellbeing or accommodation.",
                    "Ask about societies, sport, or fees.",
                ],
                "needs_human_handoff": False,
                "confidence": 0.9,
            }
        if re.search(r"what('s| is) your name|who are you", query_lower):
            return {
                "answer_markdown": (
                    f"I'm UniBot, a Heriot-Watt student assistant. I assume {campus} campus unless "
                    "you tell me otherwise. I help with university "
                    "information such as academics, student support, accommodation, campus services, "
                    "societies, sport, and fees."
                ),
                "follow_up_suggestions": [],
                "needs_human_handoff": False,
                "confidence": 0.9,
            }
        if re.search(r"thank", query_lower):
            return {
                "answer_markdown": (
                    "You're welcome. If you need Heriot-Watt information, ask about academics, "
                    "student support, accommodation, campus life, sport, or fees."
                ),
                "follow_up_suggestions": [],
                "needs_human_handoff": False,
                "confidence": 0.9,
            }
        if re.search(r"^(bye|goodbye)\b", query_lower):
            return {
                "answer_markdown": (
                    "Goodbye. If you need Heriot-Watt help later, I can assist with academics, "
                    "student support, accommodation, campus life, sport, and fees."
                ),
                "follow_up_suggestions": [],
                "needs_human_handoff": False,
                "confidence": 0.9,
            }

        recent = chat_history[-4:] if chat_history else []
        history_text = "\n".join(f"{message['role']}: {message['content']}" for message in recent)
        result = self.backend.chat_json(
            system_prompt=(
                "You are a concise university assistant. Return JSON with keys: "
                "answer_markdown, follow_up_suggestions (array), needs_human_handoff (bool), confidence (0-1)."
            ),
            user_prompt=(
                f"Default campus context: {campus}. Assume this campus unless the user specifies another one.\n\n"
                f"History:\n{history_text}\n\n"
                f"User query: {query_en}\n"
                "If factual data is unknown, say so clearly. Keep answer concise."
            ),
            model=self.settings.chat_model,
        )
        answer = str(result.get("answer_markdown", "")).strip() or "How can I help you today?"
        follow_ups = result.get("follow_up_suggestions") or []
        if not isinstance(follow_ups, list):
            follow_ups = []
        try:
            confidence = float(result.get("confidence", 0.45))
        except (TypeError, ValueError):
            confidence = 0.45
        return {
            "answer_markdown": answer,
            "follow_up_suggestions": [str(item) for item in follow_ups][:3],
            "needs_human_handoff": bool(result.get("needs_human_handoff", False)),
            "confidence": max(0.0, min(1.0, confidence)),
        }

    @streamed_answer
    def _grounded_answer(
        self,
        query: str,
        query_en: str,
        relevant_chunks: list[RetrievedChunk],
        chat_history: list[dict[str, Any]],
        campus: str,
        language: str,
    ) -> dict[str, Any]:
        """Answer from the evidence, written directly in ``language``.

        Returns the usual answer payload plus ``written_in``: the language the
        text is already in, so it is not machine-translated afterwards.
        """
        if not relevant_chunks:
            return {
                "answer_markdown": (
                    "I could not find that information in the current knowledge base. "
                    "Please ask another campus-related question or contact support."
                ),
                "follow_up_suggestions": [
                    "Ask about library opening hours",
                    "Ask about parking rules",
                    "Ask about housing services",
                ],
                "needs_human_handoff": True,
                "confidence": 0.25,
            }

        evidence_text = "\n\n".join(
            f"[{index}] {chunk.chunk.title or 'Untitled page'} ({chunk.chunk.source})\n{chunk.chunk.text}"
            for index, chunk in enumerate(relevant_chunks, start=1)
        )
        recent = chat_history[-6:] if chat_history else []
        history_text = "\n".join(f"{message['role']}: {message['content']}" for message in recent)
        language_label = LANGUAGE_LABELS.get(language, "English")
        question_lines = f"Student's question: {query}"
        if query_en.strip() != query.strip():
            question_lines += f"\nEnglish translation of the question: {query_en}"
        result = self.backend.chat_json(
            system_prompt=GROUNDED_ANSWER_INSTRUCTIONS.format(campus=campus, language=language_label),
            user_prompt=(
                f"Evidence from official Heriot-Watt pages:\n{evidence_text}\n\n"
                f"Conversation so far:\n{history_text or '(none)'}\n\n"
                f"{question_lines}"
            ),
            model=self.settings.chat_model,
            max_tokens=self.settings.answer_max_tokens,
        )
        answer = str(result.get("answer_markdown", "")).strip()
        if not answer:
            answer = "I could not generate a grounded answer from the retrieved evidence."
        follow_ups = result.get("follow_up_suggestions") or []
        if not isinstance(follow_ups, list):
            follow_ups = []
        try:
            confidence = float(result.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        return {
            "answer_markdown": answer,
            "follow_up_suggestions": [str(item) for item in follow_ups][:3],
            "needs_human_handoff": bool(result.get("needs_human_handoff", False)),
            "confidence": max(0.0, min(1.0, confidence)),
            "written_in": language,
            "source_titles": result.get("source_titles", {}),
        }

    # "[3]" or a grouped "[1, 4]", with any whitespace in front of it.
    _MARKER = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")
    _MARKER_RUN = re.compile(r"(?:\[\d+\])+")

    def _cite_pages(
        self,
        relevant_chunks: list[RetrievedChunk],
        answer_markdown: str,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Renumber evidence markers as citations in order of first use.

        Returns the rewritten answer, the citations (number, title, url) and the
        matching source excerpts, so every [n] in the text has a citation n.
        Markers that point at no evidence are dropped.
        """
        number_for_page: dict[str, int] = {}
        citations: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []

        def rewrite(match: re.Match[str]) -> str:
            numbers: list[int] = []
            for part in match.group(2).split(","):
                number = cite(int(part) - 1)
                if number is not None and number not in numbers:
                    numbers.append(number)
            if not numbers:
                return ""
            return match.group(1) + "".join(f"[{number}]" for number in numbers)

        def cite(index: int) -> int | None:
            if not 0 <= index < len(relevant_chunks):
                return None
            retrieved = relevant_chunks[index]
            chunk = retrieved.chunk
            if chunk.source not in number_for_page:
                number_for_page[chunk.source] = len(number_for_page) + 1
                citations.append(
                    {
                        "number": number_for_page[chunk.source],
                        "title": chunk.title or chunk.source,
                        "url": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "source_url": chunk.source,
                    }
                )
                sources.append(
                    {
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "excerpt": chunk.text[:220] + ("..." if len(chunk.text) > 220 else ""),
                        "score": round(float(retrieved.score), 4),
                    }
                )
            return number_for_page[chunk.source]

        rewritten = self._MARKER.sub(rewrite, answer_markdown)
        # "[1][1]" (two passages of one page cited together) reads as one citation.
        rewritten = self._MARKER_RUN.sub(
            lambda run: "".join(f"[{n}]" for n in dict.fromkeys(re.findall(r"\d+", run.group(0)))),
            rewritten,
        )
        if not citations:
            # The answer drew on the knowledge base but cited nothing inline:
            # still show the best-matching pages it was given, so the student
            # can check the claims.
            for index in sorted(
                range(len(relevant_chunks)), key=lambda i: relevant_chunks[i].score, reverse=True
            ):
                if len(number_for_page) >= self._MAX_CONSULTED_SOURCES:
                    break
                cite(index)
        return rewritten, citations, sources

    _MAX_CONSULTED_SOURCES = 3

    def chat(
        self,
        query: str,
        session_id: str | None = None,
        ui_language: str = "en",
        auto_detect: bool = True,
        campus: str | None = None,
        location_context: dict[str, Any] | None = None,
    ) -> RagResponse:
        conversation_id = session_id or str(uuid.uuid4())
        history = self.session_store.history_for(conversation_id)
        effective_campus = _normalize_campus(campus)
        if not campus:
            mentioned = [name for name in ("Edinburgh", "Dubai", "Malaysia", "Orkney", "Scottish Borders")
                         if re.search(r"\b"+re.escape(name)+r"\b", query, re.IGNORECASE)]
            if len(mentioned) == 1:
                effective_campus = mentioned[0]

        normalized_ui_language = _normalize_ui_language(ui_language)
        detected_language, detection_confidence = self._detect_language(query)
        confident = (detected_language in SUPPORTED_LANGUAGES and
                     detection_confidence >= self.settings.language_detection_confidence_threshold)
        if auto_detect:
            previous = self.session_store.get_language_lock(conversation_id)
            effective_language = detected_language if confident else (previous or normalized_ui_language)
            requests = {
                "zh-Hans": r"(?:\b(?:answer|reply|respond|write|speak)\s+(?:only\s+)?in\s+(?:simplified\s+)?Chinese\b|(?:用|使用|以)(?:简体)?中文(?:回答|回复|作答))",
                "en": r"(?:\b(?:answer|reply|respond|write|speak)\s+(?:only\s+)?in\s+English\b|(?:用|使用|以)英语(?:回答|回复|作答))",
                "ar": r"\b(?:answer|reply|respond|write|speak)\s+(?:only\s+)?in\s+Arabic\b",
                "hi": r"\b(?:answer|reply|respond|write|speak)\s+(?:only\s+)?in\s+Hindi\b",
            }
            native_names = {
                "en": ("英语|英文", "अंग्रेज़ी|अंग्रेजी", "الإنجليزية|الانجليزية"),
                "zh-Hans": ("中文|简体中文|汉语", "चीनी|सरलीकृत चीनी", "الصينية|الصينية المبسطة"),
                "hi": ("印地语", "हिंदी|हिन्दी", "الهندية"),
                "ar": ("阿拉伯语", "अरबी", "العربية"),
            }
            for language, (chinese, hindi, arabic) in native_names.items():
                requests[language] += (
                    r"|(?:用|使用|以)(?:" + chinese + r")(?:回答|回复|作答)"
                    r"|(?:" + hindi + r")\s+में\s+(?:उत्तर|जवाब|बात)"
                    r"|(?:أجب|اجب|الرد|رد|الإجابة)\s+(?:باللغة\s+)?ب?(?:" + arabic + r")"
                )
            matches = [(match.start(), language) for language, pattern in requests.items()
                       for match in re.finditer(pattern, query, re.IGNORECASE)
                       if not re.search(r"(?:not|never|don't)\s*$", query[:match.start()], re.IGNORECASE)]
            if matches:
                effective_language = max(matches)[1]
            self.session_store.set_language_lock(conversation_id, effective_language)
        else:
            self.session_store.clear_language_lock(conversation_id)
            effective_language = normalized_ui_language

        log_event(
            logger,
            "language_resolution",
            conversation_id=conversation_id,
            ui_language=normalized_ui_language,
            auto_detect=auto_detect,
            campus=effective_campus,
            detected_language=detected_language,
            detection_confidence=round(detection_confidence, 4),
            effective_language=effective_language,
        )

        # English version of the question, used for routing and retrieval. If
        # machine translation is unavailable this is the original text; the
        # original-language search below still finds multilingual matches.
        input_language = detected_language if confident else effective_language
        query_en = self._translate_text(query, input_language, "en")

        decision = self._router_decision(query_en, history, was_translated=input_language != "en")
        signpost = signpost_for(query_en, query)
        if signpost is not None and decision.action != "RETRIEVE":
            # Never answer a crisis or a personal visa/medical/legal question with
            # "please clarify" or small talk: look up the relevant support services.
            decision = RouterDecision(action="RETRIEVE", reasoning=f"Signposted topic: {signpost.topic}.")
        relevant_chunks: list[RetrievedChunk] = []

        if decision.action == "CLARIFY":
            clarifying = decision.clarifying_question or "Could you share more details so I can help accurately?"
            answer_payload = {
                "answer_markdown": "",
                "follow_up_suggestions": [],
                "needs_human_handoff": False,
                "confidence": 0.3,
            }
        elif decision.action == "DIRECT_ANSWER":
            clarifying = None
            answer_payload = self._direct_answer_en(query_en, history, effective_campus)
        else:
            clarifying = None
            retrieved = self.vector_store.search(
                query=query_en,
                top_k=self.settings.top_k,
                campus=effective_campus,
                category=(location_context or {}).get("category") if location_context else None,
            )
            # For non-English queries, always also search with the original query.
            # text-embedding-3-small has strong multilingual support, so the original
            # query often finds relevant chunks even when translation is imperfect or
            # the translated text is identical to the original (translation failure).
            if effective_language != "en":
                retrieved_orig = self.vector_store.search(
                    query=query,
                    top_k=self.settings.top_k,
                    campus=effective_campus,
                    category=(location_context or {}).get("category") if location_context else None,
                )
                # Merge: keep best score per chunk_id
                seen: dict[str, RetrievedChunk] = {r.chunk.chunk_id: r for r in retrieved}
                for r in retrieved_orig:
                    if r.chunk.chunk_id not in seen or r.score > seen[r.chunk.chunk_id].score:
                        seen[r.chunk.chunk_id] = r
                retrieved = sorted(seen.values(), key=lambda x: x.score, reverse=True)[: self.settings.top_k]
            relevant_chunks = [
                chunk
                for chunk in retrieved
                if chunk.score >= self.settings.similarity_threshold
            ]
            if not relevant_chunks:
                relevant_chunks = retrieved[: self.settings.top_k]
            answer_payload = self._grounded_answer(
                query,
                query_en,
                relevant_chunks,
                history,
                effective_campus,
                effective_language,
            )

        answer_markdown = answer_payload["answer_markdown"]
        follow_ups = answer_payload.get("follow_up_suggestions", [])
        needs_handoff = bool(answer_payload.get("needs_human_handoff", False))
        confidence = float(answer_payload.get("confidence", 0.5))

        answer_markdown, citations, sources = self._cite_pages(relevant_chunks, answer_markdown)
        translated_titles = answer_payload.get("source_titles", {})
        if isinstance(translated_titles, dict):
            for citation in citations:
                title = translated_titles.get(citation["url"])
                if isinstance(title, str) and title.strip():
                    citation["title"] = title.strip()

        # Grounded answers are written in the reply language by the model, so
        # citation markers stay intact. Only fixed English text (greetings,
        # clarifying questions) is machine-translated.
        written_in = answer_payload.get("written_in", "en")
        if written_in != effective_language:
            answer_markdown = self._translate_text(answer_markdown, written_in, effective_language)
            follow_ups = [
                self._translate_text(str(suggestion), written_in, effective_language)
                for suggestion in follow_ups[:3]
            ]
        if clarifying:
            clarifying = self._translate_text(clarifying, "en", effective_language)

        if signpost is not None:
            note = self._translate_text(signpost.message, "en", effective_language)
            parts = [note, answer_markdown] if signpost.first else [answer_markdown, note]
            answer_markdown = "\n\n".join(part for part in parts if part)
            needs_handoff = True

        # Repair mixed-language prose once, before storing or returning the final
        # answer. Reject rewrites that change citation markers or destination URLs.
        prose = re.sub(r"https?://\S+|[\w.+-]+@[\w.-]+", "", " ".join([answer_markdown, *map(str, follow_ups)]))
        prose = re.sub(r"\b[A-Z0-9]{2,}\b", "", prose)
        foreign_script = any(
            len(re.findall(pattern, prose)) >= 2
            for language, pattern in [("zh-Hans", r"[\u3400-\u9fff]"), ("hi", r"[\u0900-\u097f]"), ("ar", r"[\u0600-\u06ff]")]
            if language != effective_language
        )
        foreign_latin = effective_language != "en" and bool(re.search(r"\b[a-z][a-z]+\b", prose, re.IGNORECASE))
        if foreign_script or foreign_latin:
            try:
                localized = self.backend.chat_json(
                    system_prompt=(
                        "LOCALIZE_FINAL: Translate ALL prose to " + LANGUAGE_LABELS[effective_language] +
                        ". Translate facility names, headings and follow-up questions. Keep acronyms, URLs, "
                        "numbers, facts and every [n] citation exactly unchanged. Add no information. "
                        "Return JSON with answer_markdown and follow_up_suggestions."
                    ),
                    user_prompt=json.dumps({"answer_markdown": answer_markdown, "follow_up_suggestions": follow_ups}, ensure_ascii=False),
                    model=self.settings.chat_model,
                )
                candidate = localized.get("answer_markdown", "")
                same_markers = sorted(re.findall(r"\[\d+\]", candidate)) == sorted(re.findall(r"\[\d+\]", answer_markdown))
                same_urls = re.findall(r"https?://[^\s<>\)]+", candidate) == re.findall(r"https?://[^\s<>\)]+", answer_markdown)
                suggestions = localized.get("follow_up_suggestions")
                if isinstance(candidate, str) and candidate.strip() and same_markers and same_urls and isinstance(suggestions, list):
                    answer_markdown = candidate
                    follow_ups = [str(item) for item in suggestions[:3]]
            except Exception as error:
                logger.warning("Language repair unavailable (%s)", type(error).__name__)

        log_event(
            logger,
            "rag_summary",
            conversation_id=conversation_id,
            action=decision.action,
            used_retrieval=decision.action == "RETRIEVE",
            retrieved_count=len(relevant_chunks),
            confidence=round(confidence, 4),
            effective_language=effective_language,
            reasoning=decision.reasoning,
        )

        persisted_message = answer_markdown or clarifying
        if persisted_message:
            self.session_store.append_exchange(conversation_id, query_en, persisted_message)

        return RagResponse(
            answer_markdown=answer_markdown,
            effective_language=effective_language,
            citations=citations,
            follow_up_suggestions=[str(item) for item in follow_ups][:3],
            needs_human_handoff=needs_handoff,
            confidence=max(0.0, min(1.0, confidence)),
            answer=answer_markdown,
            sources=sources,
            used_retrieval=decision.action == "RETRIEVE",
            clarifying_question=clarifying,
            conversation_id=conversation_id,
            session_id=conversation_id,
            detection_confidence=detection_confidence,
        )

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        return self.session_store.history_for(session_id)

    def clear_session(self, session_id: str) -> None:
        self.session_store.clear(session_id)


rag_instance: AgenticRAG | None = None
_rag_lock = threading.Lock()


def get_rag() -> AgenticRAG:
    global rag_instance
    if rag_instance is not None:
        return rag_instance

    with _rag_lock:
        if rag_instance is None:
            rag_instance = AgenticRAG.from_settings(get_settings())
    return rag_instance


def reset_rag() -> None:
    global rag_instance
    with _rag_lock:
        rag_instance = None
