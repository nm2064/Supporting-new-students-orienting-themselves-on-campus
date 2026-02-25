"""Agentic RAG implementation with OpenAI embeddings + generation and multilingual support."""
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langdetect import DetectorFactory, LangDetectException, detect_langs

from config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    CHAT_MAX_TOKENS,
    CHAT_MODEL,
    CHAT_TEMPERATURE,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INGEST_STATE_FILE,
    KNOWLEDGE_FILE,
    LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD,
    OPENAI_API_KEY,
    SIMILARITY_THRESHOLD,
    TOP_K,
    TRANSLATION_MODEL,
    USE_AZURE_OPENAI,
)

if USE_AZURE_OPENAI and AZURE_OPENAI_ENDPOINT:
    from openai import AzureOpenAI

    openai_client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    print(f"Initialized Azure OpenAI client: {AZURE_OPENAI_ENDPOINT}")
else:
    from openai import OpenAI

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("Initialized standard OpenAI client")

DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en", "zh-Hans", "hi", "ar"}
LANGUAGE_LABELS = {
    "en": "English",
    "zh-Hans": "Simplified Chinese",
    "hi": "Hindi",
    "ar": "Modern Standard Arabic",
}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RouterDecision:
    action: str  # DIRECT_ANSWER | RETRIEVE | CLARIFY
    reasoning: str = ""
    clarifying_question: Optional[str] = None


@dataclass
class RagResponse:
    answer_markdown: str
    effective_language: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    needs_human_handoff: bool = False
    confidence: float = 0.0

    # Backward-compat fields
    answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    used_retrieval: bool = False
    clarifying_question: Optional[str] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    detection_confidence: float = 0.0


def _normalize_ui_language(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    cleaned = lang.strip()
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
    return mapping.get(cleaned.lower(), "en")


def _normalize_detected_language(lang: str) -> str:
    mapping = {
        "en": "en",
        "zh": "zh-Hans",
        "zh-cn": "zh-Hans",
        "zh-tw": "zh-Hans",
        "hi": "hi",
        "ar": "ar",
    }
    return mapping.get(lang.lower(), "en")


class DocumentProcessor:
    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def load_ingest_state() -> Dict[str, Any]:
        if os.path.exists(INGEST_STATE_FILE):
            with open(INGEST_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def save_ingest_state(state: Dict[str, Any]):
        os.makedirs(os.path.dirname(INGEST_STATE_FILE), exist_ok=True)
        with open(INGEST_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)

    @staticmethod
    def needs_reingestion() -> bool:
        if not os.path.exists(KNOWLEDGE_FILE):
            raise FileNotFoundError(f"Knowledge file not found: {KNOWLEDGE_FILE}")

        current_hash = DocumentProcessor.compute_file_hash(KNOWLEDGE_FILE)
        state = DocumentProcessor.load_ingest_state()
        return state.get("file_hash") != current_hash

    @staticmethod
    def recursive_chunk(
        text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> List[Tuple[str, int, int]]:
        chunks = []
        paragraphs = re.split(r"\n\s*\n", text)
        current_chunk = ""
        current_start = 0
        position = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                position += 1
                continue

            para_len = len(para)
            if len(current_chunk) + para_len + 2 > chunk_size and current_chunk:
                chunks.append((current_chunk.strip(), current_start, position - 1))
                overlap_text = (
                    current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                )
                current_chunk = overlap_text + "\n\n" + para
                current_start = position - len(overlap_text.split("\n\n"))
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = position

            position += para_len + 2

        if current_chunk:
            chunks.append((current_chunk.strip(), current_start, position))
        return chunks

    @classmethod
    def process_file(cls, filepath: str = KNOWLEDGE_FILE) -> List[Chunk]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        raw_chunks = cls.recursive_chunk(text)
        result = []
        for i, (chunk_text, start, end) in enumerate(raw_chunks):
            result.append(
                Chunk(
                    text=chunk_text,
                    source=os.path.basename(filepath),
                    chunk_id=f"chunk_{i:04d}",
                    start_pos=start,
                    end_pos=end,
                )
            )
        return result


class VectorStore:
    COLLECTION_NAME = "knowledge_base"

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR, settings=Settings(allow_reset=True)
        )
        self.collection = None
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.COLLECTION_NAME)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )

    def _embedding_model_name(self) -> str:
        if USE_AZURE_OPENAI and AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
            return AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        if USE_AZURE_OPENAI and AZURE_OPENAI_DEPLOYMENT:
            return AZURE_OPENAI_DEPLOYMENT
        return EMBEDDING_MODEL

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = openai_client.embeddings.create(
            model=self._embedding_model_name(),
            input=texts,
        )
        return [item.embedding for item in response.data]

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return

        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embed_texts(texts)
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "source": c.source,
                        "chunk_id": c.chunk_id,
                        "start_pos": c.start_pos,
                        "end_pos": c.end_pos,
                    }
                    for c in batch
                ],
                ids=[c.chunk_id for c in batch],
            )

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        campus: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        # campus/category are currently pass-through placeholders until metadata is enriched.
        _ = campus
        _ = category

        query_embedding = self.embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                score = 1 - distance
                retrieved.append(
                    RetrievedChunk(
                        chunk=Chunk(
                            text=doc,
                            source=metadata.get("source", "knowledge.txt"),
                            chunk_id=metadata.get("chunk_id", "unknown_chunk"),
                            start_pos=metadata.get("start_pos", 0),
                            end_pos=metadata.get("end_pos", 0),
                        ),
                        score=score,
                    )
                )
        return retrieved


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

    def __init__(self):
        self.vector_store = VectorStore()
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.session_language_locks: Dict[str, str] = {}

    def ingest(self, force: bool = False) -> Dict[str, Any]:
        if not os.path.exists(KNOWLEDGE_FILE):
            raise FileNotFoundError(f"Knowledge file not found: {KNOWLEDGE_FILE}")

        needs_reingest = force or DocumentProcessor.needs_reingestion()
        if not needs_reingest:
            state = DocumentProcessor.load_ingest_state()
            return {
                "status": "skipped",
                "message": "Knowledge file unchanged, no re-ingestion needed",
                "chunks": state.get("chunk_count", 0),
                "file_hash": state.get("file_hash", ""),
            }

        chunks = DocumentProcessor.process_file(KNOWLEDGE_FILE)
        self.vector_store.add_chunks(chunks)

        file_hash = DocumentProcessor.compute_file_hash(KNOWLEDGE_FILE)
        DocumentProcessor.save_ingest_state(
            {
                "file_hash": file_hash,
                "chunk_count": len(chunks),
                "last_ingested": str(os.path.getmtime(KNOWLEDGE_FILE)),
            }
        )

        return {
            "status": "success",
            "message": f"Ingested {len(chunks)} chunks from {KNOWLEDGE_FILE}",
            "chunks": len(chunks),
            "file_hash": file_hash,
        }

    @staticmethod
    def _supports_custom_temperature(model_name: str) -> bool:
        # Azure GPT-5 chat deployments currently reject non-default temperature values.
        return "gpt-5" not in model_name.lower()

    def _chat_json(self, system_prompt: str, user_prompt: str, model: str = CHAT_MODEL) -> Dict[str, Any]:
        chat_model = self._chat_model_name(model)
        params: Dict[str, Any] = {
            "model": chat_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._supports_custom_temperature(chat_model):
            params["temperature"] = 0.1

        response = openai_client.chat.completions.create(**params)
        content = (response.choices[0].message.content or "{}").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def _chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = CHAT_MODEL,
        temperature: float = CHAT_TEMPERATURE,
        max_tokens: int = CHAT_MAX_TOKENS,
    ) -> str:
        chat_model = self._chat_model_name(model)
        params: Dict[str, Any] = {
            "model": chat_model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._supports_custom_temperature(chat_model):
            params["temperature"] = temperature

        response = openai_client.chat.completions.create(**params)
        return (response.choices[0].message.content or "").strip()

    def _chat_model_name(self, fallback_model: str) -> str:
        if USE_AZURE_OPENAI and AZURE_OPENAI_CHAT_DEPLOYMENT:
            return AZURE_OPENAI_CHAT_DEPLOYMENT
        if USE_AZURE_OPENAI and AZURE_OPENAI_DEPLOYMENT:
            return AZURE_OPENAI_DEPLOYMENT
        return fallback_model

    def _detect_language(self, text: str) -> Tuple[str, float]:
        if not text.strip():
            return "en", 0.0
        try:
            candidates = detect_langs(text)
            if not candidates:
                return "en", 0.0
            best = candidates[0]
            return _normalize_detected_language(best.lang), float(best.prob)
        except LangDetectException:
            return "en", 0.0
        except Exception:
            return "en", 0.0

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or source_lang == target_lang:
            return text

        source_label = LANGUAGE_LABELS.get(source_lang, source_lang)
        target_label = LANGUAGE_LABELS.get(target_lang, target_lang)
        system_prompt = (
            "You are a faithful translator. Preserve meaning exactly, do not add facts, "
            "and keep citation markers like [1], [2] unchanged."
        )
        user_prompt = (
            f"Translate from {source_label} to {target_label}.\n\n"
            f"Text:\n{text}"
        )
        return self._chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=TRANSLATION_MODEL,
            temperature=0.0,
            max_tokens=max(300, len(text) * 2),
        )

    def _router_decision(self, query_en: str, chat_history: List[Dict[str, Any]]) -> RouterDecision:
        query_lower = query_en.lower().strip()
        for pattern in self.DIRECT_ANSWER_PATTERNS:
            if re.search(pattern, query_lower):
                return RouterDecision(
                    action="DIRECT_ANSWER",
                    reasoning="Greeting/social pattern matched.",
                )

        has_university_keyword = any(kw in query_lower for kw in self.RETRIEVAL_KEYWORDS)
        if len(query_en.split()) < 3 and not has_university_keyword:
            return RouterDecision(
                action="CLARIFY",
                reasoning="Query too vague.",
                clarifying_question="Could you share more details about what you need help with on campus?",
            )

        if has_university_keyword:
            return RouterDecision(action="RETRIEVE", reasoning="University keyword found.")
        return self._llm_router(query_en, chat_history)

    def _llm_router(self, query_en: str, chat_history: List[Dict[str, Any]]) -> RouterDecision:
        recent = chat_history[-4:] if chat_history else []
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in recent]
        )
        result = self._chat_json(
            system_prompt=(
                "You are a routing agent for a university assistant. Return JSON with "
                "action (DIRECT_ANSWER|RETRIEVE|CLARIFY), reasoning, clarifying_question."
            ),
            user_prompt=(
                f"Chat history:\n{history_text}\n\n"
                f"Query: {query_en}\n"
                "Choose RETRIEVE for university-specific facts, CLARIFY for vague input, "
                "DIRECT_ANSWER for greeting/social/general intent."
            ),
        )
        action = str(result.get("action", "RETRIEVE")).upper()
        if action not in {"DIRECT_ANSWER", "RETRIEVE", "CLARIFY"}:
            action = "RETRIEVE"
        return RouterDecision(
            action=action,
            reasoning=str(result.get("reasoning", "LLM fallback decision")),
            clarifying_question=result.get("clarifying_question"),
        )

    def _direct_answer_en(self, query_en: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        recent = chat_history[-4:] if chat_history else []
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in recent]
        )
        result = self._chat_json(
            system_prompt=(
                "You are a concise university assistant. Return JSON with keys: "
                "answer_markdown, follow_up_suggestions (array), needs_human_handoff (bool), confidence (0-1)."
            ),
            user_prompt=(
                f"History:\n{history_text}\n\n"
                f"User query: {query_en}\n"
                "If factual data is unknown, say so clearly. Keep answer concise."
            ),
        )
        answer = str(result.get("answer_markdown", "")).strip() or "How can I help you today?"
        follow_ups = result.get("follow_up_suggestions") or []
        if not isinstance(follow_ups, list):
            follow_ups = []
        confidence = result.get("confidence", 0.45)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.45
        return {
            "answer_markdown": answer,
            "follow_up_suggestions": [str(x) for x in follow_ups][:3],
            "needs_human_handoff": bool(result.get("needs_human_handoff", False)),
            "confidence": max(0.0, min(1.0, confidence)),
        }

    def _grounded_answer_en(
        self,
        query_en: str,
        relevant_chunks: List[RetrievedChunk],
        chat_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
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

        evidence_lines = []
        for idx, rc in enumerate(relevant_chunks, 1):
            evidence_lines.append(
                f"[{idx}] chunk_id={rc.chunk.chunk_id} source={rc.chunk.source}\n{rc.chunk.text}"
            )
        evidence_text = "\n\n".join(evidence_lines)

        recent = chat_history[-6:] if chat_history else []
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in recent]
        )
        result = self._chat_json(
            system_prompt=(
                "You are a university RAG assistant. Return JSON with keys: "
                "answer_markdown, follow_up_suggestions (array), needs_human_handoff (bool), confidence (0-1). "
                "Only make factual claims grounded in evidence. "
                "For factual claims include citation markers [1], [2], etc. "
                "If evidence is missing, explicitly say that information is unavailable."
            ),
            user_prompt=(
                f"Evidence:\n{evidence_text}\n\n"
                f"History:\n{history_text}\n\n"
                f"User query: {query_en}"
            ),
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
            "follow_up_suggestions": [str(x) for x in follow_ups][:3],
            "needs_human_handoff": bool(result.get("needs_human_handoff", False)),
            "confidence": max(0.0, min(1.0, confidence)),
        }

    def _extract_citation_indices(self, answer_markdown: str, max_index: int) -> List[int]:
        seen = set()
        for match in re.finditer(r"\[(\d+)\]", answer_markdown):
            idx = int(match.group(1)) - 1
            if 0 <= idx < max_index:
                seen.add(idx)
        if not seen and max_index > 0:
            seen = {0}
        return sorted(seen)

    def _build_citation_payloads(
        self, relevant_chunks: List[RetrievedChunk], answer_markdown: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        indices = self._extract_citation_indices(answer_markdown, len(relevant_chunks))
        citations: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        for idx in indices:
            rc = relevant_chunks[idx]
            citations.append(
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "source_url": rc.chunk.source,
                }
            )
            sources.append(
                {
                    "source": rc.chunk.source,
                    "chunk_id": rc.chunk.chunk_id,
                    "excerpt": rc.chunk.text[:220] + ("..." if len(rc.chunk.text) > 220 else ""),
                    "score": round(float(rc.score), 4),
                }
            )
        return citations, sources

    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        ui_language: str = "en",
        auto_detect: bool = True,
        campus: Optional[str] = None,
        location_context: Optional[Dict[str, Any]] = None,
    ) -> RagResponse:
        conversation_id = session_id or str(uuid.uuid4())
        if conversation_id not in self.sessions:
            self.sessions[conversation_id] = []
        history = self.sessions[conversation_id]

        normalized_ui_lang = _normalize_ui_language(ui_language)
        detection_language = normalized_ui_lang
        detection_confidence = 0.0

        if auto_detect:
            locked = self.session_language_locks.get(conversation_id)
            if locked in SUPPORTED_LANGUAGES:
                effective_language = locked
            else:
                detection_language, detection_confidence = self._detect_language(query)
                if (
                    detection_language in SUPPORTED_LANGUAGES
                    and detection_confidence >= LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD
                ):
                    effective_language = detection_language
                else:
                    effective_language = normalized_ui_lang
                self.session_language_locks[conversation_id] = effective_language
        else:
            self.session_language_locks.pop(conversation_id, None)
            effective_language = normalized_ui_lang

        print(
            json.dumps(
                {
                    "event": "language_resolution",
                    "conversation_id": conversation_id,
                    "ui_language": normalized_ui_lang,
                    "auto_detect": auto_detect,
                    "detected_language": detection_language,
                    "detection_confidence": round(detection_confidence, 4),
                    "effective_language": effective_language,
                }
            )
        )

        translation_notice = ""
        query_en = query
        if effective_language != "en":
            try:
                query_en = self._translate_text(query, effective_language, "en")
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "translation_error",
                            "stage": "query_to_en",
                            "conversation_id": conversation_id,
                            "error": str(exc),
                        }
                    )
                )
                effective_language = "en"
                translation_notice = "Note: translation failed, so I replied in English."
                query_en = query

        decision = self._router_decision(query_en, history)
        relevant_chunks: List[RetrievedChunk] = []

        if decision.action == "CLARIFY":
            clarifying = decision.clarifying_question or (
                "Could you share more details so I can help accurately?"
            )
            answer_payload = {
                "answer_markdown": "",
                "follow_up_suggestions": [],
                "needs_human_handoff": False,
                "confidence": 0.3,
            }
        elif decision.action == "DIRECT_ANSWER":
            clarifying = None
            answer_payload = self._direct_answer_en(query_en, history)
        else:
            clarifying = None
            retrieved = self.vector_store.search(
                query=query_en,
                top_k=TOP_K,
                campus=campus,
                category=(location_context or {}).get("category") if location_context else None,
            )
            relevant_chunks = [c for c in retrieved if c.score >= SIMILARITY_THRESHOLD]
            if not relevant_chunks:
                relevant_chunks = retrieved[:3]

            answer_payload = self._grounded_answer_en(query_en, relevant_chunks, history)

        answer_markdown = answer_payload["answer_markdown"]
        follow_ups = answer_payload.get("follow_up_suggestions", [])
        needs_handoff = bool(answer_payload.get("needs_human_handoff", False))
        confidence = float(answer_payload.get("confidence", 0.5))

        citations, sources = self._build_citation_payloads(relevant_chunks, answer_markdown)

        if effective_language != "en":
            try:
                if answer_markdown:
                    answer_markdown = self._translate_text(answer_markdown, "en", effective_language)
                if clarifying:
                    clarifying = self._translate_text(clarifying, "en", effective_language)
                translated_follow_ups = []
                for suggestion in follow_ups[:3]:
                    translated_follow_ups.append(
                        self._translate_text(str(suggestion), "en", effective_language)
                    )
                follow_ups = translated_follow_ups
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "translation_error",
                            "stage": "answer_from_en",
                            "conversation_id": conversation_id,
                            "error": str(exc),
                        }
                    )
                )
                effective_language = "en"
                translation_notice = "Note: translation failed, so I replied in English."

        if translation_notice:
            if answer_markdown:
                answer_markdown = f"{answer_markdown}\n\n_{translation_notice}_"
            else:
                answer_markdown = translation_notice

        print(
            json.dumps(
                {
                    "event": "rag_summary",
                    "conversation_id": conversation_id,
                    "action": decision.action,
                    "used_retrieval": decision.action == "RETRIEVE",
                    "retrieved_count": len(relevant_chunks),
                    "confidence": round(confidence, 4),
                    "effective_language": effective_language,
                }
            )
        )

        if answer_markdown:
            history.append({"role": "user", "content": query_en})
            history.append({"role": "assistant", "content": answer_markdown})
        elif clarifying:
            history.append({"role": "user", "content": query_en})
            history.append({"role": "assistant", "content": clarifying})
        if len(history) > 20:
            self.sessions[conversation_id] = history[-20:]

        response = RagResponse(
            answer_markdown=answer_markdown,
            effective_language=effective_language,
            citations=citations,
            follow_up_suggestions=[str(x) for x in follow_ups][:3],
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
        return response

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.session_language_locks:
            del self.session_language_locks[session_id]


rag_instance: Optional[AgenticRAG] = None


def get_rag() -> AgenticRAG:
    global rag_instance
    if rag_instance is None:
        rag_instance = AgenticRAG()
    return rag_instance
