import json
from types import SimpleNamespace
from unittest.mock import Mock

from agentic_rag.rag import AgenticRAG, DocumentProcessor
from agentic_rag.rag import VectorStore
from agentic_rag.settings import reload_settings


def test_vector_search_excludes_other_campus_sources():
    store = VectorStore.__new__(VectorStore)
    store.backend = Mock()
    store.backend.embed_texts.return_value = [[0.1, 0.2]]
    store.collection = Mock()
    store.collection.count.return_value = 2
    store.collection.query.return_value = {
        'documents': [['Orkney library', 'Edinburgh library']],
        'metadatas': [[{'source': 'https://www.hw.ac.uk/campuses/orkney/facilities'},
                       {'source': 'https://www.hw.ac.uk/campuses/edinburgh/facilities'}]],
        'distances': [[0.1, 0.2]],
    }
    results = store.search('Library', top_k=2, campus='Edinburgh')
    assert len(results) == 1
    assert results[0].chunk.text == 'Edinburgh library'


def test_empty_knowledge_base_does_not_call_embeddings():
    store = VectorStore.__new__(VectorStore)
    store.backend = Mock()
    store.collection = Mock()
    store.collection.count.return_value = 0
    assert store.search('library', top_k=5, campus='Edinburgh') == []
    store.backend.embed_texts.assert_not_called()


class _DummyBackend:
    def chat_json(self, *args, **kwargs):
        raise AssertionError("LLM router should not be called in this test")


class _DummyVectorStore:
    collection = object()


def test_router_prefers_retrieve_for_broad_student_queries():
    settings = reload_settings()
    rag = AgenticRAG(
        settings=settings,
        backend=_DummyBackend(),
        vector_store=_DummyVectorStore(),
        document_processor=DocumentProcessor(settings),
    )

    decision = rag._router_decision(
        "How do I enrol and where can I find teaching timetables?",
        [],
    )

    assert decision.action == "RETRIEVE"
    assert "retrieval" in decision.reasoning.lower()


def test_direct_greeting_response_states_university_purpose():
    settings = reload_settings()
    rag = AgenticRAG(
        settings=settings,
        backend=_DummyBackend(),
        vector_store=_DummyVectorStore(),
        document_processor=DocumentProcessor(settings),
    )

    response = rag._direct_answer_en("hey", [], "Edinburgh")

    assert "UniBot" in response["answer_markdown"]
    assert "Heriot-Watt" in response["answer_markdown"]
    assert "student assistant" in response["answer_markdown"]
    assert "Edinburgh" in response["answer_markdown"]


def test_document_processor_uses_page_urls_for_json_sources(tmp_path, monkeypatch):
    knowledge_file = tmp_path / "knowledge_pages.json"
    knowledge_file.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "category": "academics",
                        "site_key": "hwu",
                        "site_label": "Heriot-Watt University",
                        "title": "Enrolment",
                        "url": "https://www.hw.ac.uk/students/enrolment",
                        "source_url": "https://www.hw.ac.uk/students",
                        "summary": "How to enrol.",
                        "headings": ["Enrolment"],
                        "extracted_text": "Enrolment details for current students.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KNOWLEDGE_FILE", str(knowledge_file))
    settings = reload_settings()

    chunks = DocumentProcessor(settings).process_file()

    assert chunks
    assert chunks[0].source == "https://www.hw.ac.uk/students/enrolment"
    assert chunks[0].chunk_id.startswith("page_0000_chunk_00_")
    assert "Title: Enrolment" in chunks[0].text
