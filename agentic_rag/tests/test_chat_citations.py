"""Seam S2: AgenticRAG.chat() over a real (temporary) knowledge index with a fake LLM."""

import json
from dataclasses import replace

import pytest

from agentic_rag.rag import AgenticRAG
from agentic_rag.settings import get_settings
from agentic_rag.tests.fakes import FakeKnowledgeLLM, FakeTranslator

OPENING_HOURS_URL = "https://www.hw.ac.uk/uk/students/library/opening-hours"
STUDY_SPACES_URL = "https://www.hw.ac.uk/uk/students/library/study-spaces"
SPORT_URL = "https://www.hw.ac.uk/uk/students/sport"

PAGES = [
    {
        "title": "Library opening hours",
        "url": OPENING_HOURS_URL,
        "category": "library",
        "extracted_text": (
            "The Edinburgh campus library is open 24 hours a day during term time. "
            "Over the summer vacation the library closes at 10pm on weekdays.\n\n"
            "Library staff answer questions at the help desk from 9am to 5pm. "
            "You need your student ID card to enter the library after 6pm."
        ),
    },
    {
        "title": "Library study spaces",
        "url": STUDY_SPACES_URL,
        "category": "library",
        "extracted_text": (
            "Group study rooms in the library can be booked online for up to three hours. "
            "Silent study areas are on the upper floor of the library."
        ),
    },
    {
        "title": "Sports centre",
        "url": SPORT_URL,
        "category": "sport",
        "extracted_text": "The Oriam sports centre offers a gym, a swimming pool and fitness classes.",
    },
]


@pytest.fixture
def knowledge_settings(tmp_path):
    knowledge_file = tmp_path / "knowledge.json"
    knowledge_file.write_text(json.dumps({"pages": PAGES}), encoding="utf-8")
    chroma_dir = tmp_path / "chroma"
    return replace(
        get_settings(),
        knowledge_file=knowledge_file,
        chroma_persist_dir=chroma_dir,
        ingest_state_file=chroma_dir / "ingest_state.json",
        chunk_size=1200,
        chunk_overlap=150,
        top_k=12,
        similarity_threshold=0.0,
    )


def build_rag(settings, llm, **kwargs) -> AgenticRAG:
    rag = AgenticRAG.from_settings(settings, llm=llm, **kwargs)
    rag.ingest(force=True)
    return rag


def test_cited_answer_lists_each_cited_page_with_title_and_link_numbered_as_in_the_text(knowledge_settings):
    llm = FakeKnowledgeLLM(
        "You can book a group study room online {cite:group study rooms}. "
        "During term the library is open 24 hours a day {cite:open 24 hours}."
    )
    rag = build_rag(knowledge_settings, llm)

    response = rag.chat("When is the library open and can I book a study room?", session_id="s1")

    assert response.used_retrieval is True
    assert response.answer_markdown == (
        "You can book a group study room online [1]. "
        "During term the library is open 24 hours a day [2]."
    )
    assert [(c["number"], c["title"], c["url"]) for c in response.citations] == [
        (1, "Library study spaces", STUDY_SPACES_URL),
        (2, "Library opening hours", OPENING_HOURS_URL),
    ]


def test_two_passages_from_the_same_page_share_one_citation(knowledge_settings):
    small_chunks = replace(knowledge_settings, chunk_size=200, chunk_overlap=0)
    llm = FakeKnowledgeLLM(
        "During term the library is open 24 hours a day {cite:open 24 hours}. "
        "After 6pm you need your student ID card to get in {cite:after 6pm}."
    )
    rag = build_rag(small_chunks, llm)

    response = rag.chat("What are the library opening hours?", session_id="s2")

    assert response.answer_markdown == (
        "During term the library is open 24 hours a day [1]. "
        "After 6pm you need your student ID card to get in [1]."
    )
    assert [(c["number"], c["url"]) for c in response.citations] == [(1, OPENING_HOURS_URL)]


def test_markers_that_point_at_no_evidence_are_removed_rather_than_left_dangling(knowledge_settings):
    llm = FakeKnowledgeLLM(
        "During term the library is open 24 hours a day {cite:open 24 hours}. "
        "It also has a rooftop cafe [42]."
    )
    rag = build_rag(knowledge_settings, llm)

    response = rag.chat("What are the library opening hours?", session_id="s3")

    assert response.answer_markdown == (
        "During term the library is open 24 hours a day [1]. It also has a rooftop cafe."
    )
    assert [c["number"] for c in response.citations] == [1]


def test_grouped_markers_become_one_marker_per_cited_page(knowledge_settings):
    llm = FakeKnowledgeLLM("The library has 24-hour access and bookable rooms [1, 2, 99].")
    rag = build_rag(knowledge_settings, llm)

    response = rag.chat("Tell me about the library opening hours and study rooms", session_id="s4")

    assert response.answer_markdown == "The library has 24-hour access and bookable rooms [1][2]."
    assert [c["number"] for c in response.citations] == [1, 2]


def test_answer_without_markers_still_links_the_pages_it_was_given_best_match_first(knowledge_settings):
    llm = FakeKnowledgeLLM("During term the library never closes.")
    rag = build_rag(knowledge_settings, llm)

    response = rag.chat("What are the library opening hours during term?", session_id="s5")

    urls = [c["url"] for c in response.citations]
    assert urls[0] == OPENING_HOURS_URL
    assert len(urls) == len(set(urls)) <= 3
    assert [c["number"] for c in response.citations] == list(range(1, len(urls) + 1))


ARABIC_LIBRARY_QUESTION = "ما هي ساعات عمل المكتبة خلال الفصل الدراسي؟"


def test_arabic_question_gets_an_arabic_answer_whose_citations_survive(knowledge_settings):
    translator = FakeTranslator({ARABIC_LIBRARY_QUESTION: "What are the library opening hours during term?"})
    llm = FakeKnowledgeLLM(
        "During term the library is open 24 hours a day {cite:open 24 hours}.",
        replies_in={"Arabic": "المكتبة مفتوحة على مدار الساعة خلال الفصل الدراسي {cite:open 24 hours}."},
    )
    rag = build_rag(knowledge_settings, llm, translator=translator)

    response = rag.chat(ARABIC_LIBRARY_QUESTION, session_id="s6")

    assert response.effective_language == "ar"
    assert response.answer_markdown == "المكتبة مفتوحة على مدار الساعة خلال الفصل الدراسي [1]."
    assert [(c["number"], c["url"]) for c in response.citations] == [(1, OPENING_HOURS_URL)]


def test_translation_outage_still_gives_a_cited_answer_in_the_students_language(knowledge_settings):
    llm = FakeKnowledgeLLM(
        "During term the library is open 24 hours a day {cite:open 24 hours}.",
        replies_in={"Arabic": "المكتبة مفتوحة على مدار الساعة {cite:open 24 hours}."},
    )
    rag = build_rag(knowledge_settings, llm, translator=FakeTranslator(fail=True))

    response = rag.chat(ARABIC_LIBRARY_QUESTION, session_id="s7")

    assert response.effective_language == "ar"
    assert response.answer_markdown == "المكتبة مفتوحة على مدار الساعة [1]."
    assert response.citations[0]["url"] == OPENING_HOURS_URL
