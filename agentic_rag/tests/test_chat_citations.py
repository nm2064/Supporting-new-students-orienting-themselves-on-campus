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


def test_pages_are_indexed_as_passages_and_the_question_is_searched_as_a_query(knowledge_settings):
    llm = FakeKnowledgeLLM("The library is open 24 hours a day {cite:open 24 hours}.")
    rag = build_rag(knowledge_settings, llm)

    rag.chat("What are the library opening hours?", session_id="s8")

    page_types = {kind for kind, text in llm.embedded if "Library opening hours" in text}
    question_types = {kind for kind, text in llm.embedded if text == "What are the library opening hours?"}
    assert page_types == {"passage"}
    assert question_types == {"query"}


def test_adjacent_markers_for_the_same_page_collapse_into_one(knowledge_settings):
    small_chunks = replace(knowledge_settings, chunk_size=200, chunk_overlap=0)
    llm = FakeKnowledgeLLM("The library is open all day in term and needs an ID card late {cite:open 24 hours}{cite:after 6pm}.")
    rag = build_rag(small_chunks, llm)

    response = rag.chat("What are the library opening hours?", session_id="s9")

    assert response.answer_markdown == "The library is open all day in term and needs an ID card late [1]."


# ---- signposting to people for personal visa, medical, legal and crisis matters ---------

VISA_PAGE = "https://www.hw.ac.uk/students/international-students/uk-campus-students/visas-and-immigration"
CRISIS_PAGE = (
    "https://www.hw.ac.uk/students/health-and-wellbeing/edinburgh-campus/"
    "student-wellbeing-services/counselling/crisis-support"
)


def test_personal_visa_questions_still_get_the_facts_but_are_pointed_to_the_immigration_team(knowledge_settings):
    llm = FakeKnowledgeLLM("The library is open 24 hours a day {cite:open 24 hours}.")
    rag = build_rag(knowledge_settings, llm)

    response = rag.chat("Will my student visa be cancelled if I fail a course at the library?", session_id="v1")

    assert response.answer_markdown.startswith("The library is open 24 hours a day [1].")
    assert VISA_PAGE in response.answer_markdown
    assert response.needs_human_handoff is True


def test_a_crisis_message_gets_urgent_help_first_even_when_it_is_too_short_to_route(knowledge_settings):
    rag = build_rag(knowledge_settings, FakeKnowledgeLLM("unused"))

    response = rag.chat("I feel suicidal", session_id="c1")

    assert "999" in response.answer_markdown
    assert "116 123" in response.answer_markdown
    assert CRISIS_PAGE in response.answer_markdown
    assert response.needs_human_handoff is True
    assert response.clarifying_question is None


def test_everyday_questions_that_share_words_with_sensitive_topics_are_not_signposted(knowledge_settings):
    llm = FakeKnowledgeLLM("The Oriam sports centre offers a gym and fitness classes {cite:sports centre}.")
    rag = build_rag(knowledge_settings, llm)

    for i, question in enumerate([
        "Can I book a badminton court at the sports centre?",
        "Is the library still open during exams?",
        "Where do campus security or police patrol at night near the sports centre?",
    ]):
        response = rag.chat(question, session_id=f"n{i}")
        assert response.needs_human_handoff is False, question
        assert "advice" not in response.answer_markdown.lower(), question


# ---- privacy: server-side conversation history is short-lived ---------------------------


def test_an_idle_conversation_is_forgotten_after_thirty_minutes(knowledge_settings):
    from agentic_rag.rag import SessionStore

    clock = {"now": 1_000.0}
    store = SessionStore(idle_ttl_s=30 * 60, clock=lambda: clock["now"])
    rag = build_rag(knowledge_settings, FakeKnowledgeLLM("Open 24 hours {cite:open 24 hours}."), session_store=store)

    rag.chat("What are the library opening hours?", session_id="p1")
    clock["now"] += 29 * 60
    assert rag.get_session_history("p1") != []
    clock["now"] += 31 * 60
    assert rag.get_session_history("p1") == []


def test_the_server_keeps_a_bounded_number_of_conversations_dropping_the_least_recent(knowledge_settings):
    from agentic_rag.rag import SessionStore

    clock = {"now": 0.0}
    store = SessionStore(max_sessions=2, clock=lambda: clock["now"])
    rag = build_rag(knowledge_settings, FakeKnowledgeLLM("Open 24 hours {cite:open 24 hours}."), session_store=store)

    for session in ("a", "b", "c"):
        clock["now"] += 1
        rag.chat("What are the library opening hours?", session_id=session)

    assert rag.get_session_history("a") == []
    assert rag.get_session_history("b") != [] and rag.get_session_history("c") != []


def test_auto_detection_changes_language_when_student_switches_to_chinese(knowledge_settings):
    chinese = '请告诉我图书馆的开放时间。'
    llm = FakeKnowledgeLLM(
        'The library is open 24 hours a day {cite:open 24 hours}.',
        replies_in={'Simplified Chinese': '图书馆全天开放{cite:open 24 hours}。'},
    )
    rag = build_rag(knowledge_settings, llm, translator=FakeTranslator({chinese:'What are the library opening hours?'}))
    rag.chat('What are the library opening hours?', session_id='switch')
    answer = rag.chat(chinese, session_id='switch')
    assert answer.effective_language == 'zh-Hans'
    assert answer.answer_markdown == '图书馆全天开放[1]。'
    assert answer.citations[0]['url'] == OPENING_HOURS_URL


@pytest.mark.parametrize('question', ['GRID怎么走？', '图书馆', 'Learning Commons几点关门？'])
def test_chinese_script_is_detected_with_short_queries_and_english_place_names(knowledge_settings, question):
    llm = FakeKnowledgeLLM('Library information {cite:open 24 hours}.', replies_in={
        'Simplified Chinese':'图书馆全天开放{cite:open 24 hours}。'})
    rag = build_rag(knowledge_settings, llm, translator=FakeTranslator({question:'What are the library opening hours?'}))
    answer = rag.chat(question, session_id='mixed')
    assert answer.effective_language == 'zh-Hans'
    assert answer.answer_markdown == '图书馆全天开放[1]。'


def test_mixed_chinese_answer_is_localized_without_losing_citations(knowledge_settings):
    class MixedLanguageLLM(FakeKnowledgeLLM):
        def chat_json(self, system_prompt, user_prompt, **kwargs):
            if 'LOCALIZE_FINAL' in system_prompt:
                return {'answer_markdown':'图书馆全天开放[1]。学习共享空间开放至午夜。',
                        'follow_up_suggestions':['如何预约学习室？']}
            return super().chat_json(system_prompt, user_prompt, **kwargs)
    question = '图书馆什么时候开放？'
    rag = build_rag(knowledge_settings, MixedLanguageLLM(
        '图书馆全天开放{cite:open 24 hours}。Learning Commons open until midnight.',
        follow_ups=['How can I book a study room?']),
        translator=FakeTranslator({question:'What are the library opening hours?'}))
    response = rag.chat(question)
    assert response.answer_markdown == '图书馆全天开放[1]。学习共享空间开放至午夜。'
    assert response.follow_up_suggestions == ['如何预约学习室？']
    assert response.citations[0]['url'] == OPENING_HOURS_URL


def test_auto_detect_honors_explicit_reply_language_and_keeps_input_language_separate(knowledge_settings):
    llm = FakeKnowledgeLLM('The library is open {cite:open 24 hours}.', replies_in={
        'Simplified Chinese':'图书馆全天开放{cite:open 24 hours}。'})
    rag = build_rag(knowledge_settings, llm, translator=FakeTranslator())
    response = rag.chat('What are the library opening hours? Please answer in Chinese.', session_id='request')
    assert response.effective_language == 'zh-Hans'
    assert response.answer_markdown == '图书馆全天开放[1]。'


def test_short_ambiguous_acknowledgement_keeps_previous_language_and_clear_english_switches_back(knowledge_settings):
    chinese='图书馆什么时候开放？'
    llm=FakeKnowledgeLLM('Library information.', replies_in={'Simplified Chinese':'图书馆信息。'})
    rag=build_rag(knowledge_settings,llm,translator=FakeTranslator({chinese:'What are the library opening hours?'}))
    rag.chat(chinese,session_id='weak')
    assert rag.chat('ok',session_id='weak').effective_language == 'zh-Hans'
    assert rag.chat('What are the library opening hours?',session_id='weak').effective_language == 'en'


def test_manual_reply_language_is_respected_with_a_chinese_question(knowledge_settings):
    chinese='图书馆什么时候开放？'
    rag=build_rag(knowledge_settings,FakeKnowledgeLLM('The library is open {cite:open 24 hours}.'),translator=FakeTranslator({chinese:'What are the library opening hours?'}))
    result=rag.chat(chinese,ui_language='en',auto_detect=False)
    assert result.effective_language == 'en'
    assert result.used_retrieval


def test_chinese_citation_titles_are_localized_but_keep_original_links(knowledge_settings):
    class LocalizedSources(FakeKnowledgeLLM):
        def chat_json(self, *args, **kwargs):
            result = super().chat_json(*args, **kwargs)
            result['source_titles'] = {OPENING_HOURS_URL:'图书馆开放时间'}
            return result
    question='图书馆什么时候开放？'
    rag=build_rag(knowledge_settings,LocalizedSources('图书馆全天开放{cite:open 24 hours}。'),translator=FakeTranslator({question:'What are the library opening hours?'}))
    response=rag.chat(question)
    assert response.citations[0]['title'] == '图书馆开放时间'
    assert response.citations[0]['url'] == OPENING_HOURS_URL
    assert response.answer_markdown == '图书馆全天开放[1]。'


def test_language_repair_cannot_drop_citations_or_replace_source_links(knowledge_settings):
    class UnsafeRewrite(FakeKnowledgeLLM):
        def chat_json(self, system_prompt, user_prompt, **kwargs):
            if 'LOCALIZE_FINAL' in system_prompt:
                return {'answer_markdown':'图书馆开放。','follow_up_suggestions':[]}
            return super().chat_json(system_prompt,user_prompt,**kwargs)
    question='图书馆什么时候开放？'
    rag=build_rag(knowledge_settings,UnsafeRewrite('图书馆 open all day {cite:open 24 hours}.'),translator=FakeTranslator({question:'What are the library opening hours?'}))
    response=rag.chat(question)
    assert '[1]' in response.answer_markdown
    assert response.citations[0]['url'] == OPENING_HOURS_URL


@pytest.mark.parametrize('language,answer,suggestion', [
    ('en','The library is open 24 hours [1].','How do I book a room?'),
    ('zh-Hans','图书馆全天开放[1]。','如何预约学习室？'),
    ('hi','पुस्तकालय 24 घंटे खुला है[1]।','कमरा कैसे बुक करूँ?'),
    ('ar','المكتبة مفتوحة على مدار 24 ساعة[1].','كيف أحجز غرفة؟'),
])
def test_every_reply_language_repairs_foreign_prose_and_preserves_citations(knowledge_settings, language, answer, suggestion):
    class MultilingualRepair(FakeKnowledgeLLM):
        def chat_json(self, system_prompt, user_prompt, **kwargs):
            if 'LOCALIZE_FINAL' in system_prompt:
                return {'answer_markdown':answer,'follow_up_suggestions':[suggestion]}
            return super().chat_json(system_prompt,user_prompt,**kwargs)
    # English must repair non-English output too, not just the reverse.
    bad = {
        'en':'图书馆全天开放{cite:open 24 hours}。',
        'zh-Hans':'पुस्तकालय खुला है{cite:open 24 hours}।',
        'hi':'المكتبة مفتوحة{cite:open 24 hours}.',
        'ar':'Library open 24 hours {cite:open 24 hours}.',
    }[language]
    rag=build_rag(knowledge_settings,MultilingualRepair(bad),translator=FakeTranslator())
    result=rag.chat('What are the library opening hours?',ui_language=language,auto_detect=False)
    assert result.answer_markdown == answer
    assert result.follow_up_suggestions == [suggestion]
    assert result.citations[0]['url'] == OPENING_HOURS_URL


@pytest.mark.parametrize('language,question,reply', [
    ('en','What are the library opening hours?','The library is open 24 hours'),
    ('zh-Hans','Learning Commons几点关门？','图书馆全天开放'),
    ('hi','GRID कहाँ है?','पुस्तकालय 24 घंटे खुला है'),
    ('ar','أين مكتبة GRID؟','المكتبة مفتوحة على مدار الساعة'),
])
def test_every_language_can_be_detected_after_a_different_language(knowledge_settings, language, question, reply):
    llm=FakeKnowledgeLLM('The library is open 24 hours {cite:open 24 hours}.', replies_in={
        'Simplified Chinese':'图书馆全天开放{cite:open 24 hours}。',
        'Hindi':'पुस्तकालय 24 घंटे खुला है{cite:open 24 hours}।',
        'Modern Standard Arabic':'المكتبة مفتوحة على مدار الساعة{cite:open 24 hours}.'})
    rag=build_rag(knowledge_settings,llm,translator=FakeTranslator({question:'What are the library opening hours?'}))
    # Seed a different language in the same conversation.
    rag.chat('What are the library opening hours?',ui_language='ar' if language=='en' else 'en',auto_detect=False,session_id='languages')
    result=rag.chat(question,session_id='languages')
    assert result.effective_language == language
    assert reply in result.answer_markdown
    assert result.citations[0]['url'] == OPENING_HOURS_URL


@pytest.mark.parametrize('instruction,language', [
    ('请用阿拉伯语回答','ar'),
    ('الرجاء الرد باللغة الإنجليزية','en'),
    ('कृपया अंग्रेज़ी में उत्तर दें','en'),
    ('请用印地语回答','hi'),
])
def test_explicit_reply_language_requests_work_in_supported_scripts(knowledge_settings, instruction, language):
    rag=build_rag(knowledge_settings,FakeKnowledgeLLM('Library information.'),translator=FakeTranslator())
    assert rag.chat('Library opening hours? '+instruction).effective_language == language


@pytest.mark.parametrize('language,title', [('en','Library hours'),('zh-Hans','图书馆时间'),('hi','पुस्तकालय का समय'),('ar','ساعات المكتبة')])
def test_source_title_localization_preserves_links_in_every_language(knowledge_settings, language, title):
    class SourceTitles(FakeKnowledgeLLM):
        def chat_json(self,*args,**kwargs):
            result=super().chat_json(*args,**kwargs)
            result['source_titles']={OPENING_HOURS_URL:title}
            return result
    rag=build_rag(knowledge_settings,SourceTitles('24 {cite:open 24 hours}.'),translator=FakeTranslator())
    result=rag.chat('What are the library opening hours?',ui_language=language,auto_detect=False)
    assert result.citations[0]['title'] == title
    assert result.citations[0]['url'] == OPENING_HOURS_URL
    assert result.citations[0]['number'] == 1
