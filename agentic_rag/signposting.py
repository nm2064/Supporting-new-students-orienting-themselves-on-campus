"""Point students to people for matters a chatbot must not advise on.

UniBot can share official information, but personal visa, medical and legal
situations need a qualified person, and a student in crisis needs urgent help
before anything else. ``signpost_for`` recognises these topics and returns the
message and the official service to direct the student to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HWU_STUDENTS = "https://www.hw.ac.uk/students"


@dataclass(frozen=True)
class Signpost:
    topic: str
    message: str
    # Crisis help goes before the answer; other signposts follow it.
    first: bool = False


_CRISIS = Signpost(
    topic="crisis",
    first=True,
    message=(
        "**If you are in danger or at risk of harming yourself, please get help now.** "
        "Call **999** in an emergency, or talk to Samaritans any time on **116 123** (free, 24/7). "
        "The university's [crisis support page]"
        f"({HWU_STUDENTS}/health-and-wellbeing/edinburgh-campus/student-wellbeing-services/counselling/crisis-support) "
        "lists more people you can contact today. You don't have to deal with this alone."
    ),
)

_SIGNPOSTS: list[tuple[re.Pattern[str], Signpost]] = [
    (
        re.compile(
            r"\b(suicid\w*|kill (my|him|her)self|end (my|it all)|self[- ]?harm\w*|hurt(ing)? myself|"
            r"want to die|don'?t want to (live|be here)|overdose)\b"
        ),
        _CRISIS,
    ),
    (
        re.compile(r"\b(visa|immigration|brp|evisa|cas|ukvi|biometric|leave to remain|work permit)\b"),
        Signpost(
            topic="visa",
            message=(
                "_For advice on your own visa or immigration situation, please speak to the university's "
                "immigration advisers. See [Visas and immigration]"
                f"({HWU_STUDENTS}/international-students/uk-campus-students/visas-and-immigration). "
                "I can share general information, but not personal immigration advice._"
            ),
        ),
    ),
    (
        re.compile(
            r"\b(symptom\w*|diagnos\w*|medication|prescription|ill|sick|injur\w*|pain|doctor|gp|"
            r"medical|pregnan\w*|infection|fever)\b"
        ),
        Signpost(
            topic="medical",
            message=(
                "_For medical advice, please speak to a health professional: register with or call the "
                f"[campus doctor]({HWU_STUDENTS}/health-and-wellbeing/edinburgh-campus/doctor), call "
                "NHS 24 on 111 when it's urgent, or 999 in an emergency. I can't give medical advice._"
            ),
        ),
    ),
    (
        re.compile(
            # Phrases rather than bare words: "court" and "police" also appear in
            # questions about sports courts and campus security.
            r"\b(legal (advice|action|help|problem|issue)|lawyer|solicitor|sue|sued|suing|court case|"
            r"taken to court|arrested|charged with|police (charged|arrested|caution)|tenancy dispute|"
            r"landlord dispute|deposit dispute|contract dispute|discriminat\w*|harass\w*)\b"
        ),
        Signpost(
            topic="legal",
            message=(
                "_For advice on a legal or personal dispute, the Students' Union "
                "[Advice Hub](https://www.hwunion.com/advice-hub) offers free, confidential help. "
                "I can't give legal advice._"
            ),
        ),
    ),
]


def signpost_for(*texts: str) -> Signpost | None:
    """The most urgent signpost matching any of ``texts``, or None."""
    combined = " ".join(text.lower() for text in texts if text)
    for pattern, signpost in _SIGNPOSTS:
        if pattern.search(combined):
            return signpost
    return None
