"""Test doubles for system boundaries (OpenAI SDK, embeddings, translation)."""

from dataclasses import replace
from types import SimpleNamespace

from agentic_rag.settings import get_settings


class FakeOpenAIClient:
    """Stands in for the OpenAI SDK client at the system boundary; records requests."""

    def __init__(self, output_text='{"ok": true}', status="completed"):
        self.response_requests: list[dict] = []
        self.completion_requests: list[dict] = []
        self.embedding_requests: list[dict] = []
        self._output_text = output_text
        self._status = status
        self.responses = SimpleNamespace(create=self._create_response)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_completion))
        self.embeddings = SimpleNamespace(create=self._create_embedding)

    def _create_response(self, **params):
        self.response_requests.append(params)
        return SimpleNamespace(output_text=self._output_text, status=self._status)

    def _create_completion(self, **params):
        self.completion_requests.append(params)
        message = SimpleNamespace(content=self._output_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def _create_embedding(self, **params):
        self.embedding_requests.append(params)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.5]) for _ in params["input"]])


def luna_settings(**overrides):
    return replace(
        get_settings(),
        use_azure_openai=True,
        azure_openai_endpoint="https://example.services.ai.azure.com/openai/v1",
        azure_openai_api_key="unit-test-key",
        azure_openai_chat_deployment="gpt-6-luna",
        azure_openai_embedding_deployment="text-embedding-3-small",
        **overrides,
    )


class FakeKnowledgeLLM:
    """Fake adapter for the LLM gateway interface (chat_json, embed_texts).

    Embeddings are a deterministic bag of words, so retrieval behaves like a
    tiny real index. Answers come from a template in which ``{cite:<phrase>}``
    becomes the ``[n]`` marker of whichever evidence block contains <phrase>,
    which is what a real model does when it cites the evidence it was shown.
    """

    DIMENSIONS = 256

    def __init__(
        self,
        answer_template: str = "",
        follow_ups: list[str] | None = None,
        replies_in: dict[str, str] | None = None,
    ):
        self.answer_template = answer_template
        self.follow_ups = follow_ups or []
        # Like a real model told to reply in a language: if the system prompt
        # names one of these languages, the reply uses that template instead.
        self.replies_in = replies_in or {}
        self.prompts: list[dict[str, str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import math
        import re
        import zlib

        vectors = []
        for text in texts:
            vector = [0.0] * self.DIMENSIONS
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vector[zlib.crc32(word.encode()) % self.DIMENSIONS] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def chat_json(self, system_prompt: str, user_prompt: str, model=None, max_tokens=None) -> dict:
        import re

        self.prompts.append({"system": system_prompt, "user": user_prompt})
        blocks = re.split(r"(?m)^(?=\[\d+\])", user_prompt)
        evidence = {int(re.match(r"\[(\d+)\]", block).group(1)): block for block in blocks if re.match(r"\[\d+\]", block)}

        def cite(match: "re.Match[str]") -> str:
            phrase = match.group(1).lower()
            for number, block in sorted(evidence.items()):
                if phrase in block.lower():
                    return f"[{number}]"
            raise AssertionError(f"fake LLM was not shown evidence containing {phrase!r}")

        template = next(
            (reply for language, reply in self.replies_in.items() if language in system_prompt),
            self.answer_template,
        )
        answer = re.sub(r"\{cite:([^}]+)\}", cite, template)
        return {
            "answer_markdown": answer,
            "follow_up_suggestions": self.follow_ups,
            "needs_human_handoff": False,
            "confidence": 0.8,
        }


class FakeTranslator:
    """Fake adapter for the machine-translation seam.

    Known phrases translate cleanly. Anything else comes back with its citation
    brackets mangled, which is what real machine translation can do to "[1]".
    """

    def __init__(self, phrasebook: dict[str, str] | None = None, fail: bool = False):
        self.phrasebook = phrasebook or {}
        self.fail = fail

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if self.fail:
            raise ConnectionError("translation service unreachable")
        if text in self.phrasebook:
            return self.phrasebook[text]
        return text.replace("[", "【").replace("]", "】")
