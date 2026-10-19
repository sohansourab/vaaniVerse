"""Unit tests for conversation memory and its plumbing into the prompt.

Offline: history is plain data, and the provider assembly is driven with
monkeypatched HTTP/SDK calls, so no model and no network are involved.
"""
from __future__ import annotations

import pytest

from reasoning import prompter
from reasoning.generator import Generator, GroqProvider, OllamaProvider
from reasoning.prompter import _trim_history, build_no_result_prompt, build_prompt
from retrieval.vector_store import RetrievedChunk


# ── helpers ────────────────────────────────────────────────────────────────────

def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id = "book_p1_0",
        text     = "జీవుడు దేహమును నిజమని భ్రమించుచున్నాడు.",
        score    = 0.71,
        metadata = {"source": "book.pdf", "page_number": 1, "script": "telugu"},
    )


class _FakeResp:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


# ── _trim_history ──────────────────────────────────────────────────────────────

def test_trim_discards_malformed_entries():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "not a chat turn"},
        {"role": "assistant", "content": "   "},
        {"nope": "no role"},
        {"role": "assistant", "content": "ok"},
        "a bare string",          # must not crash the trimmer
        None,
    ]
    assert _trim_history(history) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


def test_trim_keeps_only_the_last_turns(monkeypatch):
    monkeypatch.setattr(prompter, "MAX_HISTORY_TURNS", 1)  # → last 2 entries
    history = [{"role": "user", "content": str(i)} for i in range(5)]
    assert [t["content"] for t in _trim_history(history)] == ["3", "4"]


def test_trim_enforces_the_character_budget(monkeypatch):
    monkeypatch.setattr(prompter, "MAX_HISTORY_CHARS", 10)
    history = [
        {"role": "user", "content": "12345"},
        {"role": "assistant", "content": "67890"},
        {"role": "user", "content": "abcdefghij"},
    ]
    trimmed = _trim_history(history)
    assert trimmed == [{"role": "user", "content": "abcdefghij"}]


def test_trim_of_nothing_is_empty():
    assert _trim_history(None) == []
    assert _trim_history([]) == []


# ── build_prompt ───────────────────────────────────────────────────────────────

def test_build_prompt_carries_history_without_rendering_it():
    history = [
        {"role": "user", "content": "prior question about eruka"},
        {"role": "assistant", "content": "prior answer"},
    ]
    package = build_prompt("ఎరుక అంటే ఏమిటి?", [_chunk()], "telugu", history=history)

    assert package.history == history
    # History informs the model as chat turns, never as citable evidence.
    assert "prior question" not in package.user_prompt
    assert "SOURCE BLOCKS" in package.user_prompt


def test_build_no_result_prompt_carries_history_without_rendering_it():
    package = build_no_result_prompt(
        "ఎరుక అంటే ఏమిటి?",
        "telugu",
        history=[{"role": "user", "content": "prior question"}],
    )
    assert package.history == [{"role": "user", "content": "prior question"}]
    assert "prior question" not in package.user_prompt


# ── Generator plumbing ─────────────────────────────────────────────────────────

class _Recorder:
    name = "ollama"
    model_name = "fake"

    def __init__(self):
        self.history = "unset"
        self.system_prompt = None

    def generate(self, prompt, **kwargs):
        self.history = kwargs.get("history")
        self.system_prompt = kwargs.get("system_prompt")
        return "ok"


def test_generator_forwards_history_to_the_provider(monkeypatch):
    gen = Generator(provider="ollama")
    recorder = _Recorder()
    monkeypatch.setattr(gen, "get_provider", lambda name=None: recorder)

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    package = build_prompt("ఎరుక", [_chunk()], "telugu", history=history)

    gen.generate(package)
    assert recorder.history == package.history == history


def test_ollama_turns_history_into_chat_turns(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, **kwargs):
        captured["messages"] = json["messages"]
        return _FakeResp({"message": {"content": "ok"}})

    monkeypatch.setattr("reasoning.generator.httpx.post", fake_post)

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    OllamaProvider(base_url="http://localhost:11434", model="m").generate(
        "q", system_prompt="SYS", history=history
    )

    assert captured["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "q"},
    ]


def test_groq_turns_history_into_chat_turns(monkeypatch):
    captured = {}

    class _Msg:
        content = "ok"

    class _Choice:
        message = _Msg()

    class _Completions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return type("_Resp", (), {"choices": [_Choice()]})()

    class _Client:
        chat = type("_Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr("reasoning.generator.Groq", lambda api_key=None: _Client())

    history = [{"role": "user", "content": "hi"}]
    GroqProvider(api_key="k", model="m").generate("q", system_prompt="SYS", history=history)

    assert captured["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "q"},
    ]
