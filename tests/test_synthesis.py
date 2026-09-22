import pytest

from research_assistant_cli.synthesis import synthesize_answer


class FakeMessages:
    def create(self, model, max_tokens, system, messages):
        text_block = type("T", (), {"text": "This is a synthesized answer.", "type": "text"})()
        return type("M", (), {"content": [text_block]})()


class FakeAnthropicClient:
    def __init__(self, api_key=None):
        self.messages = FakeMessages()


def test_synthesize_answer_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr(
        "research_assistant_cli.synthesis.anthropic.Anthropic", FakeAnthropicClient
    )

    chunks = [{"title": "A", "url": "https://a.com", "chunk_text": "some text"}]
    assert synthesize_answer("what is X?", chunks) == "This is a synthesized answer."


def test_synthesize_answer_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError):
        synthesize_answer("what is X?", [])