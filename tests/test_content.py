import pytest
import requests

from research_assistant_cli.content import chunk_text, fetch_page_text


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_fetch_page_text_success(monkeypatch):
    monkeypatch.setattr(
        "research_assistant_cli.content.requests.get",
        lambda url, timeout, headers: FakeResponse("<html>raw</html>"),
    )
    monkeypatch.setattr(
        "research_assistant_cli.content.trafilatura.extract",
        lambda html: "Extracted article text.",
    )

    assert fetch_page_text("https://example.com") == "Extracted article text."


def test_fetch_page_text_no_extractable_content_returns_none(monkeypatch):
    monkeypatch.setattr(
        "research_assistant_cli.content.requests.get",
        lambda url, timeout, headers: FakeResponse("<html></html>"),
    )
    monkeypatch.setattr(
        "research_assistant_cli.content.trafilatura.extract", lambda html: None
    )

    assert fetch_page_text("https://example.com") is None


def test_fetch_page_text_raises_after_retries_exhausted(monkeypatch):
    def raise_connection_error(url, timeout, headers):
        raise requests.exceptions.ConnectionError("simulated network failure")

    monkeypatch.setattr(
        "research_assistant_cli.content.requests.get", raise_connection_error
    )
    monkeypatch.setattr(
        "research_assistant_cli.resilience.time.sleep", lambda seconds: None
    )

    with pytest.raises(requests.exceptions.ConnectionError):
        fetch_page_text("https://example.com")


def test_chunk_text_overlap_between_consecutive_chunks():
    text = " ".join(f"word{i}" for i in range(250))
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) == 3
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_shorter_than_chunk_size_returns_one_chunk():
    text = "just a few words here"
    assert chunk_text(text, chunk_size=100, overlap=20) == [text]


def test_chunk_text_overlap_not_smaller_than_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=50)