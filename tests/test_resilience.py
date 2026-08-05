from resilience import retry_with_backoff
import pytest


def test_succeeds_on_first_try():
    call_count = 0

    @retry_with_backoff()
    def always_works():
        nonlocal call_count
        call_count += 1
        return "success"

    result = always_works()

    assert result == "success"
    assert call_count == 1


def test_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr("resilience.time.sleep", lambda seconds: None)

    call_count = 0

    @retry_with_backoff(max_attempts=3, base_delay=1)
    def fails_twice_then_succeeds():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("temporary failure")
        return "success"

    result = fails_twice_then_succeeds()

    assert result == "success"
    assert call_count == 3


def test_raises_after_max_attempts(monkeypatch):
    monkeypatch.setattr("resilience.time.sleep", lambda seconds: None)

    call_count = 0

    @retry_with_backoff(max_attempts=3, base_delay=1)
    def always_fails():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("permanent failure")

    with pytest.raises(ConnectionError):
        always_fails()

    assert call_count == 3


def test_does_not_retry_non_matching_exception(monkeypatch):
    monkeypatch.setattr("resilience.time.sleep", lambda seconds: None)

    call_count = 0

    @retry_with_backoff(max_attempts=3, base_delay=1, retry_on=(ConnectionError,))
    def raises_wrong_type():
        nonlocal call_count
        call_count += 1
        raise ValueError("not a retryable error")

    with pytest.raises(ValueError):
        raises_wrong_type()

    assert call_count == 1
