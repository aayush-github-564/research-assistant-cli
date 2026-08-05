from models import SearchResult


def test_search_result_equality():
    result1 = SearchResult(
        title="Python Decorators Explained",
        url="https://example.com/decorators",
        snippet="A guide to decorators in Python.",
    )
    result2 = SearchResult(
        title="Python Decorators Explained",
        url="https://example.com/decorators",
        snippet="A guide to decorators in Python.",
    )

    assert result1 == result2


def test_search_result_str():
    result = SearchResult(
        title="Python Decorators Explained",
        url="https://example.com/decorators",
        snippet="A guide to decorators in Python.",
    )

    expected = (
        "Python Decorators Explained\n"
        "https://example.com/decorators\n"
        "A guide to decorators in Python.\n"
    )

    assert str(result) == expected
