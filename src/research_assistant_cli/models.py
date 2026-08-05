from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def __str__(self):
        return f"{self.title}\n{self.url}\n{self.snippet}\n"
