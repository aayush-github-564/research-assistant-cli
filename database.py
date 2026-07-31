import sqlite3
from datetime import datetime, timezone

from models import SearchResult

DB_PATH = "research.db"


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.connection = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                search_id INTEGER NOT NULL,
                title TEXT,
                url TEXT NOT NULL,
                snippet TEXT,
                FOREIGN KEY (search_id) REFERENCES searches(id)
            )
        """)
        self.connection.commit()

    def save_search(self, query: str, results: list[SearchResult]) -> int:
        """Saves a search and its results as ONE transaction.
        Either everything lands, or nothing does."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO searches (query, created_at) VALUES (?, ?)",
                (query, datetime.now(timezone.utc).isoformat()),
            )
            search_id = cursor.lastrowid

            for r in results:
                cursor.execute(
                    "INSERT INTO results (search_id, title, url, snippet) VALUES (?, ?, ?, ?)",
                    (search_id, r.title, r.url, r.snippet),
                )

            self.connection.commit()
            return search_id

        except sqlite3.Error:
            self.connection.rollback()
            raise

    def get_recent_searches(self, limit: int = 5) -> list[dict]:
        """Returns the last N searches, most recent first."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT id, query, created_at FROM searches ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        return [{"id": r[0], "query": r[1], "created_at": r[2]} for r in rows]

    def get_results_for_search(self, search_id: int) -> list[SearchResult]:
        """Returns all results tied to a given search."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT title, url, snippet FROM results WHERE search_id = ?",
            (search_id,),
        )
        rows = cursor.fetchall()
        return [SearchResult(title=r[0], url=r[1], snippet=r[2]) for r in rows]

    def search_history(self, keyword: str) -> list[dict]:
        """Finds past searches whose query text matches a keyword."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT id, query, created_at FROM searches WHERE query LIKE ? ORDER BY created_at DESC",
            (f"%{keyword}%",),
        )
        rows = cursor.fetchall()
        return [{"id": r[0], "query": r[1], "created_at": r[2]} for r in rows]

    def close(self):
        self.connection.close()
