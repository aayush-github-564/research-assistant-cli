from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import threading
import time

from models import SearchResult
from paths import get_db_path

DB_PATH = "research.db"


class Database:
    def __init__(self, shutdown_event: threading.Event, db_path: Path | None = None):
        # check_same_thread=False is required because search_multiple() runs
        # each provider's .search() in a separate worker thread (via
        # loop.run_in_executor), and every provider shares this one Database
        # instance for caching. Without this flag, sqlite3 raises a
        # ProgrammingError the moment a non-creator thread touches the
        # connection. The _lock below is what actually makes concurrent
        # access safe — check_same_thread=False only removes the hard crash,
        # it doesn't provide real thread-safety on its own.
        self.connection = sqlite3.connect(
            db_path or get_db_path(), check_same_thread=False
        )
        self.shutdown_event = shutdown_event
        self._lock = threading.Lock()
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                cache_key TEXT PRIMARY KEY,
                results_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self.connection.commit()

    def get_cached_result(
        self, cache_key: str, ttl_seconds: int
    ) -> list[SearchResult] | None:
        """Returns cached results if present and still within TTL, else None."""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT results_json, created_at FROM search_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            results_json, created_at = row
            if time.time() - created_at >= ttl_seconds:
                return None  # expired — treat as a miss

            raw_list = json.loads(results_json)
            return [SearchResult(**r) for r in raw_list]

    def set_cached_result(self, cache_key: str, results: list[SearchResult]) -> None:
        """Stores (or overwrites) a cache entry for the given key."""
        with self._lock:
            cursor = self.connection.cursor()
            results_json = json.dumps([r.__dict__ for r in results])
            cursor.execute(
                "INSERT OR REPLACE INTO search_cache (cache_key, results_json, created_at) VALUES (?, ?, ?)",
                (cache_key, results_json, time.time()),
            )
            self.connection.commit()

    def save_search(self, query: str, results: list[SearchResult]) -> int:
        """Saves a search and its results as ONE transaction.
        Either everything lands, or nothing does."""
        with self._lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute(
                    "INSERT INTO searches (query, created_at) VALUES (?, ?)",
                    (query, datetime.now(UTC).isoformat()),
                )
                search_id = cursor.lastrowid

                for r in results:
                    if self.shutdown_event.is_set():
                        self.connection.rollback()
                        print("Shutdown requested mid-save — rolled back cleanly.")
                        raise SystemExit(0)

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
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT id, query, created_at FROM searches ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [{"id": r[0], "query": r[1], "created_at": r[2]} for r in rows]

    def get_results_for_search(self, search_id: int) -> list[SearchResult]:
        """Returns all results tied to a given search."""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT title, url, snippet FROM results WHERE search_id = ?",
                (search_id,),
            )
            rows = cursor.fetchall()
            return [SearchResult(title=r[0], url=r[1], snippet=r[2]) for r in rows]

    def search_history(self, keyword: str) -> list[dict]:
        """Finds past searches whose query text matches a keyword."""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT id, query, created_at FROM searches WHERE query LIKE ? ORDER BY created_at DESC",
                (f"%{keyword}%",),
            )
            rows = cursor.fetchall()
            return [{"id": r[0], "query": r[1], "created_at": r[2]} for r in rows]

    def close(self):
        self.connection.close()
