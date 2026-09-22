import threading
import time
from pathlib import Path

from research_assistant_cli.database import Database
from research_assistant_cli.providers import TavilySearchProvider
from dotenv import load_dotenv

load_dotenv()

db = Database(threading.Event(), db_path=Path("measure_temp.db"))
tavily = TavilySearchProvider(db)
query = "a query you haven't run before"

start = time.perf_counter()
tavily.search(query)
cold_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()
tavily.search(query)  # same query, within TTL — hits cache
warm_ms = (time.perf_counter() - start) * 1000

print(f"Cold: {cold_ms:.1f}ms  Warm: {warm_ms:.1f}ms")
print(f"Latency reduction: {(cold_ms - warm_ms) / cold_ms * 100:.1f}%")
db.close()