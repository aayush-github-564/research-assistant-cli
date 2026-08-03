import os
import sys
from pathlib import Path

APP_NAME = "research-assistant-cli"


def get_data_dir() -> Path:
    """Where persistent data (our SQLite DB) lives — OS-appropriate location."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share"

    data_dir = base / APP_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_db_path() -> Path:
    return get_data_dir() / "research.db"
