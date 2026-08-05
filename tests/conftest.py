import threading

import pytest

from research_assistant_cli.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(
        shutdown_event=threading.Event(),
        db_path=tmp_path / "test.db",
    )
    yield database
    database.close()
