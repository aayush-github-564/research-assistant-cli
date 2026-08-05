from pathlib import Path

from paths import get_data_dir, get_db_path


def test_windows_path_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setattr("paths.sys.platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))

    data_dir = get_data_dir()

    assert data_dir == tmp_path / "AppData" / "Roaming" / "research-assistant-cli"
    assert data_dir.exists()


def test_windows_path_falls_back_when_appdata_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("paths.sys.platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr("paths.Path.home", lambda: tmp_path)

    data_dir = get_data_dir()

    assert data_dir == tmp_path / "AppData" / "Roaming" / "research-assistant-cli"


def test_macos_path_uses_application_support(monkeypatch, tmp_path):
    monkeypatch.setattr("paths.sys.platform", "darwin")
    monkeypatch.setattr("paths.Path.home", lambda: tmp_path)

    data_dir = get_data_dir()

    assert (
        data_dir
        == tmp_path / "Library" / "Application Support" / "research-assistant-cli"
    )
    assert data_dir.exists()


def test_linux_path_uses_local_share(monkeypatch, tmp_path):
    monkeypatch.setattr("paths.sys.platform", "linux")
    monkeypatch.setattr("paths.Path.home", lambda: tmp_path)

    data_dir = get_data_dir()

    assert data_dir == tmp_path / ".local" / "share" / "research-assistant-cli"
    assert data_dir.exists()


def test_get_db_path_appends_filename(monkeypatch, tmp_path):
    monkeypatch.setattr("paths.sys.platform", "linux")
    monkeypatch.setattr("paths.Path.home", lambda: tmp_path)

    db_path = get_db_path()

    assert db_path.name == "research.db"
    assert db_path.parent == tmp_path / ".local" / "share" / "research-assistant-cli"
