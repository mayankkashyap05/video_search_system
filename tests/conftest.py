"""
API tests run against an isolated SQLite DB and a temporary storage folder.
The ML pipeline modules (Whisper/BLIP/Qdrant/LLM) are stubbed so the tests
exercise the HTTP layer only and don't need the heavy models downloaded.
"""
import os
import sys
import types
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="clipsearch-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["JWT_SECRET"] = "test-secret"


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _FakeTask:
    def __init__(self):
        self.calls = []

    def delay(self, *args, **kwargs):
        self.calls.append((args, kwargs))


FAKE_SEARCH_RESULTS = []


def _fake_search(query, n_results=5, video_id=None, owner_id=None):
    if not owner_id:
        raise ValueError("owner_id required")
    results = [r for r in FAKE_SEARCH_RESULTS if r.get("owner_id") == owner_id]
    if video_id:
        results = [r for r in results if r["video_id"] == video_id]
    return results[:n_results]


_stub("app.core.qdrant_indexing", search=_fake_search, index_video=lambda *a, **k: None)
_stub("app.core.video_chat", ask_about_video=lambda q, v, o: {"answer": "stub", "sources": []})
_stub(
    "app.workers.tasks",
    process_video_task=_FakeTask(),
    process_video_from_url_task=_FakeTask(),
)
_stub(
    "app.core.live_sessions",
    start_session=lambda **k: None,
    get_session=lambda *a, **k: None,
    stop_session=lambda *a, **k: False,
    list_sessions=lambda **k: [],
)

import app.storage.s3_storage as storage  # noqa: E402

storage.STORAGE_DIR = _TMP / "storage"
storage.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

from fastapi.testclient import TestClient  # noqa: E402
from app.api.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


_counter = {"n": 0}


@pytest.fixture
def make_user(client):
    def _make():
        _counter["n"] += 1
        email = f"user{_counter['n']}@example.com"
        res = client.post("/auth/signup", json={"email": email, "password": "password123"})
        assert res.status_code == 200, res.text
        body = res.json()
        return {"email": email, "token": body["access_token"], "user_id": body["user_id"]}
    return _make


@pytest.fixture
def auth():
    def _auth(user):
        return {"Authorization": "Bearer " + user["token"]}
    return _auth


@pytest.fixture
def upload_video(client, auth):
    """Upload fake media bytes through the real /videos endpoint."""
    def _upload(user, filename="lecture.mp4", content=None, content_type="video/mp4"):
        payload = content if content is not None else bytes(range(256)) * 40  # 10240 bytes
        res = client.post(
            "/videos",
            files={"file": (filename, payload, content_type)},
            headers=auth(user),
        )
        assert res.status_code == 200, res.text
        return res.json()["video_id"], payload
    return _upload


@pytest.fixture
def fake_search_results():
    FAKE_SEARCH_RESULTS.clear()
    yield FAKE_SEARCH_RESULTS
    FAKE_SEARCH_RESULTS.clear()
