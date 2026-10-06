"""Tests for authenticated video playback: /videos/{id}/stream and /videos/{id}/playback-token."""
import jwt

from app.auth.users import create_playback_token, JWT_SECRET, JWT_ALGORITHM


def test_owner_can_stream_full_video(client, make_user, auth, upload_video):
    user = make_user()
    video_id, payload = upload_video(user)

    res = client.get(f"/videos/{video_id}/stream", headers=auth(user))
    assert res.status_code == 200
    assert res.content == payload
    assert res.headers["content-type"] == "video/mp4"
    assert res.headers["accept-ranges"] == "bytes"
    assert res.headers["content-length"] == str(len(payload))
    assert "no-store" in res.headers["cache-control"]


def test_unauthenticated_request_is_rejected(client, make_user, upload_video):
    user = make_user()
    video_id, _ = upload_video(user)

    assert client.get(f"/videos/{video_id}/stream").status_code == 401
    assert client.get(f"/videos/{video_id}/stream?token=not-a-token").status_code == 401
    assert client.get(f"/videos/{video_id}/playback-token").status_code == 401


def test_other_user_cannot_stream_video(client, make_user, auth, upload_video):
    owner = make_user()
    intruder = make_user()
    video_id, _ = upload_video(owner)

    res = client.get(f"/videos/{video_id}/stream", headers=auth(intruder))
    assert res.status_code == 404
    assert res.json()["detail"] == "Video not found"  # same as a missing video: no disclosure
    assert client.get(f"/videos/{video_id}/playback-token", headers=auth(intruder)).status_code == 404


def test_missing_video_returns_404(client, make_user, auth):
    user = make_user()
    res = client.get("/videos/does-not-exist/stream", headers=auth(user))
    assert res.status_code == 404
    assert res.json()["detail"] == "Video not found"


def test_range_request_returns_partial_content(client, make_user, auth, upload_video):
    user = make_user()
    video_id, payload = upload_video(user)

    res = client.get(f"/videos/{video_id}/stream", headers={**auth(user), "Range": "bytes=100-199"})
    assert res.status_code == 206
    assert res.content == payload[100:200]
    assert res.headers["content-range"] == f"bytes 100-199/{len(payload)}"
    assert res.headers["content-length"] == "100"

    # Open-ended range (what browsers send when seeking).
    res = client.get(f"/videos/{video_id}/stream", headers={**auth(user), "Range": "bytes=10000-"})
    assert res.status_code == 206
    assert res.content == payload[10000:]
    assert res.headers["content-range"] == f"bytes 10000-{len(payload) - 1}/{len(payload)}"

    # Suffix range.
    res = client.get(f"/videos/{video_id}/stream", headers={**auth(user), "Range": "bytes=-16"})
    assert res.status_code == 206
    assert res.content == payload[-16:]

    # End beyond file size is clamped, not rejected.
    res = client.get(f"/videos/{video_id}/stream", headers={**auth(user), "Range": "bytes=10200-999999"})
    assert res.status_code == 206
    assert res.content == payload[10200:]


def test_invalid_ranges(client, make_user, auth, upload_video):
    user = make_user()
    video_id, payload = upload_video(user)

    # Unsatisfiable: start past end of file.
    res = client.get(f"/videos/{video_id}/stream", headers={**auth(user), "Range": "bytes=999999-"})
    assert res.status_code == 416
    assert res.headers["content-range"] == f"bytes */{len(payload)}"

    # Start after end.
    res = client.get(f"/videos/{video_id}/stream", headers={**auth(user), "Range": "bytes=500-100"})
    assert res.status_code == 416

    # Malformed header is ignored -> full response.
    res = client.get(f"/videos/{video_id}/stream", headers={**auth(user), "Range": "garbage"})
    assert res.status_code == 200
    assert res.content == payload


def test_head_request_has_headers_without_body(client, make_user, auth, upload_video):
    user = make_user()
    video_id, payload = upload_video(user)
    res = client.head(f"/videos/{video_id}/stream", headers=auth(user))
    assert res.status_code == 200
    assert res.headers["content-length"] == str(len(payload))
    assert res.headers["accept-ranges"] == "bytes"
    assert res.content == b""


def test_content_type_follows_uploaded_format(client, make_user, auth, upload_video):
    user = make_user()
    cases = {"clip.webm": "video/webm", "talk.mkv": "video/x-matroska", "voice.m4a": "audio/mp4", "note.mp3": "audio/mpeg"}
    for filename, expected in cases.items():
        video_id, _ = upload_video(user, filename=filename, content=b"x" * 64)
        res = client.get(f"/videos/{video_id}/stream", headers=auth(user))
        assert res.status_code == 200
        assert res.headers["content-type"] == expected, filename


def test_playback_token_flow(client, make_user, auth, upload_video):
    owner = make_user()
    video_id, payload = upload_video(owner)

    res = client.get(f"/videos/{video_id}/playback-token", headers=auth(owner))
    assert res.status_code == 200
    body = res.json()
    assert body["video_id"] == video_id
    assert body["stream_url"].startswith(f"/videos/{video_id}/stream?token=")
    assert body["content_type"] == "video/mp4"

    # Token works without an Authorization header, including range requests.
    res = client.get(body["stream_url"], headers={"Range": "bytes=0-9"})
    assert res.status_code == 206
    assert res.content == payload[:10]

    # Token is bound to that video: cannot be reused for another owned video.
    other_video_id, _ = upload_video(owner)
    res = client.get(f"/videos/{other_video_id}/stream?token={body['token']}")
    assert res.status_code == 401

    # A playback token is not a session token.
    res = client.get("/videos", headers={"Authorization": "Bearer " + body["token"]})
    assert res.status_code == 401


def test_playback_token_for_other_users_video_is_rejected(client, make_user, upload_video):
    owner = make_user()
    intruder = make_user()
    video_id, _ = upload_video(owner)
    # Even a correctly signed playback token minted for another user doesn't grant access.
    forged = create_playback_token(intruder["user_id"], video_id)
    res = client.get(f"/videos/{video_id}/stream?token={forged}")
    assert res.status_code == 404


def test_expired_playback_token(client, make_user, upload_video):
    owner = make_user()
    video_id, _ = upload_video(owner)
    expired = jwt.encode(
        {"sub": owner["user_id"], "video_id": video_id, "scope": "playback", "exp": 0, "iat": 0},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )
    res = client.get(f"/videos/{video_id}/stream?token={expired}")
    assert res.status_code == 401


def test_url_ingest_pending_file_is_reported(client, make_user, auth):
    user = make_user()
    res = client.post("/videos/from-url?url=https://example.com/video", headers=auth(user))
    assert res.status_code == 200
    video_id = res.json()["video_id"]
    res = client.get(f"/videos/{video_id}/stream", headers=auth(user))
    assert res.status_code == 409


def test_search_still_works_and_exposes_video_id(client, make_user, auth, upload_video, fake_search_results):
    user = make_user()
    video_a, _ = upload_video(user, filename="lecture1.mp4")
    video_b, _ = upload_video(user, filename="sql.mp4")
    fake_search_results.extend([
        {"text": "DBMS is a database management system", "timestamp": 763.0, "type": "speech", "video_id": video_a, "owner_id": user["user_id"], "score": 0.9},
        {"text": "Database management...", "timestamp": 2530.0, "type": "speech", "video_id": video_b, "owner_id": user["user_id"], "score": 0.8},
    ])

    res = client.get("/search?q=DBMS&limit=5", headers=auth(user))
    assert res.status_code == 200
    body = res.json()
    assert body["query"] == "DBMS"
    assert [r["video_id"] for r in body["results"]] == [video_a, video_b]
    assert body["results"][0]["timestamp"] == 763.0
    assert body["results"][0]["type"] == "speech"

    res = client.get(f"/search?q=DBMS&video_id={video_a}", headers=auth(user))
    assert [r["video_id"] for r in res.json()["results"]] == [video_a]

    # Scoping to another user's video is still a 404.
    other = make_user()
    assert client.get(f"/search?q=DBMS&video_id={video_a}", headers=auth(other)).status_code == 404


def test_existing_endpoints_unchanged(client, make_user, auth, upload_video):
    user = make_user()
    video_id, _ = upload_video(user)
    assert client.get("/health").json() == {"status": "ok"}
    videos = client.get("/videos", headers=auth(user)).json()
    assert videos["count"] == 1 and videos["videos"][0]["video_id"] == video_id
    job_id = videos["videos"][0]["job_id"]
    job = client.get(f"/jobs/{job_id}", headers=auth(user)).json()
    assert job["video_id"] == video_id and job["status"] == "pending"
    chat = client.get(f"/chat?video_id={video_id}&question=hi", headers=auth(user)).json()
    assert chat["answer"] == "stub"
