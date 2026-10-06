"""
FastAPI application: upload videos, trigger processing, check status,
retrieve results, and search. Also serves the frontend at "/".

Run with a single process (embedded Qdrant locks its data folder):
    uvicorn app.api.main:app --host 127.0.0.1 --port 8000
"""
import re
import uuid
import mimetypes
from pathlib import Path
from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.config import TEMP_DIR, BASE_DIR
from app.core.video_chat import ask_about_video
from app.db.database import get_db, engine, Base
from app.db.models import Video, ProcessingJob, JobStatus, User
from app.storage.s3_storage import upload_file, get_file_size, iter_file_range
from app.workers.tasks import process_video_task, process_video_from_url_task
from app.core.qdrant_indexing import search as qdrant_search
from app.auth.users import (
    hash_password, verify_password, create_access_token, get_current_user,
    create_playback_token, get_playback_user, PLAYBACK_TOKEN_EXPIRE_SECONDS,
)
from app.core.live_sessions import start_session, get_session, stop_session, list_sessions

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Video Search API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only: restrict this to your real frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = TEMP_DIR / "uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/signup")
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with that email already exists")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id, "email": user.email}


@app.post("/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id, "email": user.email}


@app.post("/videos")
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video_id = str(uuid.uuid4())
    # Strip any directory parts from the client-supplied filename
    # (blocks "../" tricks and Windows path characters in storage keys).
    safe_name = Path(file.filename or "upload").name or "upload"
    local_path = TEMP_UPLOAD_DIR / f"{video_id}_{safe_name}"

    with open(local_path, "wb") as f:
        f.write(await file.read())

    storage_key = f"videos/{video_id}/{safe_name}"
    upload_file(local_path, storage_key)

    video = Video(id=video_id, owner_id=current_user.id, filename=safe_name, storage_key=storage_key)
    db.add(video)
    db.commit()

    job = ProcessingJob(id=str(uuid.uuid4()), video_id=video_id, status=JobStatus.PENDING)
    db.add(job)
    db.commit()

    process_video_task.delay(job.id, video_id, storage_key, safe_name)

    local_path.unlink(missing_ok=True)

    return {"video_id": video_id, "job_id": job.id, "status": job.status}


@app.post("/videos/from-url")
def upload_video_from_url(
    url: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video_id = str(uuid.uuid4())

    video = Video(id=video_id, owner_id=current_user.id, filename=url, storage_key="pending")
    db.add(video)
    db.commit()

    job = ProcessingJob(id=str(uuid.uuid4()), video_id=video_id, status=JobStatus.PENDING)
    db.add(job)
    db.commit()

    process_video_from_url_task.delay(job.id, video_id, url)

    return {"video_id": video_id, "job_id": job.id, "status": job.status}


def _get_owned_video_or_404(db: Session, video_id: str, current_user: User) -> Video:
    video = db.query(Video).filter(Video.id == video_id, Video.owner_id == current_user.id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


# --- Video playback ----------------------------------------------------------

# Browsers don't know every container by extension; make the common ones explicit.
_EXTRA_MIME_TYPES = {
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".m4a": "audio/mp4",
    ".ogv": "video/ogg",
    ".opus": "audio/ogg",
}
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def get_video_content_type(video: Video) -> str:
    # The stored key keeps the original extension (even for URL ingests,
    # where `filename` is the source URL), so derive the type from it.
    name = Path(video.storage_key).name if video.storage_key else video.filename
    suffix = Path(name).suffix.lower()
    if suffix in _EXTRA_MIME_TYPES:
        return _EXTRA_MIME_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


def _parse_range_header(range_header: str | None, file_size: int):
    """
    Return (start, end) inclusive byte offsets for a single-range request,
    None when the header is absent/malformed (serve the full file), or raise
    416 when the range is syntactically valid but unsatisfiable.
    """
    if not range_header:
        return None
    match = _RANGE_RE.match(range_header.strip())
    if not match:
        # Malformed Range headers must be ignored (RFC 9110 §14.2).
        return None
    start_s, end_s = match.groups()
    if not start_s and not end_s:
        return None
    if not start_s:
        # Suffix range: last N bytes.
        suffix = int(end_s)
        if suffix == 0:
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")
        start = max(file_size - suffix, 0)
        end = file_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
        if start >= file_size or (end_s and end < start):
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")
        end = min(end, file_size - 1)
    return start, end


def _get_video_file_size_or_error(video: Video) -> int:
    if not video.storage_key or video.storage_key == "pending":
        # URL ingests have no file until the download stage finishes.
        raise HTTPException(status_code=409, detail="This video is still being prepared")
    try:
        return get_file_size(video.storage_key)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Video file not found")


def stream_video(video: Video, request: Request) -> Response:
    file_size = _get_video_file_size_or_error(video)
    content_type = get_video_content_type(video)
    download_name = Path(video.storage_key).name
    headers = {
        "Accept-Ranges": "bytes",
        # Authenticated per-user media: never let shared caches keep it.
        "Cache-Control": "private, no-store",
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(download_name)}",
    }

    try:
        byte_range = _parse_range_header(request.headers.get("range"), file_size)
    except HTTPException as exc:
        if exc.status_code == 416:
            headers["Content-Range"] = f"bytes */{file_size}"
            return Response(status_code=416, headers=headers)
        raise

    if byte_range is None:
        start, end, status_code = 0, file_size - 1, 200
    else:
        start, end = byte_range
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    headers["Content-Length"] = str(end - start + 1 if file_size else 0)

    if request.method == "HEAD" or file_size == 0:
        return Response(status_code=status_code, headers=headers, media_type=content_type)

    return StreamingResponse(
        iter_file_range(video.storage_key, start, end),
        status_code=status_code,
        headers=headers,
        media_type=content_type,
    )


@app.get("/videos/{video_id}/playback-token")
def get_video_playback_token(
    video_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Short-lived, single-video token the <video> element can use (see get_playback_user)."""
    video = _get_owned_video_or_404(db, video_id, current_user)
    token = create_playback_token(current_user.id, video.id)
    return {
        "video_id": video.id,
        "token": token,
        "expires_in": PLAYBACK_TOKEN_EXPIRE_SECONDS,
        "stream_url": f"/videos/{video.id}/stream?token={token}",
        "content_type": get_video_content_type(video),
    }


@app.api_route("/videos/{video_id}/stream", methods=["GET", "HEAD"])
def stream_owned_video(
    video_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_playback_user),
):
    """Stream the original uploaded file with HTTP Range support (browser seeking)."""
    video = _get_owned_video_or_404(db, video_id, current_user)
    return stream_video(video, request)


@app.get("/jobs/{job_id}")
def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = (
        db.query(ProcessingJob)
        .join(Video, Video.id == ProcessingJob.video_id)
        .filter(ProcessingJob.id == job_id, Video.owner_id == current_user.id)
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "video_id": job.video_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "error_message": job.error_message,
        "summary": job.summary,
        "chapters": job.chapters_json,
    }


@app.get("/search")
def search_videos(
    q: str,
    limit: int = 5,
    video_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if video_id:
        _get_owned_video_or_404(db, video_id, current_user)
    results = qdrant_search(q, n_results=limit, video_id=video_id, owner_id=current_user.id)
    return {"query": q, "results": results}


@app.get("/chat")
def chat_about_video(
    video_id: str,
    question: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_video_or_404(db, video_id, current_user)
    result = ask_about_video(question, video_id, current_user.id)
    return result


@app.get("/videos")
def list_videos(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    videos = (
        db.query(Video)
        .filter(Video.owner_id == current_user.id)
        .order_by(Video.created_at.desc())
        .all()
    )

    results = []
    for video in videos:
        latest_job = (
            db.query(ProcessingJob)
            .filter(ProcessingJob.video_id == video.id)
            .order_by(ProcessingJob.created_at.desc())
            .first()
        )
        results.append({
            "video_id": video.id,
            "filename": video.filename,
            "created_at": video.created_at.isoformat() if video.created_at else None,
            "job_id": latest_job.id if latest_job else None,
            "status": latest_job.status if latest_job else None,
            "current_stage": latest_job.current_stage if latest_job else None,
            "summary": latest_job.summary if latest_job else None,
        })

    return {"videos": results, "count": len(results)}


class LiveStartRequest(BaseModel):
    source_url: str
    session_id: str | None = None
    language: str | None = None


@app.post("/live/start")
def start_live_session(
    payload: LiveStartRequest,
    current_user: User = Depends(get_current_user),
):
    session_id = payload.session_id or str(uuid.uuid4())
    try:
        start_session(
            session_id=session_id,
            source_url=payload.source_url,
            language=payload.language,
            owner_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"session_id": session_id, "status": "started"}


def _get_owned_session_or_404(session_id: str, current_user: User):
    session = get_session(session_id, owner_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Live session not found")
    return session


@app.get("/live")
def list_live_sessions(current_user: User = Depends(get_current_user)):
    return {"sessions": list_sessions(owner_id=current_user.id)}


@app.get("/live/{session_id}/transcript")
def get_live_transcript(session_id: str, current_user: User = Depends(get_current_user)):
    session = _get_owned_session_or_404(session_id, current_user)
    return {"session_id": session_id, "transcript": session.transcriber.get_transcript()}


@app.get("/live/{session_id}/search")
def search_live_session(
    session_id: str,
    q: str,
    limit: int = 5,
    current_user: User = Depends(get_current_user),
):
    # Ownership check first (raises 404 if this user doesn't own the session)
    _get_owned_session_or_404(session_id, current_user)
    # Live chunks are indexed into Qdrant under video_id=session_id (see
    # StreamingTranscriber.add_chunk -> index_video), so we can reuse the
    # same qdrant_search() the regular /search endpoint uses.
    results = qdrant_search(q, n_results=limit, video_id=session_id, owner_id=current_user.id)
    return {"session_id": session_id, "query": q, "results": results}


@app.post("/live/{session_id}/stop")
def stop_live_session(session_id: str, current_user: User = Depends(get_current_user)):
    stopped = stop_session(session_id, owner_id=current_user.id)
    if not stopped:
        raise HTTPException(status_code=404, detail="Live session not found")
    return {"session_id": session_id, "status": "stopped"}


# MUST stay last: a "/" mount catches everything, so it has to be registered
# after all API routes. Serves frontend/index.html at http://127.0.0.1:8000/
app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")
