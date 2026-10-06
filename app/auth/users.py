"""
User authentication: password hashing + JWT session tokens.
JWT_SECRET is optional: if not set in .env, a random one is generated once
and saved to data/.jwt_secret (so sessions survive restarts).
"""
import os
import secrets
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

from passlib.context import CryptContext
import jwt
from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from app.config import DATA_DIR
from app.db.database import get_db
from app.db.models import User


def _load_or_create_secret() -> str:
    secret_file = DATA_DIR / ".jwt_secret"
    if secret_file.exists():
        return secret_file.read_text().strip()
    secret = secrets.token_urlsafe(48)
    secret_file.write_text(secret)
    return secret


JWT_SECRET = os.environ.get("JWT_SECRET") or _load_or_create_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(raw_password: str) -> str:
    return pwd_context.hash(raw_password)


def verify_password(raw_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(raw_password, hashed_password)


def create_access_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired, please sign in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token")
    # Scoped tokens (e.g. video playback tokens) are never valid as a session.
    if payload.get("scope"):
        raise HTTPException(status_code=401, detail="Invalid session token")
    return payload["sub"]


def _resolve_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    user_id = decode_access_token(token)
    return _resolve_user(db, user_id)


# --- Playback tokens -------------------------------------------------------
# The browser's <video> element cannot send an Authorization header, so media
# is fetched with a short-lived token in the query string instead. That token
# is NOT the session JWT: it carries scope="playback" and is bound to a single
# video_id, so leaking it (browser history, logs) exposes at most one video
# the user already owns, for a limited time.

PLAYBACK_TOKEN_EXPIRE_SECONDS = 60 * 60
PLAYBACK_SCOPE = "playback"


def create_playback_token(user_id: str, video_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "video_id": video_id,
        "scope": PLAYBACK_SCOPE,
        "exp": now + timedelta(seconds=PLAYBACK_TOKEN_EXPIRE_SECONDS),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_playback_token(token: str, video_id: str) -> str:
    """Return the user_id if `token` is a valid playback token for `video_id`."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Playback link expired, please reopen the video")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid playback token")
    if payload.get("scope") != PLAYBACK_SCOPE or payload.get("video_id") != video_id:
        raise HTTPException(status_code=401, detail="Invalid playback token")
    return payload["sub"]


def get_playback_user(
    video_id: str,
    token: str | None = None,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> User:
    """
    Auth dependency for media streaming: accepts either the normal session
    bearer token (header) or a video-bound playback token (?token=...).
    """
    if authorization and authorization.startswith("Bearer "):
        return get_current_user(authorization=authorization, db=db)
    if token:
        return _resolve_user(db, decode_playback_token(token, video_id))
    raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
