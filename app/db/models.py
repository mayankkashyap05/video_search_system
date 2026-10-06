"""
SQLAlchemy models: users, videos, processing jobs, and (legacy) API keys.
IDs are plain string UUIDs so the same models work on SQLite and Postgres.
"""
import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Text, Enum, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=_uuid)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)
    videos = relationship("Video", back_populates="owner", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"
    id = Column(String(36), primary_key=True, default=_uuid)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    storage_key = Column(String, nullable=False)
    duration_seconds = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    owner = relationship("User", back_populates="videos")
    jobs = relationship("ProcessingJob", back_populates="video", cascade="all, delete-orphan")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    id = Column(String(36), primary_key=True, default=_uuid)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    current_stage = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    chapters_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    video = relationship("Video", back_populates="jobs")


class ApiKey(Base):  # legacy, unused
    __tablename__ = "api_keys"
    id = Column(String(36), primary_key=True, default=_uuid)
    key_hash = Column(String, nullable=False, unique=True)
    label = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    revoked = Column(String, default="false")
