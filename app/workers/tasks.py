"""
Video processing pipeline, run in a background thread pool inside the API
process (replaces Celery + Redis). Tasks keep a `.delay(...)` method so the
API code calling `process_video_task.delay(...)` is unchanged.

WORKER_THREADS (env, default 1): how many videos process at once. Keep at 1
unless you have lots of RAM -- Whisper/BLIP/pyannote are memory-hungry.
"""
import os
import json
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import TEMP_DIR
from app.db.database import SessionLocal
from app.db.models import ProcessingJob, JobStatus, Video
from app.storage.s3_storage import download_file, upload_file
from app.core.preprocessing import extract_audio, extract_keyframes
from app.core.transcription import transcribe
from app.core.vision import caption_keyframes
from app.core.summary import generate_summary
from app.core.qdrant_indexing import index_video
from app.core.diarization import diarize
from app.core.merge_speakers import merge_transcript_with_speakers
from app.core.url_download import download_video_from_url

PIPELINE_DIR = TEMP_DIR / "video_processing"

_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKER_THREADS", "1")))


def _log_failure(fut):
    exc = fut.exception()
    if exc:
        print("Background task failed:")
        traceback.print_exception(type(exc), exc, exc.__traceback__)


def background_task(fn):
    """Give fn a Celery-style .delay() that runs it on the thread pool."""
    def delay(*args, **kwargs):
        fut = _executor.submit(fn, *args, **kwargs)
        fut.add_done_callback(_log_failure)
        return fut
    fn.delay = delay
    return fn


def _update_job(db, job_id: str, **fields):
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if job:
        for key, value in fields.items():
            setattr(job, key, value)
        db.commit()


def _run_pipeline(db, job_id: str, video_id: str, local_video_path: Path):
    """Shared stages: preprocess -> transcribe -> diarize -> caption -> summarize -> index."""
    _update_job(db, job_id, current_stage="preprocessing")
    audio_path = extract_audio(local_video_path)
    keyframes = extract_keyframes(local_video_path)

    _update_job(db, job_id, current_stage="transcribing")
    transcript = transcribe(audio_path, video_id)

    # Speaker labels need pyannote + a free (but gated) Hugging Face token.
    # No token = skip this stage; everything else works without speaker labels.
    if os.environ.get("HF_TOKEN"):
        _update_job(db, job_id, current_stage="diarizing")
        diarize(audio_path, video_id)
        transcript = merge_transcript_with_speakers(video_id)

    _update_job(db, job_id, current_stage="captioning")
    captions = caption_keyframes(keyframes, video_id)

    _update_job(db, job_id, current_stage="summarizing")
    result = generate_summary(transcript, captions, video_id)

    _update_job(db, job_id, current_stage="indexing")
    owner_video = db.query(Video).filter(Video.id == video_id).first()
    index_video(transcript, captions, video_id, owner_video.owner_id)

    _update_job(
        db, job_id,
        status=JobStatus.COMPLETED,
        current_stage="done",
        summary=result["summary"],
        chapters_json=json.dumps(result["chapters"]),
    )


@background_task
def process_video_task(job_id: str, video_id: str, storage_key: str, filename: str):
    """Pipeline for videos uploaded as a file."""
    db = SessionLocal()
    local_dir = PIPELINE_DIR / video_id
    try:
        _update_job(db, job_id, status=JobStatus.PROCESSING, current_stage="downloading")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_video_path = local_dir / f"{video_id}{Path(filename).suffix}"
        download_file(storage_key, local_video_path)
        _run_pipeline(db, job_id, video_id, local_video_path)
    except Exception as e:
        _update_job(db, job_id, status=JobStatus.FAILED, error_message=str(e))
        raise
    finally:
        db.close()
        shutil.rmtree(local_dir, ignore_errors=True)


@background_task
def process_video_from_url_task(job_id: str, video_id: str, video_url: str):
    """Pipeline for videos submitted as a URL."""
    db = SessionLocal()
    local_dir = PIPELINE_DIR / video_id
    try:
        _update_job(db, job_id, status=JobStatus.PROCESSING, current_stage="downloading")
        local_video_path = download_video_from_url(video_url, local_dir, video_id)

        storage_key = f"videos/{video_id}/{local_video_path.name}"
        upload_file(local_video_path, storage_key)
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.storage_key = storage_key
            db.commit()

        _run_pipeline(db, job_id, video_id, local_video_path)
    except Exception as e:
        _update_job(db, job_id, status=JobStatus.FAILED, error_message=str(e))
        raise
    finally:
        db.close()
        shutil.rmtree(local_dir, ignore_errors=True)
