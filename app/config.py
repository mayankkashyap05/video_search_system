"""
Central configuration: shared paths and pipeline constants.
Everything (DB, storage, vector index, temp files) lives under data/ so the
project runs with no external services.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
PROCESSED_DIR = DATA_DIR / "processed"
STORAGE_DIR = DATA_DIR / "storage"      # replaces S3/MinIO
QDRANT_PATH = DATA_DIR / "qdrant"       # embedded Qdrant (replaces Qdrant server)
TEMP_DIR = DATA_DIR / "tmp"             # scratch space (replaces /tmp/...)

for d in (VIDEOS_DIR, PROCESSED_DIR, STORAGE_DIR, QDRANT_PATH, TEMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = "base"
KEYFRAME_INTERVAL_SECONDS = 10
