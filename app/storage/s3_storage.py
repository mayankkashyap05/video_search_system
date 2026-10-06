"""
Local-disk storage with the same interface the old S3/MinIO module had
(upload_file / download_file / generate_presigned_url), so no caller changes.
Files live under data/storage/<storage_key>.
"""
import shutil
from pathlib import Path
from app.config import STORAGE_DIR


def _resolve(storage_key: str) -> Path:
    path = (STORAGE_DIR / storage_key).resolve()
    if STORAGE_DIR.resolve() not in path.parents:   # block ../ traversal
        raise ValueError(f"Invalid storage key: {storage_key}")
    return path


def ensure_bucket():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def upload_file(local_path: Path, storage_key: str) -> str:
    dest = _resolve(storage_key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_path, dest)
    return storage_key


def download_file(storage_key: str, local_path: Path) -> Path:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_resolve(storage_key), local_path)
    return local_path


def generate_presigned_url(storage_key: str, expires_in: int = 3600) -> str:
    return str(_resolve(storage_key))
