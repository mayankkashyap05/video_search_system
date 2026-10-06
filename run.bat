@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe python -m venv .venv
call .venv\Scripts\activate
if not exist .venv\.installed (
    pip install -r requirements.txt && echo done> .venv\.installed
)
REM No --reload and a single process: embedded Qdrant locks its folder.
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
