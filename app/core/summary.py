"""
Pipeline stage: Summary & chapter generation.

Merges transcript + visual captions into one chronological timeline and asks
a LOCAL LLM (Ollama, see app/core/llm.py) for a JSON summary + chapters.
If the LLM isn't available, falls back to a simple built-in extractive
summary so processing still completes with no extra setup.
"""
import json
from app.config import PROCESSED_DIR
from app.core.llm import chat

MAX_TIMELINE_CHARS = 8000  # keep within a small local model's context window

SUMMARY_PROMPT_TEMPLATE = """You are given a chronological timeline of a video, combining
speech transcript segments and visual scene descriptions, each with a timestamp in seconds.

Produce a JSON object with exactly this shape and nothing else (no markdown fences, no preamble):
{{
  "summary": "a 3-5 sentence overview of the whole video",
  "chapters": [
    {{"start": <seconds:int>, "title": "<short chapter title>"}}
  ]
}}

Guidelines:
- Chapters should mark meaningful topic/scene changes, not every timestamp.
- Aim for roughly 1 chapter per 1-3 minutes of content, fewer for short videos.
- Titles should be short (3-8 words) and specific enough to be searchable.

TIMELINE:
{timeline}
"""


def _build_timeline(transcript: list[dict], captions: list[dict]) -> str:
    events = []
    for seg in transcript:
        events.append((seg["start"], f"[{seg['start']:.0f}s] SPEECH: {seg['text']}"))
    for cap in captions:
        events.append((cap["timestamp"], f"[{cap['timestamp']}s] VISUAL: {cap['caption']}"))
    events.sort(key=lambda e: e[0])
    lines = [line for _, line in events]
    # Thin out evenly (keeping start-to-end coverage) if too long for the model.
    while sum(len(l) + 1 for l in lines) > MAX_TIMELINE_CHARS and len(lines) > 4:
        lines = lines[::2]
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    if not isinstance(data.get("summary"), str) or not isinstance(data.get("chapters"), list):
        raise ValueError("JSON missing summary/chapters")
    return data


def _fallback_summary(transcript: list[dict], captions: list[dict]) -> dict:
    """No-LLM fallback: first few spoken lines as the summary, one chapter
    per ~2 minutes titled with the first words spoken in that window."""
    texts = [s["text"] for s in transcript if s["text"].strip()]
    summary = " ".join(texts[:4])[:600] or (captions[0]["caption"] if captions else "No speech detected.")
    chapters, next_mark = [], 0
    for seg in transcript:
        if seg["start"] >= next_mark and seg["text"].strip():
            chapters.append({"start": int(seg["start"]), "title": " ".join(seg["text"].split()[:7])})
            next_mark = seg["start"] + 120
    return {"summary": summary, "chapters": chapters}


def generate_summary(transcript: list[dict], captions: list[dict], video_stem: str) -> dict:
    out_path = PROCESSED_DIR / video_stem / "summary.json"
    if out_path.exists():
        print(f"Using cached summary at {out_path}")
        return json.loads(out_path.read_text())

    prompt = SUMMARY_PROMPT_TEMPLATE.format(timeline=_build_timeline(transcript, captions))
    try:
        result = _parse_json(chat([{"role": "user", "content": prompt}], max_tokens=1200, json_mode=True))
    except Exception as e:
        print(f"  Local LLM summary unavailable ({e}); using built-in fallback summary.")
        result = _fallback_summary(transcript, captions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result
