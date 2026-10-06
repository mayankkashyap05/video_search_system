"""
Local LLM client (no API key, no paid service).

Talks to any OpenAI-compatible endpoint. Default is Ollama running on your
own machine (https://ollama.com), which is free and fully offline:
    ollama pull llama3.2:3b
Override with LLM_BASE_URL / LLM_MODEL / LLM_API_KEY in .env if you ever
want a different server.
"""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

LLM_BASE_URL = (os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL") or "llama3.2:3b"
LLM_API_KEY = os.environ.get("LLM_API_KEY") or ""


def chat(messages: list[dict], max_tokens: int = 800, json_mode: bool = False, timeout: int = 300) -> str:
    """Send a chat request and return the reply text. Raises RuntimeError
    with a human-readable hint if the local LLM isn't reachable."""
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Can't reach the local LLM at {LLM_BASE_URL}. Install Ollama from https://ollama.com, "
            f"then run:  ollama pull {LLM_MODEL}"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Local LLM error {resp.status_code}: {resp.text[:300]} "
                           f"(if the model is missing, run: ollama pull {LLM_MODEL})")
    return resp.json()["choices"][0]["message"]["content"].strip()
