"""
Video Q&A chatbot: retrieval-augmented Q&A grounded in a specific video's
transcript and visual captions, answered by a LOCAL LLM (see app/core/llm.py).
"""
from app.core.qdrant_indexing import search
from app.core.llm import chat

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a specific video's content.
You will be given excerpts from the video's transcript and visual descriptions, each with a timestamp.
Answer the student's question using ONLY the provided excerpts. If the excerpts don't contain
enough information to answer, say so honestly rather than guessing.
Always cite the timestamp(s) your answer is based on, like this: (at 45s).
Keep answers concise and directly focused on what was asked.
"""


def _build_context(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        source = "Speech" if c["type"] == "speech" else "Visual"
        lines.append(f"[{c['timestamp']:.0f}s] {source}: {c['text']}")
    return "\n".join(lines)


def ask_about_video(question: str, video_id: str, owner_id: str, n_context_chunks: int = 6) -> dict:
    """Returns {"answer": str, "sources": [{"timestamp", "type", "text"}, ...]}."""
    chunks = search(question, n_results=n_context_chunks, video_id=video_id, owner_id=owner_id)
    if not chunks:
        return {
            "answer": "I couldn't find any indexed content for this video yet. "
                      "Make sure processing has completed.",
            "sources": [],
        }

    context = _build_context(chunks)
    try:
        answer_text = chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"VIDEO EXCERPTS:\n{context}\n\nQUESTION: {question}"},
            ],
            max_tokens=500,
        )
    except RuntimeError as e:
        # Local LLM not running: still return the retrieved moments so the
        # user gets something useful, plus a clear explanation.
        answer_text = f"(Local AI model unavailable: {e})\n\nMost relevant moments are listed below."

    return {
        "answer": answer_text,
        "sources": [
            {"timestamp": c["timestamp"], "type": c["type"], "text": c["text"]}
            for c in chunks
        ],
    }
