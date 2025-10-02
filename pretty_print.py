"""Utility functions for producing human-readable logs."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List


def format_chatml(messages: Iterable[Dict[str, str]]) -> str:
    """Return a nicely formatted representation of ChatML-style messages."""

    lines: List[str] = []
    for message in messages:
        role = message.get("role", "unknown").upper()
        content = message.get("content", "").rstrip()
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines).strip()


def format_model_response(responses: List[Dict]) -> str:
    """Best-effort formatting of model responses."""

    if not responses:
        return "<no response>"
    parts = []
    for idx, response in enumerate(responses, start=1):
        text = response.get("text", "").strip()
        parts.append(f"Response #{idx}:\n{text}")
    return "\n\n".join(parts)


def format_meta_summary(summary: Dict) -> str:
    """Pretty format the meta-evaluation summary for logging."""

    return json.dumps(summary, indent=2, ensure_ascii=False)


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Return ``text`` truncated to at most ``max_chars`` characters."""

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
