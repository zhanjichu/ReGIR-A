"""Parsing and reward helpers for the agent's compact action language."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .prompts import TOOLS

_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.IGNORECASE | re.DOTALL)
_CALL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*$")


@dataclass(frozen=True)
class Decision:
    kind: str
    tool: str | None = None
    image_ref: str | None = None


def completion_text(value: Any) -> str:
    """Normalize TRL chat completions and plain text to a string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
        return "\n".join(parts).strip()
    return str(value).strip()


def parse_tool_call(text: str) -> tuple[str, str] | None:
    match = _CODE_RE.search(text or "")
    if not match:
        return None
    call = _CALL_RE.fullmatch(match.group(1))
    if not call:
        return None
    tool, image_ref = call.groups()
    if tool not in TOOLS or image_ref not in {"image", "image1", "image2"}:
        return None
    return tool, image_ref


def parse_decision(text: str) -> Decision | None:
    normalized = completion_text(text)
    lowered = normalized.lower()
    if re.fullmatch(r"\s*restored\s+successfully\.?\s*", lowered):
        return Decision("success")

    call = parse_tool_call(normalized)
    if call is None:
        return None

    tool, image_ref = call
    if lowered.startswith("error previous tool:"):
        kind = "error"
    elif lowered.startswith("continue to restore:"):
        kind = "continue"
    else:
        kind = "select"
    # Correction uses image1; continuation uses image2.
    if kind == "error" and image_ref != "image1":
        return None
    if kind == "continue" and image_ref != "image2":
        return None
    if kind == "select" and _CODE_RE.fullmatch(normalized) is None:
        return None
    if len(_CODE_RE.findall(normalized)) != 1:
        return None
    match = _CODE_RE.search(normalized)
    prefix = normalized[:match.start()].strip().lower()
    expected_prefix = {"error": "error previous tool:", "continue": "continue to restore:", "select": ""}[kind]
    if prefix != expected_prefix or normalized[match.end():].strip():
        return None
    return Decision(kind, tool, image_ref)
