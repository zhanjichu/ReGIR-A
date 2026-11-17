"""Dataset loading and validation shared by both training stages."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any


def _image_path(value: Any, root: Path, path_mappings: dict[str, str] | None = None) -> str:
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Image entries must be non-empty file paths; got {type(value).__name__}.")
    for old, new in sorted((path_mappings or {}).items(), key=lambda pair: -len(pair[0])):
        if value == old or value.startswith(old.rstrip("/") + "/"):
            value = str(Path(new) / value[len(old):].lstrip("/"))
            break
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def read_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser()
    with data_path.open("r", encoding="utf-8") as stream:
        if data_path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset must contain a non-empty JSON array or JSON Lines records.")
    return records


def load_image_dataset(
    path: str | Path, max_samples: int | None = None,
    path_mappings: dict[str, str] | None = None,
):
    """Load a JSON/JSONL conversation dataset and decode its image paths lazily."""
    from datasets import Dataset, Image, Sequence

    data_path = Path(path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}. Update the train_file/validation_file "
            "setting in the YAML configuration."
        )
    records = read_records(data_path)
    dataset = Dataset.from_list(records)
    required = {"messages", "images"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset {data_path} is missing required columns: {sorted(missing)}")

    dataset = dataset.map(
        lambda row: {
            "images": [
                _image_path(image, data_path.parent, path_mappings)
                for image in (row["images"] if isinstance(row["images"], list) else [row["images"]])
            ]
        },
        desc="Resolving image paths",
    )
    dataset = dataset.cast_column("images", Sequence(Image()))

    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be a positive integer or null.")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if not len(dataset):
        raise ValueError(f"Dataset is empty: {data_path}")
    return dataset


def validate_records(path: str | Path, *, check_images: bool = False) -> dict[str, Any]:
    from collections import Counter
    from .decisions import parse_decision
    from .prompts import QUERY_REFLECT, QUERY_SELECT, SYSTEM_PROMPT, TOOLS

    records = read_records(path)
    counts = Counter()
    errors = []
    for index, record in enumerate(records):
        try:
            messages = normalize_messages(record["messages"])
            if [item["role"] for item in messages] != ["system", "user", "assistant"]:
                raise ValueError("Expected one system, user and assistant turn.")
            target = parse_decision(assistant_target(messages))
            if target is None:
                raise ValueError("Invalid assistant action.")
            images = record["images"]
            expected_count = 1 if target.kind == "select" else 2
            if not isinstance(images, list) or len(images) != expected_count:
                raise ValueError("Image count does not match Q1/Q2.")
            if messages[0]["content"].split() != SYSTEM_PROMPT.split():
                raise ValueError("System prompt differs from the inference prompt.")
            query = messages[1]["content"]
            if target.kind == "select" and (query != QUERY_SELECT or target.image_ref != "image"):
                raise ValueError("Q1 query or image argument differs from inference.")
            if target.kind != "select" and not any(
                query == QUERY_REFLECT.format(previous_tool=tool) for tool in TOOLS
            ):
                raise ValueError("Q2 must identify a valid previous tool.")
            if check_images:
                from PIL import Image
                for image in images:
                    with Image.open(_image_path(image, Path(path).resolve().parent)) as decoded:
                        decoded.verify()
            counts[target.kind] += 1
        except (KeyError, ValueError, TypeError, OSError) as exc:
            errors.append({"index": index, "error": str(exc)})
    return {"records": len(records), "actions": dict(counts), "errors": errors, "images_checked": check_images}


def content_to_text(content: Any) -> str:
    """Convert plain-text or typed chat content into readable text."""
    if isinstance(content, str):
        return re.sub(r"<image\s*/?>", "", content, flags=re.IGNORECASE).strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return re.sub(r"<image\s*/?>", "", "".join(parts), flags=re.IGNORECASE).strip()
    raise ValueError(f"Unsupported message content type: {type(content).__name__}")


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        raise ValueError("Each record must contain a messages list.")
    messages = []
    for message in raw_messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError("messages entries must use system, user, or assistant roles.")
        messages.append(
            {"role": message["role"], "content": content_to_text(message.get("content", ""))}
        )
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("Each record must contain at least one user message.")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("Each record must contain an assistant target.")
    return messages


def assistant_target(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"].strip()
    raise ValueError("Could not find an assistant target in the conversation.")
