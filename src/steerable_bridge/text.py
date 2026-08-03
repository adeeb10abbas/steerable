from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import PurePosixPath
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
BRACKET_COORD_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
UNBRACKETED_COORD_RE = re.compile(
    r"\b(?:at|from|to|near|position(?:ed)?(?:\s+at)?|coordinate(?:s)?(?:\s+at)?)\s*"
    r"\(?\s*(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*\)?",
    re.IGNORECASE,
)
ANGLE_PLACEHOLDER_RE = re.compile(r"<[^>]+>")
BRACKET_LIKE_RE = re.compile(r"[\[\]]")


def normalize_text(value: str) -> str:
    """ASCII task-join key used by the independently reproduced join audit."""
    value = unicodedata.normalize("NFKC", value).casefold().replace("_", " ")
    return WHITESPACE_RE.sub(" ", NON_ALNUM_RE.sub(" ", value)).strip()


def normalize_semantic_text(value: str) -> str:
    """Unicode-preserving form normalization for commands and subtasks."""
    value = unicodedata.normalize("NFKC", value).casefold().replace("_", " ")
    normalized = "".join(character if character.isalnum() else " " for character in value)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:length]}"


def coordinate_pairs(text: str) -> list[tuple[float, float]]:
    matches = list(BRACKET_COORD_RE.finditer(text))
    spans = [match.span() for match in matches]
    pairs = [(float(match.group(1)), float(match.group(2))) for match in matches]
    for match in UNBRACKETED_COORD_RE.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        pairs.append((float(match.group(1)), float(match.group(2))))
    return pairs


def source_collection(path: str) -> str:
    normalized = path.replace("\\", "/").casefold()
    for source in ("bridge_data_v2", "bridge_data_v1", "rss", "icra", "flap"):
        if f"/{source}/" in normalized:
            return source
    return "unknown"


def task_family(path: str) -> str:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if "train" in parts:
        train_index = len(parts) - 1 - list(reversed(parts)).index("train")
        if train_index > 0:
            return parts[train_index - 1]
    return "unknown"


def unique_normalized_strings(values: Iterable[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = normalize_text(value)
        if normalized and normalized not in result:
            result[normalized] = value.strip()
    return result
