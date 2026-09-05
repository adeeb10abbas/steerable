"""Fail-closed queue-row and episode-instruction binding for V4 DROID tasks."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.online_correction_v4.droid_task_files.constants import (
    ENV_QUEUE_ROW,
    ENV_QUEUE_ROW_SHA256,
    QUEUE_ROW_REQUIRED_KEYS,
    STUDY_ID,
)


class InstructionBindingError(ValueError):
    """Raised when the bound queue row or instruction digest is invalid."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_queue_row_bytes(row: Mapping[str, Any]) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class BoundEpisodeInstruction:
    episode_id: str
    fixture_id: str
    goal: str
    prompt_text: str
    prompt_sha256: str
    env_seed: int
    queue_row_sha256: str
    queue_row_path: str

    @property
    def instruction(self) -> dict[str, str]:
        return {"default": self.prompt_text}


def _fail(message: str) -> None:
    raise InstructionBindingError(message)


def load_bound_instruction(
    *,
    expected_fixture: str,
    expected_goal: str | None = None,
    queue_row_path: str | None = None,
    queue_row_sha256: str | None = None,
) -> BoundEpisodeInstruction:
    raw_path = queue_row_path or os.environ.get(ENV_QUEUE_ROW)
    expected_hash = queue_row_sha256 or os.environ.get(ENV_QUEUE_ROW_SHA256)
    if not raw_path or not expected_hash:
        _fail(f"{ENV_QUEUE_ROW} and {ENV_QUEUE_ROW_SHA256} are required")
    path = Path(raw_path).resolve()
    if not path.is_file():
        _fail(f"queue row path does not exist: {path}")
    digest = sha256_file(path)
    if digest != expected_hash:
        _fail("queue row digest mismatch")

    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InstructionBindingError(f"cannot read queue row: {exc}") from exc
    if not isinstance(row, dict):
        _fail("queue row must be a JSON object")
    for key in QUEUE_ROW_REQUIRED_KEYS:
        if key not in row:
            _fail(f"queue row missing required key: {key}")
    if row.get("campaign") != STUDY_ID:
        _fail("queue row campaign mismatch")
    fixture_id = row.get("fixture")
    if fixture_id != expected_fixture:
        _fail(f"queue row fixture {fixture_id!r} != expected {expected_fixture!r}")
    factors = row.get("factors")
    if not isinstance(factors, dict):
        _fail("queue row factors must be an object")
    goal = factors.get("goal")
    if not isinstance(goal, str) or not goal:
        _fail("queue row factors.goal is required")
    if expected_goal is not None and goal != expected_goal:
        _fail(f"queue row goal {goal!r} != expected task relation {expected_goal!r}")

    prompt_text = row.get("prompt_text")
    prompt_sha256 = row.get("prompt_sha256")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        _fail("queue row prompt_text is required")
    if not isinstance(prompt_sha256, str) or len(prompt_sha256) != 64:
        _fail("queue row prompt_sha256 must be a lowercase SHA-256 digest")
    if sha256_bytes(prompt_text.encode("utf-8")) != prompt_sha256:
        _fail("queue row prompt_sha256 does not match prompt_text bytes")

    env_seed = row.get("env_seed")
    if type(env_seed) is not int:
        _fail("queue row env_seed must be an integer")

    episode_id = row.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        _fail("queue row episode_id is required")

    return BoundEpisodeInstruction(
        episode_id=episode_id,
        fixture_id=str(fixture_id),
        goal=goal,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        env_seed=env_seed,
        queue_row_sha256=digest,
        queue_row_path=str(path),
    )
