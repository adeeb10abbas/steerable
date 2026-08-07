#!/usr/bin/env python3
"""Evaluate retained fixed-observation responses for a Phase-C release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contract import (
    EXPERIMENT_ID,
    MODEL_CONTRACTS,
    PROMPT_FORMS,
    PROMPTS,
    SCHEMA_PREFIX,
    canonical_json_bytes,
    load_jsonl,
    prompt_sha256,
    sha256_file,
)


class GateError(ValueError):
    """Raised when fixed-observation evidence is incomplete or malformed."""


def _artifact_array(value: Any) -> np.ndarray | None:
    """Load a hash-bound numeric ``.npy`` artifact when one is referenced.

    Cosmos futures are tens of millions of pixels.  Requiring those arrays to
    be copied into JSONL made the prospective gate needlessly large.  The live
    collectors instead retain each array on the PVC and put its path, hash,
    shape, and dtype in the compact response row.  Inline nested lists remain
    supported for tests and small action-only probes.
    """

    if not isinstance(value, dict):
        return None
    required = {"path", "sha256", "shape", "dtype"}
    if set(value) < required:
        raise GateError("array artifact must name path, sha256, shape, and dtype")
    path = Path(value["path"])
    if not path.is_file():
        raise GateError(f"retained array is missing: {path}")
    digest_builder = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    if digest != value["sha256"]:
        raise GateError(f"retained array hash mismatch: {path}")
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception as error:  # pragma: no cover - numpy supplies details
        raise GateError(f"cannot load retained array {path}: {error}") from error
    if list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
        raise GateError(f"retained array metadata mismatch: {path}")
    if array.dtype.kind not in "fiu":
        raise GateError(f"retained array must contain finite numeric values: {path}")
    flat = array.reshape(-1)
    for start in range(0, flat.size, 1024 * 1024):
        if not np.isfinite(flat[start : start + 1024 * 1024]).all():
            raise GateError(f"retained array must contain finite numeric values: {path}")
    return array


def _numeric_array(value: Any) -> np.ndarray:
    artifact = _artifact_array(value)
    if artifact is not None:
        return artifact
    if isinstance(value, bool) or not isinstance(value, (list, tuple)):
        raise GateError("response arrays must be nested lists of finite numbers")
    try:
        output = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GateError("response arrays must contain finite numbers") from error
    if output.size == 0 or not np.isfinite(output).all():
        raise GateError("response arrays must not be empty")
    return output


def _mean_difference(left: Any, right: Any, *, squared: bool) -> float:
    lhs, rhs = _numeric_array(left), _numeric_array(right)
    if lhs.shape != rhs.shape:
        raise GateError("response array shapes differ")
    lhs_flat, rhs_flat = lhs.reshape(-1), rhs.reshape(-1)
    total = 0.0
    for start in range(0, lhs_flat.size, 1024 * 1024):
        first = lhs_flat[start : start + 1024 * 1024].astype(np.float64, copy=False)
        second = rhs_flat[start : start + 1024 * 1024].astype(np.float64, copy=False)
        delta = first - second
        total += float(np.square(delta).sum() if squared else np.abs(delta).sum())
    mean = total / lhs_flat.size
    return math.sqrt(mean) if squared else mean


def _rms(left: Any, right: Any) -> float:
    return _mean_difference(left, right, squared=True)


def _mae(left: Any, right: Any) -> float:
    return _mean_difference(left, right, squared=False)


def evaluate_records(records: Iterable[dict[str, Any]], *, model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_CONTRACTS:
        raise GateError(f"unregistered model_id: {model_id}")
    records = list(records)
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("model_id") != model_id:
            raise GateError("mixed or incorrect model_id in fixed-observation evidence")
        form, condition = record.get("prompt_family"), record.get("condition")
        if form not in PROMPT_FORMS or condition not in ("left", "left_exact_repeat", "right"):
            raise GateError("unknown prompt family or probe condition")
        if condition in groups.setdefault(form, {}):
            raise GateError(f"duplicate {form}/{condition} response")
        relation = "right" if condition == "right" else "left"
        prompt = PROMPTS[form][relation]
        if record.get("prompt") != prompt or record.get("prompt_sha256") != prompt_sha256(prompt):
            raise GateError(f"prompt bytes changed for {form}/{condition}")
        groups[form][condition] = record
    if set(groups) != set(PROMPT_FORMS) or any(
        set(group) != {"left", "left_exact_repeat", "right"} for group in groups.values()
    ):
        raise GateError("all four forms require left, exact-left-repeat, and right responses")
    form_results: dict[str, Any] = {}
    all_pass = True
    future_required = MODEL_CONTRACTS[model_id]["fixed_observation_future_required"]
    for form in PROMPT_FORMS:
        left, repeat, right = (groups[form][key] for key in ("left", "left_exact_repeat", "right"))
        observation_hashes = {item.get("observation_sha256") for item in (left, repeat, right)}
        sampling_seeds = {item.get("sampling_seed") for item in (left, repeat, right)}
        same_observation = len(observation_hashes) == 1 and None not in observation_hashes
        same_sampling_seed = len(sampling_seeds) == 1 and None not in sampling_seeds
        action_repeat_rms = _rms(left.get("actions"), repeat.get("actions"))
        action_prompt_rms = _rms(left.get("actions"), right.get("actions"))
        exact_action_repeat = action_repeat_rms == 0.0
        action_sensitive = action_prompt_rms > 0.0
        future_repeat_mae = None
        future_prompt_mae = None
        exact_future_repeat = True
        future_sensitive = True
        if future_required:
            future_repeat_mae = _mae(left.get("decoded_future"), repeat.get("decoded_future"))
            future_prompt_mae = _mae(left.get("decoded_future"), right.get("decoded_future"))
            exact_future_repeat = future_repeat_mae == 0.0
            future_sensitive = future_prompt_mae > 0.0
        passed = all(
            (same_observation, same_sampling_seed, exact_action_repeat, action_sensitive, exact_future_repeat, future_sensitive)
        )
        all_pass &= passed
        form_results[form] = {
            "passed": passed,
            "same_observation_sha256": same_observation,
            "same_sampling_seed": same_sampling_seed,
            "action_exact_repeat_rms": action_repeat_rms,
            "action_left_right_rms": action_prompt_rms,
            "future_required": future_required,
            "future_exact_repeat_mae": future_repeat_mae,
            "future_left_right_mae": future_prompt_mae,
        }
    return {
        "schema_version": f"{SCHEMA_PREFIX}-fixed-observation-gate-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "behavioral_episode_count": 0,
        "model_request_count": len(records),
        "prompt_forms": form_results,
        "exact_repeat_passed": all(result["action_exact_repeat_rms"] == 0.0 and (not result["future_required"] or result["future_exact_repeat_mae"] == 0.0) for result in form_results.values()),
        "prompt_only_sensitivity_passed": all(result["action_left_right_rms"] > 0.0 and (not result["future_required"] or result["future_left_right_mae"] > 0.0) for result in form_results.values()),
        "passed": all_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, choices=tuple(MODEL_CONTRACTS))
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_records(load_jsonl(args.responses), model_id=args.model_id)
    result["responses"] = {
        "path": str(args.responses),
        "bytes": args.responses.stat().st_size,
        "sha256": sha256_file(args.responses),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
