#!/usr/bin/env python3
"""Evaluate retained fixed-observation responses for a Phase-C release gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

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


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, bool) or not isinstance(value, (list, tuple)):
        raise GateError("response arrays must be nested lists of finite numbers")
    output: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            output.extend(_flatten_numeric(item))
        elif isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise GateError("response arrays must contain finite numbers")
        else:
            output.append(float(item))
    if not output:
        raise GateError("response arrays must not be empty")
    return output


def _rms(left: Any, right: Any) -> float:
    lhs, rhs = _flatten_numeric(left), _flatten_numeric(right)
    if len(lhs) != len(rhs):
        raise GateError("response array shapes differ")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lhs, rhs)) / len(lhs))


def _mae(left: Any, right: Any) -> float:
    lhs, rhs = _flatten_numeric(left), _flatten_numeric(right)
    if len(lhs) != len(rhs):
        raise GateError("future array shapes differ")
    return sum(abs(a - b) for a, b in zip(lhs, rhs)) / len(lhs)


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

