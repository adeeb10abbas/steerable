#!/usr/bin/env python3
"""Hash-close the split V3-E004 FastWAM RoboTwin cohort.

Seed 9400 was released by the four-cell smoke gate.  Seeds 9401--9426 run in
a second immutable output root.  This postprocessor joins those two sources
without copying or rerunning an episode, verifies every registered binding,
and keeps setup-invalid attempts outside the behavioral denominator.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_ID = "fastwam_robotwin"
ARENA = "robotwin"
SEEDS = tuple(range(9400, 9427))
LEVELS = (0.0, 1.0)
RELATIONS = ("left", "right")
FAILURES = {"pick_failed", "transport_failed", "wrong_side", "release_failed", "correct"}


class Invalid(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty evidence file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        require(isinstance(value, dict), f"non-object JSONL row: {path}:{number}")
        rows.append(value)
    return rows


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _validate_slice(
    root: Path,
    *,
    expected_seeds: Sequence[int],
    expected_status: str,
    expected_hashes: Mapping[str, str],
    queue_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    episodes_path = root / "behavioral_episodes.jsonl"
    progress_path = root / "queue_progress.jsonl"
    manifest = load_json(manifest_path)
    expected_count = 4 * len(expected_seeds)
    require(manifest.get("schema_version") == "vla-wam-shared-v3e004-fastwam-runtime-manifest-v1", f"wrong runtime manifest: {root}")
    require(manifest.get("status") == expected_status, f"wrong runtime status: {root}")
    require(manifest.get("model_id") == MODEL_ID and manifest.get("arena") == ARENA, f"wrong runtime identity: {root}")
    require(manifest.get("requested_seeds") == list(expected_seeds), f"runtime seed slice differs: {root}")
    require(manifest.get("whole_seeds_complete") == len(expected_seeds), f"whole-seed count differs: {root}")
    require(manifest.get("behavioral_episode_count") == expected_count, f"episode count differs: {root}")
    for name, expected in expected_hashes.items():
        require(manifest.get(name) == expected, f"{name} differs in {root}")

    rows = load_jsonl(episodes_path)
    require(len(rows) == expected_count, f"behavioral JSONL count differs: {root}")
    seed_records: list[dict[str, Any]] = []
    for seed in expected_seeds:
        marker_path = root / f"seed_{seed}_complete.json"
        marker = load_json(marker_path)
        require(marker.get("seed") == seed, f"seed marker identity differs: {marker_path}")
        require(marker.get("status") == "complete_four_valid_behavioral_cells", f"seed marker incomplete: {marker_path}")
        require(marker.get("behavioral_episode_count") == 4, f"seed marker count differs: {marker_path}")
        require(marker.get("infrastructure_failure_count") == 0, f"seed marker contains invalid cells: {marker_path}")
        compact_paths = [Path(path) for path in marker.get("compact_episode_paths", [])]
        require(len(compact_paths) == 4, f"seed marker lacks four compact rows: {marker_path}")
        seed_records.append({"seed": seed, "marker": record(marker_path), "compact_episode_paths": [str(path.resolve()) for path in compact_paths]})

    episode_records: list[dict[str, Any]] = []
    for row in rows:
        cell_id = row.get("cell_id")
        require(cell_id in queue_by_id, f"unregistered FastWAM cell: {cell_id}")
        queue = queue_by_id[cell_id]
        for name in ("model_id", "arena", "environment_seed", "sampling_seed", "relation", "prompt", "symmetry_level_s"):
            require(row.get(name) == queue.get(name), f"{cell_id}: {name} differs from queue")
        require(type(row.get("success")) is bool, f"{cell_id}: success is not boolean")
        require(row.get("failure_category") in FAILURES, f"{cell_id}: failure category differs")
        require((row["failure_category"] == "correct") == row["success"], f"{cell_id}: success/category mismatch")
        require(type(row.get("action_distinct")) is bool, f"{cell_id}: action distinction unavailable")

        source_path = Path(str(row.get("source_result_path")))
        require(row.get("source_result_sha256") == sha256(source_path), f"{cell_id}: source result hash differs")
        source = load_json(source_path)
        nested = source.get("v3e004", {})
        require(nested.get("cell_id") == cell_id, f"{cell_id}: nested cell differs")
        require(nested.get("registration_sha256") == expected_hashes["registration_sha256"], f"{cell_id}: registration binding differs")
        require(nested.get("queue_sha256") == expected_hashes["queue_sha256"], f"{cell_id}: queue binding differs")
        require(nested.get("candidate_sha256") == expected_hashes["candidate_sha256"], f"{cell_id}: candidate binding differs")
        require(nested.get("model_blind_gate_sha256") == expected_hashes["model_blind_gate_sha256"], f"{cell_id}: blind-gate binding differs")
        require(nested.get("model_specific_gate_sha256") == expected_hashes["model_specific_gate_sha256"], f"{cell_id}: model-gate binding differs")
        reset_hash = nested.get("initial_physical_fingerprint_sha256")
        require(isinstance(reset_hash, str) and len(reset_hash) == 64, f"{cell_id}: reset fingerprint unavailable")

        compact_path = source_path.with_name("e004_episode.json")
        require(load_json(compact_path) == row, f"{cell_id}: compact row differs from JSONL source")
        video_path = Path(str(row.get("simulator_video")))
        action_value = row.get("executed_action_trace")
        require(isinstance(action_value, Mapping), f"{cell_id}: action trace binding unavailable")
        action_path = Path(str(action_value.get("path")))
        action_record = record(action_path)
        require(action_value.get("sha256") == action_record["sha256"], f"{cell_id}: action trace hash differs")
        episode_records.append(
            {
                "cell_id": cell_id,
                "environment_seed": row["environment_seed"],
                "symmetry_level_s": row["symmetry_level_s"],
                "relation": row["relation"],
                "success": row["success"],
                "failure_category": row["failure_category"],
                "initial_physical_fingerprint_sha256": reset_hash,
                "compact_episode": record(compact_path),
                "source_result": record(source_path),
                "simulator_video": record(video_path),
                "executed_action_trace": action_record,
            }
        )

    for seed in expected_seeds:
        for level in LEVELS:
            pair = [item for item in episode_records if item["environment_seed"] == seed and item["symmetry_level_s"] == level]
            require({item["relation"] for item in pair} == set(RELATIONS), f"seed {seed} s={level}: pair incomplete")
            require(len({item["initial_physical_fingerprint_sha256"] for item in pair}) == 1, f"seed {seed} s={level}: LEFT/RIGHT reset differs")

    progress = load_jsonl(progress_path)
    require(progress and progress[-1].get("event") == "seed_complete", f"queue progress is not closed: {root}")
    slice_record = {
        "root": str(root),
        "requested_seeds": list(expected_seeds),
        "manifest": record(manifest_path),
        "behavioral_episodes": record(episodes_path),
        "queue_progress": record(progress_path),
        "seed_markers": seed_records,
    }
    return rows, slice_record, episode_records


def build(
    *,
    registration_path: Path,
    queue_path: Path,
    candidate_path: Path,
    model_blind_gate_path: Path,
    model_specific_gate_path: Path,
    smoke_root: Path,
    continuation_root: Path,
    smoke_runner_path: Path,
    continuation_runner_path: Path,
    setup_invalid_logs: Sequence[Path],
) -> dict[str, Any]:
    registration = load_json(registration_path)
    queue_rows = load_jsonl(queue_path)
    registration_sha = sha256(registration_path)
    queue_sha = sha256(queue_path)
    candidate_sha = sha256(candidate_path)
    blind_sha = sha256(model_blind_gate_path)
    specific_sha = sha256(model_specific_gate_path)
    require(registration.get("amendment_id") == "V3-E004", "wrong registration")
    require(registration.get("queue", {}).get("sha256") == queue_sha, "registration does not bind queue")
    require(registration.get("layout", {}).get("robotwin_stretch_candidate_sha256") == candidate_sha, "registration does not bind FastWAM candidate")
    expected_rows = [row for row in queue_rows if row.get("model_id") == MODEL_ID]
    require(len(expected_rows) == 108, "registered FastWAM queue is not 108 cells")
    queue_by_id = {row["cell_id"]: row for row in expected_rows}
    require(len(queue_by_id) == 108, "registered FastWAM queue has duplicate ids")

    blind = load_json(model_blind_gate_path)
    require(blind.get("passed") is True and blind.get("model_request_count") == 0 and blind.get("behavioral_episode_count") == 0, "model-blind gate did not pass at zero requests")
    require(blind.get("candidate_sha256") == candidate_sha, "model-blind gate candidate differs")
    specific = load_json(model_specific_gate_path)
    require(specific.get("status") == "passed_exact_repeat_and_left_right_prompt_sensitivity", "model-specific gate did not pass")
    require(specific.get("behavioral_episodes") == 0 and specific.get("model_action_requests") == 3, "model-specific gate counts differ")

    expected_hashes = {
        "registration_sha256": registration_sha,
        "queue_sha256": queue_sha,
        "candidate_sha256": candidate_sha,
        "model_blind_gate_sha256": blind_sha,
        "model_specific_gate_sha256": specific_sha,
        "checkpoint_sha256": "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63",
        "dataset_stats_sha256": "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095",
    }
    smoke_rows, smoke_slice, smoke_records = _validate_slice(
        smoke_root,
        expected_seeds=(9400,),
        expected_status="smoke_complete",
        expected_hashes=expected_hashes,
        queue_by_id=queue_by_id,
    )
    fresh_rows, fresh_slice, fresh_records = _validate_slice(
        continuation_root,
        expected_seeds=tuple(range(9401, 9427)),
        expected_status="requested_queue_slice_complete",
        expected_hashes=expected_hashes,
        queue_by_id=queue_by_id,
    )
    rows = smoke_rows + fresh_rows
    cell_ids = {row["cell_id"] for row in rows}
    require(len(rows) == len(cell_ids) == 108 and cell_ids == set(queue_by_id), "split sources do not exactly cover the registered FastWAM cohort")

    invalid_records: list[dict[str, Any]] = []
    required_invalid = {
        "smoke_seed9400_attempt01.log": "vla_wam_v2_protocol",
        "smoke_seed9400_attempt02.log": "modelscope",
    }
    require({Path(path).name for path in setup_invalid_logs} == set(required_invalid), "exactly the two registered setup-invalid smoke logs are required")
    for path in sorted(map(Path, setup_invalid_logs)):
        text = path.read_text(encoding="utf-8", errors="replace")
        require(required_invalid[path.name].lower() in text.lower(), f"setup-invalid reason differs: {path}")
        invalid_records.append(
            {
                "status": "setup_invalid_excluded_from_behavioral_denominator",
                "model_action_requests": 0,
                "behavioral_episodes": 0,
                "log": record(path),
            }
        )

    smoke_runner = record(smoke_runner_path)
    continuation_runner = record(continuation_runner_path)
    require(smoke_runner["sha256"] == continuation_runner["sha256"], "split cohort used different FastWAM runtime bytes")
    return {
        "schema_version": "vla-wam-shared-v3e004-fastwam-full-cohort-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "complete_hash_closed_108_registered_behavioral_cells",
        "created_at_utc": utc_now(),
        "behavioral_denominator": {"registered": 108, "valid": 108, "setup_invalid_excluded": 2},
        "registered_bindings": {
            "registration": record(registration_path),
            "queue": record(queue_path),
            "layout_candidate": record(candidate_path),
            "model_blind_gate": record(model_blind_gate_path),
            "model_specific_gate": record(model_specific_gate_path),
            **expected_hashes,
        },
        "runtime": {
            "fastwam_commit": "068d3fd70c89df3726c09893f47b75a624b20c02",
            "smoke_runner": smoke_runner,
            "continuation_runner": continuation_runner,
            "split_runner_bytes_identical": True,
        },
        "source_slices": [smoke_slice, fresh_slice],
        "episodes": sorted(smoke_records + fresh_records, key=lambda row: (row["environment_seed"], row["symmetry_level_s"], row["relation"])),
        "setup_invalid_attempts": invalid_records,
        "claim_boundary": {
            "robotwin_only_never_pooled_with_droid": True,
            "all_behavioral_failures_retained": True,
            "setup_invalid_attempts_excluded": True,
            "seed_9400_smoke_reused_without_rerun": True,
            "symmetric_object_layout_not_symmetric_robot_or_embodiment": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-blind-gate", type=Path, required=True)
    parser.add_argument("--model-specific-gate", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--smoke-runner", type=Path, required=True)
    parser.add_argument("--continuation-runner", type=Path, required=True)
    parser.add_argument("--setup-invalid-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build(
            registration_path=args.registration,
            queue_path=args.queue,
            candidate_path=args.candidate,
            model_blind_gate_path=args.model_blind_gate,
            model_specific_gate_path=args.model_specific_gate,
            smoke_root=args.smoke_root,
            continuation_root=args.continuation_root,
            smoke_runner_path=args.smoke_runner,
            continuation_runner_path=args.continuation_runner,
            setup_invalid_logs=args.setup_invalid_log,
        )
    except (Invalid, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    args.output.write_bytes(canonical_bytes(manifest))
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()), "sha256": sha256(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
