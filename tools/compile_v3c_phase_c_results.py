#!/usr/bin/env python3
"""Compile one complete V3-C001 model without copying raw rollouts into Git."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from tools.vla_wam_v3_episode_schema import parse_jsonl_record


EXPERIMENT_ID = "V3-C001"
MODELS = {
    "groot_n17_droid_vla": "action_only_no_decodable_future",
    "cosmos3_edge_policy_droid": "decoded_future_required",
    "cosmos3_nano_policy_droid": "decoded_future_required",
}
PROMPT_FAMILIES = (
    "direct_command",
    "short_command",
    "goal_as_outcome",
    "desired_plus_negated_opposite",
)
RELATIONS = ("left", "right")
FAILURE_CLASSES = (
    "correct",
    "pick_failed",
    "transport_failed",
    "wrong_side",
    "release_failed",
)
SEEDS = tuple(range(8500, 8520))


class CompileError(ValueError):
    """Raised when retained evidence is incomplete or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise CompileError("invalid binomial count")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar(left_success: Iterable[bool], right_success: Iterable[bool]) -> dict[str, Any]:
    pairs = list(zip(left_success, right_success, strict=True))
    left_only = sum(left and not right for left, right in pairs)
    right_only = sum(right and not left for left, right in pairs)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(left_only, right_only)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "both_success": sum(left and right for left, right in pairs),
        "left_only": left_only,
        "right_only": right_only,
        "neither": sum(not left and not right for left, right in pairs),
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def _artifact(record: dict[str, Any], name: str) -> Path:
    value = record.get(name)
    if not isinstance(value, dict):
        raise CompileError(f"missing artifact record: {name}")
    path = Path(value.get("path", ""))
    if not path.is_file():
        raise CompileError(f"missing retained artifact: {path}")
    if path.stat().st_size != value.get("bytes") or sha256_file(path) != value.get("sha256"):
        raise CompileError(f"artifact integrity mismatch: {path}")
    return path


def _video_metadata(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - live RoboLab env supplies OpenCV
        raise CompileError("OpenCV is required to validate retained viewport videos") from error
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded, _ = capture.read()
    capture.release()
    if not opened or not decoded or frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise CompileError(f"viewport video is not decodable: {path}")
    return {"frame_count": frame_count, "width": width, "height": height, "fps": fps}


def _load_registration(path: Path, model_id: str) -> dict[int, list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("model_id") == model_id]
    if len(rows) != 160:
        raise CompileError(f"registration must contain 160 {model_id} cells")
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    if set(by_seed) != set(SEEDS):
        raise CompileError("registration seed set changed")
    for seed, seed_rows in by_seed.items():
        seed_rows.sort(key=lambda item: item["within_seed_execution_order"])
        conditions = {(row["prompt_family"], row["relation"]) for row in seed_rows}
        if len(seed_rows) != 8 or conditions != set((f, r) for f in PROMPT_FAMILIES for r in RELATIONS):
            raise CompileError(f"seed {seed} registration is not a complete eight-cell block")
    return dict(by_seed)


def _select_reports(gate_root: Path, model_id: str) -> tuple[list[Path], list[dict[str, Any]]]:
    selected: list[Path] = []
    provenance: list[dict[str, Any]] = []
    for seed in SEEDS:
        candidates = sorted(gate_root.glob(f"whole_seed*seed{seed}_attempt*.json"))
        valid: list[tuple[Path, dict[str, Any]]] = []
        for candidate in candidates:
            try:
                value = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if (
                value.get("passed") is True
                and value.get("experiment_id") == EXPERIMENT_ID
                and value.get("model_id") == model_id
                and value.get("seed") == seed
                and value.get("behavioral_episode_count") == 8
                and value.get("infrastructure_episode_count") == 0
                and len(value.get("cells", [])) == 8
            ):
                valid.append((candidate, value))
        if len(valid) != 1:
            raise CompileError(
                f"seed {seed} requires exactly one complete valid report; found {len(valid)} "
                f"among {[str(path) for path in candidates]}"
            )
        report_path, _ = valid[0]
        selected.append(report_path)
        provenance.append({
            "seed": seed,
            "path": str(report_path),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
            "all_candidates": [str(path) for path in candidates],
        })
    return selected, provenance


def _validate_future_trace(path: Path, seed: int) -> tuple[int, list[list[int]]]:
    trace = json.loads(path.read_text())
    requests = trace.get("requests")
    if not isinstance(requests, list) or len(requests) != trace.get("model_request_count"):
        raise CompileError(f"decoded future trace request count mismatch: {path}")
    shapes: list[list[int]] = []
    for request in requests:
        if request.get("requested_sampling_seed") != seed or request.get("server_sampling_seed") != seed:
            raise CompileError(f"server sampling-seed echo mismatch: {path}")
        action_path = Path(request.get("action_path", ""))
        future_path = Path(request.get("future_path", ""))
        for artifact_path, digest_name, shape_name in (
            (action_path, "action_sha256", "action_shape"),
            (future_path, "future_sha256", "future_shape"),
        ):
            if not artifact_path.is_file() or sha256_file(artifact_path) != request.get(digest_name):
                raise CompileError(f"request artifact integrity mismatch: {artifact_path}")
            array = np.load(artifact_path, allow_pickle=False, mmap_mode="r")
            if list(array.shape) != request.get(shape_name) or not np.isfinite(array).all():
                raise CompileError(f"request artifact shape/content mismatch: {artifact_path}")
        if request.get("action_shape") != [32, 8]:
            raise CompileError(f"Cosmos action chunk changed shape: {action_path}")
        shape = request.get("future_shape")
        if not isinstance(shape, list) or len(shape) != 4 or shape[0] != 33 or shape[-1] != 3:
            raise CompileError(f"Cosmos decoded future changed shape: {future_path}")
        shapes.append(shape)
    return len(requests), shapes


def _cell_row(
    report: dict[str, Any],
    cell: dict[str, Any],
    registered: dict[str, Any],
    *,
    future_required: bool,
    report_path: Path,
) -> dict[str, Any]:
    if cell.get("registered_cell_id") != registered.get("registered_cell_id"):
        raise CompileError("whole-seed execution order differs from the registration")
    if cell.get("prompt_family") != registered.get("prompt_family") or cell.get("relation") != registered.get("relation"):
        raise CompileError("cell condition differs from the registration")
    if cell.get("initial_state_sha256") != report.get("matched_initial_state_sha256"):
        raise CompileError("cell reset hash differs within matched seed")
    artifacts = cell.get("artifacts", {})
    actions_path = _artifact(artifacts, "executed_actions")
    state_path = _artifact(artifacts, "state_trace")
    episode_path = _artifact(artifacts, "behavioral_jsonl")
    video_path = _artifact(artifacts, "viewport_video")
    actions = np.load(actions_path, allow_pickle=False)
    if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] != cell.get("actions_executed"):
        raise CompileError(f"executed action shape mismatch: {actions_path}")
    if not np.isfinite(actions).all():
        raise CompileError(f"executed actions contain non-finite values: {actions_path}")
    if sum(1 for _ in state_path.open()) != actions.shape[0] + 1:
        raise CompileError(f"state trace must contain actions+1 rows: {state_path}")
    episode_lines = [line for line in episode_path.read_text().splitlines() if line.strip()]
    if len(episode_lines) != 1:
        raise CompileError(f"behavioral JSONL must contain one row: {episode_path}")
    raw = parse_jsonl_record(episode_lines[0])
    comparisons = {
        "registered_cell_id": cell.get("registered_cell_id"),
        "prompt": registered.get("prompt"),
        "prompt_family": cell.get("prompt_family"),
        "requested_relation": cell.get("relation"),
        "requested_success": cell.get("requested_success"),
        "failure_taxonomy": cell.get("failure_taxonomy"),
    }
    for key, expected in comparisons.items():
        if raw.get(key) != expected:
            raise CompileError(f"behavioral JSONL {key} mismatch: {episode_path}")
    if raw.get("measurements") != cell.get("measurements"):
        raise CompileError(f"behavioral JSONL measurement mismatch: {episode_path}")
    video = _video_metadata(video_path)
    future_count = 0
    future_shapes: list[list[int]] = []
    if future_required:
        future_trace_path = _artifact(artifacts, "decoded_future_trace")
        future_count, future_shapes = _validate_future_trace(future_trace_path, report["seed"])
    elif "decoded_future_trace" in artifacts:
        raise CompileError("action-only model unexpectedly claims decoded futures")
    return {
        "schema_version": "vla-wam-shared-v3c-compiled-episode-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": report["model_id"],
        "seed": report["seed"],
        "registered_cell_id": cell["registered_cell_id"],
        "within_seed_execution_order": cell["within_seed_execution_order"],
        "prompt_family": cell["prompt_family"],
        "prompt": registered["prompt"],
        "prompt_sha256": registered["prompt_sha256"],
        "relation": cell["relation"],
        "requested_success": cell["requested_success"],
        "failure_taxonomy": cell["failure_taxonomy"],
        "actions_executed": int(actions.shape[0]),
        "measurements": cell["measurements"],
        "initial_state_sha256": cell["initial_state_sha256"],
        "model_request_count": future_count if future_required else None,
        "decoded_future_shapes": sorted({tuple(shape) for shape in future_shapes}),
        "video": video,
        "artifacts": artifacts,
        "source_whole_seed_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
    }


def summarize(rows: list[dict[str, Any]], *, model_id: str) -> dict[str, Any]:
    if len(rows) != 160:
        raise CompileError("a completed model requires 160 valid behavioral rows")
    by_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[(row["prompt_family"], row["relation"])].append(row)
    condition_results: dict[str, Any] = {}
    paired_results: dict[str, Any] = {}
    for family in PROMPT_FAMILIES:
        pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for relation in RELATIONS:
            condition = sorted(by_condition[(family, relation)], key=lambda item: item["seed"])
            if len(condition) != 20:
                raise CompileError(f"{family}/{relation} requires 20 rows")
            successes = sum(row["requested_success"] for row in condition)
            condition_results[f"{family}:{relation}"] = {
                "successes": successes,
                "trials": 20,
                "success_rate": successes / 20.0,
                "wilson_95": wilson_interval(successes, 20),
                "failure_taxonomy": dict(Counter(row["failure_taxonomy"] for row in condition)),
            }
            for row in condition:
                pairs[str(row["seed"])][relation] = row
        if any(set(pair) != set(RELATIONS) for pair in pairs.values()) or len(pairs) != 20:
            raise CompileError(f"{family} LEFT/RIGHT pairing is incomplete")
        ordered = [pairs[str(seed)] for seed in SEEDS]
        shifts = [
            pair["right"]["measurements"]["signed_final_lateral_offset_m"]
            - pair["left"]["measurements"]["signed_final_lateral_offset_m"]
            for pair in ordered
        ]
        action_distinct = []
        for pair in ordered:
            left = np.load(pair["left"]["artifacts"]["executed_actions"]["path"], allow_pickle=False)
            right = np.load(pair["right"]["artifacts"]["executed_actions"]["path"], allow_pickle=False)
            common = min(len(left), len(right), 10)
            action_distinct.append(bool(common and np.any(left[:common] != right[:common])))
        mcnemar = exact_mcnemar(
            [pair["left"]["requested_success"] for pair in ordered],
            [pair["right"]["requested_success"] for pair in ordered],
        )
        paired_results[family] = {
            "matched_seed_count": 20,
            "success_discordance": mcnemar,
            "right_minus_left_endpoint_shift_m": shifts,
            "endpoint_ordering_aligned": sum(shift > 0 for shift in shifts),
            "endpoint_ordering_anti_aligned": sum(shift < 0 for shift in shifts),
            "endpoint_ordering_ties": sum(shift == 0 for shift in shifts),
            "median_right_minus_left_endpoint_shift_m": median(shifts),
            "first_10_executed_actions_distinct": sum(action_distinct),
        }
    direction_totals = {}
    for relation in RELATIONS:
        selected = [row for row in rows if row["relation"] == relation]
        successes = sum(row["requested_success"] for row in selected)
        direction_totals[relation] = {
            "successes": successes,
            "trials": len(selected),
            "descriptive_rate": successes / len(selected),
            "note": "Four repeated prompt forms per seed; not 80 independent scenes.",
        }
    return {
        "schema_version": "vla-wam-shared-v3c-model-summary-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "status": "complete_20_seed_160_behavioral_episode_result",
        "counts": {
            "matched_seeds": 20,
            "prompt_families": 4,
            "directions": 2,
            "valid_behavioral_episodes": 160,
            "valid_behavioral_failures": sum(not row["requested_success"] for row in rows),
            "infrastructure_episodes_in_denominator": 0,
            "viewport_videos": 160,
            "model_requests": sum(row["model_request_count"] or 0 for row in rows),
        },
        "success_by_condition": condition_results,
        "paired_diagnostics_by_prompt_family": paired_results,
        "direction_totals_descriptive_only": direction_totals,
        "failure_taxonomy_overall": {name: sum(row["failure_taxonomy"] == name for row in rows) for name in FAILURE_CLASSES},
        "interpretation_boundary": "Phase C is exploratory. Phrasing and direction cells share seeds; raw rates must not be treated as independent scenes, and DROID is never pooled with RoboTwin.",
    }


def compile_model(
    *, model_id: str, gate_root: Path, registration: Path, output_dir: Path
) -> dict[str, Any]:
    if model_id not in MODELS:
        raise CompileError(f"unsupported model: {model_id}")
    registered = _load_registration(registration, model_id)
    report_paths, report_provenance = _select_reports(gate_root, model_id)
    rows: list[dict[str, Any]] = []
    runtime_hashes: set[str] = set()
    release_hashes: set[str] = set()
    reset_hashes: set[str] = set()
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        seed = report["seed"]
        runtime_hashes.add(report["runtime_identity_sha256"])
        release_hashes.add(report["release_manifest_sha256"])
        reset_hashes.add(report["matched_initial_state_sha256"])
        report_cells = report["cells"]
        expected = registered[seed]
        if [cell["registered_cell_id"] for cell in report_cells] != [row["registered_cell_id"] for row in expected]:
            raise CompileError(f"seed {seed} execution order changed")
        for cell, registered_cell in zip(report_cells, expected, strict=True):
            rows.append(_cell_row(
                report,
                cell,
                registered_cell,
                future_required=MODELS[model_id] == "decoded_future_required",
                report_path=report_path,
            ))
    if len(runtime_hashes) != 1 or len(release_hashes) != 1 or len(reset_hashes) != 1:
        raise CompileError("runtime, release, or matched-reset identity changed across seeds")
    rows.sort(key=lambda row: (row["seed"], row["within_seed_execution_order"]))
    summary = summarize(rows, model_id=model_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes_path = output_dir / f"{model_id}_phase_c_episodes.jsonl"
    with episodes_path.open("x") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row).decode())
    summary_path = output_dir / f"{model_id}_phase_c_summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary))
    manifest = {
        "schema_version": "vla-wam-shared-v3c-model-evidence-manifest-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "status": summary["status"],
        "runtime_identity_sha256": next(iter(runtime_hashes)),
        "release_manifest_sha256": next(iter(release_hashes)),
        "matched_initial_state_sha256": next(iter(reset_hashes)),
        "registration": {
            "path": str(registration),
            "bytes": registration.stat().st_size,
            "sha256": sha256_file(registration),
        },
        "source_whole_seed_reports": report_provenance,
        "outputs": {
            episodes_path.name: {"bytes": episodes_path.stat().st_size, "sha256": sha256_file(episodes_path), "rows": 160},
            summary_path.name: {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        },
        "raw_retention": {
            "gate_root": str(gate_root),
            "raw_outputs_retained_on_pvc": True,
            "raw_outputs_committed_to_git": False,
        },
    }
    manifest_path = output_dir / f"{model_id}_phase_c_evidence_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=tuple(MODELS), required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = compile_model(**vars(args))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
