#!/usr/bin/env python3
"""Normalize failed pi0.5 V3-B002 launches outside behavioral denominators.

The live queue deliberately keeps operational ``attempt_events.jsonl`` and
``bridge_failure.json`` files distinct from the shared v3 episode schema.  This
tool converts one retained *raw attempt root* into one post-close manifested
``infrastructure_attempts.jsonl`` batch.  It never consumes behavioral rows.

Some early launches failed before a cell directory could be created.  In that
case the only evidence is one root queue log per lane.  Because a whole-seed
lane stops at its first failure, and this normalizer rejects any root that
contains a behavioral JSONL, the attempted cell is exactly the first released
cell in that frozen lane partition.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from experiments.v3.pi05_phase_b.contract import (
    AMENDMENT_ID,
    CHECKPOINT_MANIFEST_SHA256,
    MODEL_ID,
    OPENPI_COMMIT,
    OPENPI_CONFIG,
    ROBOLAB_COMMIT,
    RUNTIME_SCHEMA,
    STUDY_ID,
    AuthorizedCell,
    ContractError,
    ReleaseBundle,
    canonical_json_bytes,
    cells_for_lane,
    load_release_bundle,
    sha256_bytes,
    sha256_file,
)
from tools.vla_wam_v3_episode_schema import (
    INFRASTRUCTURE_SCHEMA_VERSION,
    MEASUREMENT_FRAME_DESCRIPTION,
    MEASUREMENT_FRAME_ID,
    validate_infrastructure_record,
    write_jsonl,
)


LANE_LOG_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:queue[^a-z0-9]*)?lane[^0-9]*(\d+)(?:[^0-9]|$)",
    re.I,
)
CELL_DIRECTORY_PREFIX = "v3b002__pi05__seed"
FAILURE_MARKERS = re.compile(
    r"traceback|error|exception|failed|failure|non[- ]?zero|importerror|fileexistserror",
    re.I,
)
HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVENTION_EVENTS = {
    "cooldown_started",
    "cooldown_completed",
    "emergency_hold",
    "temperature_query_failed",
    "worker_missing",
    "worker_exit_nonzero",
    "monitor_error",
}


class NormalizationError(RuntimeError):
    """Raised when retained failure evidence is absent or ambiguous."""


def _fail(message: str) -> None:
    raise NormalizationError(message)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NormalizationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise NormalizationError(f"cannot read {label} {path}: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        _fail(f"{label} must be non-empty JSONL without blank rows: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NormalizationError(f"cannot parse {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            _fail(f"{label} row is not an object: {path}:{number}")
        rows.append(value)
    return rows


def _file_record(path: Path) -> dict[str, Any]:
    value = Path(path).resolve()
    if not value.is_file() or value.stat().st_size <= 0:
        _fail(f"missing or empty retained failure evidence: {value}")
    return {"path": str(value), "sha256": sha256_file(value), "bytes": value.stat().st_size}


def _within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _validate_runtime_manifest(
    value: Mapping[str, Any], *, release: ReleaseBundle
) -> dict[str, Any]:
    """Validate a historical attempted runtime without requiring current HEAD.

    Attempts 01--03 intentionally bind older source commits.  Calling the live
    runtime validator would reject those truthful historical identities after
    a repair commit, so this validates their immutable self-hash and frozen
    scientific identity instead.
    """

    expected = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "openpi_commit": OPENPI_COMMIT,
        "robolab_commit": ROBOLAB_COMMIT,
        "openpi_config": OPENPI_CONFIG,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "release_manifest_sha256": release.manifest_sha256,
        "action_space": "joint_position_8d",
        "action_chunk_shape": [15, 8],
        "open_loop_horizon": 15,
        "action_cap": 450,
        "instruction_controller": "static_episode_prompt",
        "future_interface": "actions_only",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            _fail(f"attempt runtime does not match released {key}")
    for key in (
        "checkpoint_sha256",
        "environment_lock_sha256",
        "base_runtime_identity_sha256",
        "external_repository_diff_hash",
        "openpi_dir_status_sha256",
        "robolab_dir_status_sha256",
        "phase_b_adapter_contract_sha256",
        "live_topology_sha256",
    ):
        if not isinstance(value.get(key), str) or not HEX_64_RE.fullmatch(str(value[key])):
            _fail(f"attempt runtime lacks SHA-256 {key}")
    if not isinstance(value.get("study_git_commit"), str) or not HEX_40_RE.fullmatch(
        str(value["study_git_commit"])
    ):
        _fail("attempt runtime lacks a full study Git commit")
    topology = value.get("live_topology")
    if not isinstance(topology, dict):
        _fail("attempt runtime lacks live_topology")
    lanes = topology.get("simulator_lanes")
    if not isinstance(lanes, list) or not lanes:
        _fail("attempt runtime has no simulator lanes")
    lane_keys: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(lanes):
        if not isinstance(raw, dict) or raw.get("owner") != "ali":
            _fail(f"attempt runtime simulator lane {index} is not explicitly ali-owned")
        for key in ("pod", "pod_uid", "gpu_uuid"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                _fail(f"attempt runtime simulator lane {index} lacks {key}")
        lane_keys.add((raw["pod"], raw["pod_uid"], raw["gpu_uuid"]))
    if len(lane_keys) != len(lanes):
        _fail("attempt runtime simulator lanes are not unique")
    if value.get("live_topology_sha256") != sha256_bytes(canonical_json_bytes(topology)):
        _fail("attempt runtime live_topology self-hash changed")
    claimed = value.get("runtime_identity_sha256")
    body = {key: child for key, child in value.items() if key != "runtime_identity_sha256"}
    if not isinstance(claimed, str) or claimed != sha256_bytes(canonical_json_bytes(body)):
        _fail("attempt runtime identity self-hash changed")
    return dict(value)


def _lane_log_index(path: Path) -> int | None:
    if path.suffix.lower() not in {".log", ".out", ".err", ".txt"}:
        return None
    match = LANE_LOG_RE.search(path.name)
    return int(match.group(1)) if match else None


def _root_lane_logs(raw_root: Path, lane_count: int) -> dict[int, list[Path]]:
    logs: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(Path(raw_root).iterdir()):
        if not path.is_file():
            continue
        lane = _lane_log_index(path)
        if lane is None:
            continue
        if lane < 0 or lane >= lane_count:
            _fail(f"root queue log names an out-of-range lane {lane}: {path}")
        _file_record(path)
        logs[lane].append(path.resolve())
    return dict(logs)


def _cell_id_from_path(path: Path, raw_root: Path) -> str | None:
    current = Path(path).resolve().parent
    root = Path(raw_root).resolve()
    while _within(current, root) and current != root:
        if current.name.startswith(CELL_DIRECTORY_PREFIX):
            return current.name.replace("__", ":")
        current = current.parent
    return None


def _failure_from_events(path: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = _read_jsonl(path, "attempt events")
    failures = [
        row
        for row in rows
        if row.get("status") == "infrastructure_failed_excluded_from_denominator"
    ]
    if len(failures) != 1 or not isinstance(failures[0].get("error"), str):
        _fail(f"attempt events do not contain exactly one retained infrastructure failure: {path}")
    return str(failures[0]["error"]), rows


def _bridge_failure(path: Path) -> tuple[str, str]:
    row = _load_object(path, "bridge failure")
    cell_id = row.get("registered_cell_id")
    if not isinstance(cell_id, str):
        _fail(f"bridge failure lacks registered_cell_id: {path}")
    if row.get("denominator_eligible") is not False:
        _fail(f"bridge failure is not explicitly denominator-ineligible: {path}")
    error = row.get("error")
    if not isinstance(error, str) or not error.strip():
        _fail(f"bridge failure lacks error text: {path}")
    error_type = row.get("error_type")
    prefix = f"{error_type}: " if isinstance(error_type, str) and error_type else ""
    return cell_id, prefix + error


def _failure_stage(error: str) -> str:
    lowered = error.lower()
    for stage in (
        "queue_attempt_directory_creation_before_bridge",
        "bridge_module_import_before_model_request",
        "reset_stability_diagnostic_before_model_request",
    ):
        if stage in lowered:
            return stage
    if "fileexistserror" in lowered or "file exists" in lowered or "errno 17" in lowered:
        return "queue_attempt_directory_creation_before_bridge"
    if (
        "cannot import name 'empty'" in lowered
        or "cannot import name 'queue'" in lowered
        or "partially initialized module 'queue'" in lowered
        or "standard-library queue" in lowered
    ):
        return "bridge_module_import_before_model_request"
    if "cuda" in lowered and ("numpy" in lowered or "cpu" in lowered):
        return "reset_stability_diagnostic_before_model_request"
    return "pre_behavior_infrastructure_failure"


def _terminal_log_error(paths: Sequence[Path]) -> str:
    excerpts: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise NormalizationError(f"cannot read root queue log {path}: {exc}") from exc
        if not FAILURE_MARKERS.search(text):
            _fail(f"root queue log does not establish an infrastructure failure: {path}")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        failure_lines = [line for line in lines if FAILURE_MARKERS.search(line)]
        excerpt = "\n".join(lines[-12:] + failure_lines[-8:])
        detected_stage = _failure_stage(text)
        excerpts.append(
            f"{path.name}: {excerpt[-3800:]}\ndetected_stage={detected_stage}"
        )
    return "Retained root queue failure log(s):\n" + "\n".join(excerpts)


def _combined_evidence_hash(paths: Iterable[Path]) -> str:
    ordered = sorted({Path(path).resolve() for path in paths}, key=str)
    if not ordered:
        _fail("cannot hash an empty infrastructure evidence set")
    digest = hashlib.sha256()
    for path in ordered:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_intervention(paths: Iterable[Path]) -> bool:
    for path in paths:
        if path.name != "thermal_events.jsonl":
            continue
        for row in _read_jsonl(path, "thermal events"):
            if row.get("event") in INTERVENTION_EVENTS:
                return True
    return False


def _reset_id(attempt_dir: Path, cell: AuthorizedCell, runtime: Mapping[str, Any]) -> str:
    path = attempt_dir / "reset_attestation.json"
    if not path.is_file():
        return "not_observed_pre_behavior"
    value = _load_object(path, "reset attestation")
    for key, wanted in {
        "registered_cell_id": cell.cell_id,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }.items():
        if value.get(key) != wanted:
            _fail(f"reset attestation disagrees with {key}: {path}")
    claimed = value.get("reset_fingerprint_sha256")
    body = {key: child for key, child in value.items() if key != "reset_fingerprint_sha256"}
    if not isinstance(claimed, str) or claimed != sha256_bytes(canonical_json_bytes(body)):
        _fail(f"reset attestation self-hash changed: {path}")
    return claimed


def _attempted_cells(
    *, raw_root: Path, release: ReleaseBundle, lane_count: int
) -> list[dict[str, Any]]:
    root = Path(raw_root).resolve()
    if not root.is_dir():
        _fail(f"raw attempt root is not a directory: {root}")
    behavioral = sorted(root.rglob("raw_episode.jsonl"))
    if behavioral:
        _fail(f"raw infrastructure root contains behavioral JSONL evidence: {behavioral[0]}")

    root_logs = _root_lane_logs(root, lane_count)
    candidates: dict[str, dict[str, Any]] = {}
    evidence_paths = sorted(
        set(root.rglob("bridge_failure.json")) | set(root.rglob("attempt_events.jsonl"))
    )
    attempt_dirs = sorted({path.parent.resolve() for path in evidence_paths}, key=str)
    for attempt_dir in attempt_dirs:
        bridge_path = attempt_dir / "bridge_failure.json"
        events_path = attempt_dir / "attempt_events.jsonl"
        bridge_cell: str | None = None
        errors: list[str] = []
        sources: list[Path] = []
        if bridge_path.is_file():
            bridge_cell, error = _bridge_failure(bridge_path)
            errors.append(error)
            sources.append(bridge_path.resolve())
        if events_path.is_file():
            error, _ = _failure_from_events(events_path)
            errors.append(error)
            sources.append(events_path.resolve())
        path_cell = _cell_id_from_path(attempt_dir, root)
        cell_id = bridge_cell or path_cell
        if not isinstance(cell_id, str):
            _fail(f"cannot identify attempted cell from retained evidence: {attempt_dir}")
        if bridge_cell is not None and path_cell is not None and bridge_cell != path_cell:
            _fail(f"bridge/path cell identity mismatch: {attempt_dir}")
        if cell_id in candidates:
            _fail(f"duplicate failed attempt for one released cell: {cell_id}")
        try:
            cell = release.cell(cell_id)
        except ContractError as exc:
            raise NormalizationError(str(exc)) from exc
        lane = (cell.seed - min(item.seed for item in release.cells)) % lane_count
        candidates[cell_id] = {
            "cell": cell,
            "lane": lane,
            "attempt_dir": attempt_dir,
            "error": " | ".join(dict.fromkeys(errors)),
            "evidence": sources,
        }

    represented_lanes = {int(item["lane"]) for item in candidates.values()}
    lane_cells = {
        lane: sorted(
            cells_for_lane(release.cells, lane_index=lane, lane_count=lane_count),
            key=lambda cell: (cell.seed, cell.row["execution_order_index_within_seed"]),
        )
        for lane in range(lane_count)
    }
    for lane in range(lane_count):
        logs = root_logs.get(lane, [])
        if lane in represented_lanes:
            for candidate in candidates.values():
                if candidate["lane"] == lane:
                    candidate["evidence"].extend(logs)
                    if logs:
                        candidate["error"] = (
                            str(candidate["error"])
                            + " | "
                            + _terminal_log_error(logs)
                        )
            continue
        if not logs:
            _fail(f"lane {lane} has neither per-cell failure evidence nor a root queue log")
        first = lane_cells[lane][0]
        candidates[first.cell_id] = {
            "cell": first,
            "lane": lane,
            "attempt_dir": root,
            "error": _terminal_log_error(logs),
            "evidence": list(logs),
        }

    if {int(item["lane"]) for item in candidates.values()} != set(range(lane_count)):
        _fail("normalized evidence does not cover every attempted runtime lane")
    return sorted(
        candidates.values(),
        key=lambda item: (item["cell"].seed, item["cell"].row["execution_order_index_within_seed"]),
    )


def _artifacts(
    *, output_jsonl: Path, evidence: Sequence[Path], attempt_dir: Path
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "raw_result_jsonl": {
            "path": str(Path(output_jsonl).resolve()),
            "integrity_scope": "batch_manifest_after_close",
        }
    }
    named: list[tuple[str, Path]] = []
    for path in sorted(set(Path(value).resolve() for value in evidence), key=str):
        if path.name == "bridge_failure.json":
            key = "bridge_failure"
        elif path.name == "attempt_events.jsonl":
            key = "attempt_events"
        elif path.name == "thermal_events.jsonl":
            key = "thermal_events"
        else:
            key = "queue_log"
        named.append((key, path))
    counts: dict[str, int] = defaultdict(int)
    for base, path in named:
        index = counts[base]
        counts[base] += 1
        key = base if index == 0 else f"{base}_{index + 1}"
        artifacts[key] = _file_record(path)
    reset = Path(attempt_dir) / "reset_attestation.json"
    if reset.is_file():
        artifacts["reset_attestation"] = _file_record(reset)
    return artifacts


def normalize_attempt(
    *, repo_root: Path, release_manifest: Path, release_manifest_sha256: str,
    raw_attempt_root: Path, attempt_label: str, runtime_manifest: Path,
    output_jsonl: Path,
) -> dict[str, Any]:
    if not isinstance(attempt_label, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", attempt_label):
        _fail("attempt_label must contain only lowercase letters, digits, underscores, and hyphens")
    try:
        release = load_release_bundle(
            Path(repo_root), Path(release_manifest),
            expected_manifest_sha256=release_manifest_sha256,
        )
    except ContractError as exc:
        raise NormalizationError(str(exc)) from exc
    runtime_path = Path(runtime_manifest).resolve()
    runtime = _validate_runtime_manifest(
        _load_object(runtime_path, "runtime manifest"), release=release
    )
    lanes = runtime["live_topology"]["simulator_lanes"]
    attempted = _attempted_cells(
        raw_root=Path(raw_attempt_root), release=release, lane_count=len(lanes)
    )
    runtime_source = _file_record(runtime_path)
    records: list[dict[str, Any]] = []
    for item in attempted:
        cell: AuthorizedCell = item["cell"]
        evidence = sorted(set(item["evidence"]), key=str)
        if not evidence:
            _fail(f"attempted cell has no retained failure evidence: {cell.cell_id}")
        error = str(item["error"])
        if not error.strip():
            _fail(f"attempted cell has no error text: {cell.cell_id}")
        lane = int(item["lane"])
        lane_identity = lanes[lane]
        log_hash = _combined_evidence_hash(evidence)
        attempt_id = (
            f"{cell.cell_id}:{attempt_label}:lane{lane}:evidence{log_hash}:infrastructure"
        )
        record = {
            "schema_version": INFRASTRUCTURE_SCHEMA_VERSION,
            "record_type": "infrastructure_attempt",
            "behavioral_result_valid": False,
            "classification": "technical_invalid",
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "release_manifest_sha256": release.manifest_sha256,
            "release_fingerprint_sha256": release.release_fingerprint(cell),
            "registered_cell_id": cell.cell_id,
            "attempt_id": attempt_id,
            "raw_attempt_label": attempt_label,
            "model_id": MODEL_ID,
            "pair_id": cell.row["matched_block_id"],
            "arena": "droid_robolab",
            "environment_seed": cell.seed,
            "policy_seed": cell.seed,
            "requested_relation": cell.relation,
            "prompt": cell.row["prompt"],
            "prompt_family": cell.row["prompt_family"],
            "predicate_id": cell.row["success_predicate_id"],
            "reset_id": _reset_id(Path(item["attempt_dir"]), cell, runtime),
            "measurement_frame": MEASUREMENT_FRAME_ID,
            "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
            "checkpoint": {
                "id": OPENPI_CONFIG,
                "revision": f"v2a010-manifest-{CHECKPOINT_MANIFEST_SHA256}",
            },
            "runtime_identity": {
                "id": f"{MODEL_ID}:{runtime['runtime_identity_sha256'][:16]}",
                "sha256": runtime["runtime_identity_sha256"],
            },
            "runtime_manifest": runtime_source,
            "lane_identity": {
                key: lane_identity[key] for key in ("pod", "pod_uid", "gpu_uuid")
            },
            "artifacts": _artifacts(
                output_jsonl=Path(output_jsonl), evidence=evidence,
                attempt_dir=Path(item["attempt_dir"]),
            ),
            "stage": _failure_stage(error),
            "error": error,
            "log_hash": log_hash,
            "runtime_intervention": _runtime_intervention(evidence),
            "repair_attempt_id": None,
            "event_timeline": [
                {"sequence": 0, "stage": "released_cell_and_attempt_runtime_validated"},
                {"sequence": 1, "stage": "infrastructure_failure_evidence_retained"},
            ],
            "denominator_policy": "excluded_from_behavioral_denominator",
        }
        records.append(validate_infrastructure_record(record))
    output = Path(output_jsonl).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = write_jsonl(output, records)
    return {
        "output_jsonl": str(output),
        "output_manifest": str(output.with_name(output.name + ".manifest.json")),
        "attempt_count": len(records),
        "lane_count": len(lanes),
        "attempt_label": attempt_label,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "batch_manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--raw-attempt-root", type=Path, required=True)
    parser.add_argument("--attempt-label", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_attempt(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
