"""Fail-closed recovery of completed V3-E004 DROID behavioral attempts.

This module is deliberately offline: it imports no simulator or model client.
It recovers only three explicitly allow-listed LEFT attempts whose behavior and
raw evidence were retained before a post-behavior output/compile fault.  Raw
inputs are inventoried before and after recovery and are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from typing import Any, Iterable, Mapping

import numpy as np

from .droid_behavioral_contract import simulator_export_envelope
from .episode_compiler import build_episode_record, frozen_requested_success, write_episode
from .runtime_contract import E004RuntimeBundle, RuntimeContractError, sha256_file


RECOVERY_SCHEMA = "vla-wam-shared-v3e004-offline-recovery-v1"
RECOVERY_BASE_COMMIT = "132686fdcede9efe40f7e575016e2bff924b18b6"
REGISTRATION_SHA256 = "3e3786510ebba933d0c7a4606afa51627ea6bf706f9e64b34cfd74b29e318db2"
QUEUE_SHA256 = "af6acdb298b773575a0ddc3d16be0544c7e2430f0d62d92fe231f75e1b2f7a57"
CANDIDATE_SHA256 = "7e270419cdbd2a00e36c47a6a5e3d10c7affb225f118e536cc43917da960ba33"


@dataclass(frozen=True)
class RecoverySpec:
    name: str
    cell_id: str
    model_id: str
    source_attempt: Path
    destination_attempt: Path
    source_code_commit: str
    fault_type: str
    fault_text: str
    action_metadata_name: str | None
    policy_output_suffix: str | None
    recovery_mode: str


def production_specs(root: Path = Path("/data/users/ali/vla_wam")) -> tuple[RecoverySpec, ...]:
    root = Path(root).resolve()
    old = root / "raw/v3e004/final-2742ec0/droid"
    new = root / "raw/v3e004/final-69998d6/droid"
    nano_source = (
        root
        / "raw/v3e004/servers/nano/vla-wam-rtx-nano/attempt04/sessions/final-69998d6/droid"
    )
    return (
        RecoverySpec(
            name="pi05_s100_left",
            cell_id="v3e004:pi05:seed9400:s100:left",
            model_id="pi05_current_stack_droid",
            source_attempt=old / "pi05_current_stack_droid/shard-004-of-763/cells/v3e004__pi05__seed9400__s100__left/attempt001",
            destination_attempt=new / "pi05_current_stack_droid/shard-004-of-763/cells/v3e004__pi05__seed9400__s100__left/attempt001",
            source_code_commit="2742ec0ad32a152652a9e5c9d0fcb7ebd1449e8e",
            fault_type="RuntimeContractError",
            fault_text="bridge completed without canonical simulator export",
            action_metadata_name="seed9400_left_action_trace.json",
            policy_output_suffix="pi05_v2a010_current",
            recovery_mode="reconstruct_missing_export",
        ),
        RecoverySpec(
            name="dreamzero_s100_left",
            cell_id="v3e004:dreamzero:seed9400:s100:left",
            model_id="dreamzero_droid_action_cfg",
            source_attempt=old / "dreamzero_droid_action_cfg/shard-001-of-054/cells/v3e004__dreamzero__seed9400__s100__left/attempt001",
            destination_attempt=new / "dreamzero_droid_action_cfg/shard-001-of-054/cells/v3e004__dreamzero__seed9400__s100__left/attempt001",
            source_code_commit="2742ec0ad32a152652a9e5c9d0fcb7ebd1449e8e",
            fault_type="RuntimeContractError",
            fault_text="bridge completed without canonical simulator export",
            action_metadata_name="seed9400_left_executed_actions.json",
            policy_output_suffix="dreamzero_v2",
            recovery_mode="reconstruct_missing_export",
        ),
        RecoverySpec(
            name="nano_s100_left",
            cell_id="v3e004:nano:seed9400:s100:left",
            model_id="cosmos3_nano_policy_droid",
            source_attempt=nano_source / "cosmos3_nano_policy_droid/shard-004-of-1123/cells/v3e004__nano__seed9400__s100__left/attempt001",
            destination_attempt=new / "cosmos3_nano_policy_droid/shard-004-of-1123/cells/v3e004__nano__seed9400__s100__left/attempt001",
            source_code_commit="69998d65b0027dfb6b1ea999edb0caa206c7b0c4",
            fault_type="NameError",
            fault_text="name 'requested' is not defined",
            action_metadata_name=None,
            policy_output_suffix=None,
            recovery_mode="recompile_existing_export",
        ),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeContractError(f"invalid finite JSON: {path}: {exc}") from exc


def _record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and path.stat().st_size > 0, f"missing or empty retained file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _inventory(root: Path) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    _require(root.is_dir() and not root.is_symlink(), f"retained attempt is not a directory: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"retained source contains a symlink: {path}")
        if path.is_file():
            records.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    _require(records, f"retained source attempt is empty: {root}")
    return records


def _validate_bound_file(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not a file record")
    record = _record(Path(str(value.get("path"))))
    _require(value.get("sha256") == record["sha256"], f"{label} digest changed")
    if value.get("bytes") is not None:
        _require(value.get("bytes") == record["bytes"], f"{label} byte count changed")
    return record


def _validate_embedded_records(value: Any, label: str = "$") -> list[dict[str, Any]]:
    """Validate every conventional ``path/sha256`` record in a JSON tree."""
    output: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            output.append({"json_path": label, **_validate_bound_file(value, label)})
        for key, child in value.items():
            output.extend(_validate_embedded_records(child, f"{label}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.extend(_validate_embedded_records(child, f"{label}[{index}]"))
    return output


def _validate_named_file(value: Mapping[str, Any], path_key: str, sha_key: str, label: str) -> dict[str, Any]:
    path = Path(str(value.get(path_key))).resolve()
    record = _record(path)
    _require(value.get(sha_key) == record["sha256"], f"{label} digest changed")
    return record


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(Path(repo).resolve()), *args], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeContractError(f"git identity check failed: {exc}") from exc


def _code_binding(repo: Path, spec: RecoverySpec) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    try:
        subprocess.check_call(
            ["git", "-C", str(Path(repo).resolve()), "merge-base", "--is-ancestor", RECOVERY_BASE_COMMIT, head],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeContractError(
            f"recovery must include compiler fix {RECOVERY_BASE_COMMIT}, got {head}"
        ) from exc
    _require(_git(repo, "cat-file", "-t", spec.source_code_commit) == "commit", "source execution commit is unavailable")
    module = Path(__file__).resolve()
    tool = Path(repo).resolve() / "tools/recover_v3e004_completed_attempts.py"
    return {
        "source_execution_commit": spec.source_code_commit,
        "source_execution_tree": _git(repo, "rev-parse", f"{spec.source_code_commit}^{{tree}}"),
        "source_bridge_blob": _git(repo, "rev-parse", f"{spec.source_code_commit}:experiments/v3/phase_e/symmetric_layout_cohort_v3e004/droid_behavioral_bridge.py"),
        "required_recovery_base_commit": RECOVERY_BASE_COMMIT,
        "recovery_commit": head,
        "recovery_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "recovery_module": _record(module),
        "recovery_cli": _record(tool),
    }


def _validate_identity(value: Mapping[str, Any], *, bundle: E004RuntimeBundle, spec: RecoverySpec, label: str) -> None:
    cell = bundle.cell(spec.cell_id)
    expected = {
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "registered_cell_id": spec.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "model_id": spec.model_id,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"{label} differs for {key}")


def _validate_attempt(bundle: E004RuntimeBundle, spec: RecoverySpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = spec.source_attempt
    attempt = _load_json(root / "attempt_manifest.json")
    infra = _load_json(root / "infrastructure_invalid.json")
    capture = _load_json(root / "state_capture/state_capture.json")
    for value, label in ((attempt, "attempt manifest"), (infra, "infrastructure record"), (capture, "state capture")):
        _require(isinstance(value, dict), f"{label} is not an object")
        _validate_identity(value, bundle=bundle, spec=spec, label=label)
    _require(attempt.get("schema_version") == "vla-wam-shared-v3e004-droid-attempt-manifest-v1", "attempt schema changed")
    for key, wanted in (("registration_sha256", REGISTRATION_SHA256), ("queue_sha256", QUEUE_SHA256), ("candidate_sha256", CANDIDATE_SHA256)):
        _require(attempt.get(key) == wanted, f"attempt differs for {key}")
    argv = attempt.get("invocation_argv")
    _require(isinstance(argv, list) and "--expected-study-commit" in argv, "attempt lacks source commit argument")
    index = argv.index("--expected-study-commit")
    _require(index + 1 < len(argv) and argv[index + 1] == spec.source_code_commit, "attempt source commit changed")
    _require(infra.get("schema_version") == "vla-wam-shared-v3e004-infrastructure-attempt-v1", "infrastructure schema changed")
    _require(infra.get("behavioral_result_valid") is False and infra.get("denominator_eligible") is False, "original attempt was not excluded")
    _require(infra.get("stage") == "bridge_or_compile", "failure did not occur after behavior")
    _require(infra.get("error_type") == spec.fault_type and infra.get("error") == spec.fault_text, "attempt has a non-allow-listed failure")
    _validate_bound_file(infra.get("attempt_manifest", {}), "infrastructure attempt-manifest binding")
    _require(capture.get("schema_version") == "vla-wam-shared-v3e004-droid-state-capture-v1", "state capture schema changed")
    cell = bundle.cell(spec.cell_id)
    for key, wanted in (("environment_seed", cell.environment_seed), ("sampling_seed", cell.sampling_seed), ("requested_relation", cell.relation), ("prompt", cell.row["prompt"]), ("action_cap", cell.row["runtime_identity_requirement"]["action_cap"])):
        _require(capture.get(key) == wanted, f"state capture differs for {key}")
    steps = capture.get("steps")
    _require(isinstance(steps, list) and len(steps) >= 2, "state capture has no complete step stream")
    _require(all(isinstance(row, dict) and row.get("action_step") == index for index, row in enumerate(steps)), "state step stream is not contiguous")
    actions = len(steps) - 1
    _require(capture.get("actions_executed") == actions and actions > 0, "captured action count changed")
    partial = root / "state_capture/states.partial.jsonl"
    if partial.is_file():
        partial_rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
        _require(partial_rows == steps, "immutable partial step stream differs from closed state capture")
    detached = bool(steps[-1].get("grippers_open") is True and steps[-1].get("object_grabbed") is False)
    _require(capture.get("final_detached_release") is detached, "detached-release flag differs from final raw state")
    success = frozen_requested_success(steps, cell.relation, detached)
    _require(capture.get("requested_success") is success, "stored score differs from recomputed frozen B001 predicate")
    right_censored = bool(not success and actions == int(capture["action_cap"]))
    _require(capture.get("right_censored") is right_censored, "right-censor flag differs from immutable step count")
    return attempt, infra, capture


def _probe_mp4(path: Path) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - production uses the pinned RoboLab environment
        raise RuntimeContractError("OpenCV is required to decode-probe retained viewport evidence") from exc
    capture = cv2.VideoCapture(str(path))
    okay, frame = capture.read()
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    _require(okay and frame is not None and frames > 0 and fps > 0 and width > 0 and height > 0, "retained viewport MP4 does not decode")
    return {"frame_count": frames, "fps": fps, "width": width, "height": height, "first_frame_decoded": True}


def _find_native_output(spec: RecoverySpec, attempt: Mapping[str, Any], capture: Mapping[str, Any], video_root: Path, *, probe_video: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _require(spec.policy_output_suffix is not None, "native-output lookup is unavailable for this recovery mode")
    task = "V3E004DroidLeftTask"
    filename = "Put_the_Rubiks_cube_to_the_left_of_the_bowl_0_viewport.mp4"
    created = float(attempt["created_unix_s"])
    candidates: list[Path] = []
    for path in Path(video_root).resolve().glob(f"*_{spec.policy_output_suffix}/{task}/{filename}"):
        if path.is_file() and created <= path.stat().st_mtime <= (spec.source_attempt / "infrastructure_invalid.json").stat().st_mtime + 2.0:
            candidates.append(path.resolve())
    _require(len(candidates) == 1, f"expected one unambiguous retained viewport for {spec.name}, found {candidates}")
    video = candidates[0]
    native_root = video.parents[1]
    results_path = native_root / "episode_results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(len(rows) == 1, "native RoboLab output does not contain one completed episode")
    result = rows[0]
    _require(result.get("policy") == spec.policy_output_suffix and result.get("instruction") == bundle_prompt(spec.cell_id), "native output identity differs")
    _require(result.get("episode_step") == capture.get("actions_executed"), "native episode length differs from captured steps")
    files = [_record(path) for path in sorted(native_root.rglob("*")) if path.is_file()]
    probe = _probe_mp4(video) if probe_video else {"test_probe_bypassed": True}
    return _record(video), files, probe


_PROMPTS = {
    "v3e004:pi05:seed9400:s100:left": "Put the Rubik's cube to the left of the bowl.",
    "v3e004:dreamzero:seed9400:s100:left": "Put the Rubik's cube to the left of the bowl.",
    "v3e004:nano:seed9400:s100:left": "Put the Rubik's cube to the left of the bowl.",
}


def bundle_prompt(cell_id: str) -> str:
    return _PROMPTS[cell_id]


def _reconstruct_export(bundle: E004RuntimeBundle, spec: RecoverySpec, attempt: Mapping[str, Any], capture: Mapping[str, Any], video_root: Path, *, probe_video: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    root = spec.source_attempt
    cell = bundle.cell(spec.cell_id)
    metadata_path = root / f"action_traces/{spec.action_metadata_name}"
    metadata = _load_json(metadata_path)
    _require(isinstance(metadata, dict), "action metadata is not an object")
    embedded = _validate_embedded_records(metadata, "action_metadata")
    executed = _validate_bound_file(metadata.get("executed_actions", {}), "executed actions")
    actions = np.load(executed["path"], allow_pickle=False)
    _require(actions.ndim == 2 and actions.shape == (capture["actions_executed"], 8) and np.isfinite(actions).all(), "executed action array differs from the captured trajectory")
    runtime_path = root / "runtime_identity.json"
    gate_path = root / "live_scene_gate.json"
    runtime = _load_json(runtime_path)
    gate = _load_json(gate_path)
    _validate_embedded_records(runtime, "runtime_identity")
    _validate_embedded_records(gate, "live_scene_gate")
    video, native_files, video_probe = _find_native_output(spec, attempt, capture, video_root, probe_video=probe_video)
    future: Any = None
    future_status = "not_exposed_by_action_only_interface"
    if spec.model_id == "dreamzero_droid_action_cfg":
        manifest_record = metadata.get("future_manifest")
        _validate_bound_file(manifest_record, "DreamZero future manifest")
        future_manifest = _load_json(Path(str(manifest_record["path"])))
        _require(future_manifest.get("request_count") == capture.get("model_request_count"), "DreamZero future request count changed")
        _require(len(future_manifest.get("official_reset_decode", [])) == 1, "DreamZero retained future decode is ambiguous")
        embedded.extend(_validate_embedded_records(future_manifest, "dreamzero_future_manifest"))
        future = {"action_future_trace": _record(metadata_path), "future_manifest": dict(manifest_record)}
        future_status = "native_latent_and_official_decoded_future_retained"
    detached = bool(capture["steps"][-1]["grippers_open"] and not capture["steps"][-1]["object_grabbed"])
    success = frozen_requested_success(capture["steps"], cell.relation, detached)
    export = simulator_export_envelope(
        cell=cell,
        bundle=bundle,
        steps=capture["steps"],
        requested_success=success,
        right_censored=bool(not success and capture["actions_executed"] == capture["action_cap"]),
        final_detached_release=detached,
        live_gate=_record(gate_path),
        runtime_identity=_record(runtime_path),
        executed_action_trace=executed,
        viewport_video=video,
        future_evidence=future,
        future_evidence_status=future_status,
    )
    export.update({
        "state_capture": _record(root / "state_capture/state_capture.json"),
        "action_trace_metadata": _record(metadata_path),
        "model_request_count": capture["model_request_count"],
        "live_gate_behavioral_action_count": capture["actions_executed"],
    })
    provenance = {"native_output_files": native_files, "video_probe": video_probe, "embedded_source_records": embedded}
    return export, provenance


def _existing_export(bundle: E004RuntimeBundle, spec: RecoverySpec, capture: Mapping[str, Any], *, probe_video: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    path = spec.source_attempt / "simulator_export.json"
    export = _load_json(path)
    _require(isinstance(export, dict), "retained simulator export is not an object")
    _validate_identity(export, bundle=bundle, spec=spec, label="retained simulator export")
    embedded = _validate_embedded_records(export, "simulator_export")
    for request_index, request in enumerate(export.get("future_evidence", {}).get("requests", [])):
        _validate_named_file(request, "action_path", "action_sha256", f"Nano action request {request_index}")
        _validate_named_file(request, "future_path", "future_sha256", f"Nano future request {request_index}")
        _validate_named_file(request, "session_manifest_path", "session_manifest_sha256", f"Nano session request {request_index}")
    video = _validate_bound_file(export.get("viewport_video", {}), "Nano viewport video")
    probe = _probe_mp4(Path(video["path"])) if probe_video else {"test_probe_bypassed": True}
    cell = bundle.cell(spec.cell_id)
    detached = bool(capture["steps"][-1]["grippers_open"] and not capture["steps"][-1]["object_grabbed"])
    _require(export.get("requested_success") is frozen_requested_success(capture["steps"], cell.relation, detached), "retained export score changed")
    _require(export.get("steps") == capture.get("steps"), "retained export steps differ from immutable capture")
    return export, {"source_simulator_export": _record(path), "video_probe": probe, "embedded_source_records": embedded}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recover_attempt(*, bundle: E004RuntimeBundle, spec: RecoverySpec, repo_root: Path, video_root: Path, probe_video: bool = True) -> dict[str, Any]:
    _require(bundle.registration_sha256 == REGISTRATION_SHA256 and bundle.queue_sha256 == QUEUE_SHA256 and bundle.candidate_sha256 == CANDIDATE_SHA256, "recovery bundle differs from the prospective freeze")
    _require(not spec.destination_attempt.exists(), f"refusing to overwrite recovery destination: {spec.destination_attempt}")
    before = _inventory(spec.source_attempt)
    code = _code_binding(repo_root, spec)
    attempt, infra, capture = _validate_attempt(bundle, spec)
    if spec.recovery_mode == "reconstruct_missing_export":
        _require(not (spec.source_attempt / "simulator_export.json").exists(), "missing-export recovery source unexpectedly has an export")
        export, source_provenance = _reconstruct_export(bundle, spec, attempt, capture, video_root, probe_video=probe_video)
    elif spec.recovery_mode == "recompile_existing_export":
        export, source_provenance = _existing_export(bundle, spec, capture, probe_video=probe_video)
    else:
        raise RuntimeContractError(f"unknown recovery mode: {spec.recovery_mode}")

    destination = spec.destination_attempt.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.offline-recovery-{uuid.uuid4().hex}")
    _require(not staging.exists(), "staging directory collision")
    staging.mkdir()
    try:
        export_path = staging / "simulator_export.json"
        _write_json(export_path, export)
        row = build_episode_record(export=export, bundle=bundle, cell=bundle.cell(spec.cell_id), output_path=destination / "raw_episode.jsonl")
        row["offline_recovery"] = {
            "schema_version": RECOVERY_SCHEMA,
            "source_attempt": _record(spec.source_attempt / "attempt_manifest.json"),
            "source_infrastructure_record": _record(spec.source_attempt / "infrastructure_invalid.json"),
            "source_simulator_export": source_provenance.get("source_simulator_export"),
            "source_execution_commit": spec.source_code_commit,
            "required_recovery_base_commit": RECOVERY_BASE_COMMIT,
            "recovery_commit": code["recovery_commit"],
            "recomputed_frozen_b001_success": bool(row["success"]),
            "raw_sources_immutable": True,
        }
        manifest = write_episode(record=row, output=staging / "raw_episode.jsonl")
        manifest["jsonl_path"] = str(destination / "raw_episode.jsonl")
        _write_json(staging / "raw_episode.jsonl.manifest.json", manifest)
        after = _inventory(spec.source_attempt)
        _require(before == after, "retained raw source changed during recovery")
        recovery = {
            "schema_version": RECOVERY_SCHEMA,
            "status": "recovered_offline_without_model_or_simulator_request",
            "recovery_mode": spec.recovery_mode,
            "registered_cell_id": spec.cell_id,
            "model_id": spec.model_id,
            "source_attempt_root": str(spec.source_attempt.resolve()),
            "destination_attempt_root": str(destination),
            "source_inventory_before": before,
            "source_inventory_after": after,
            "source_inventory_identical": True,
            "original_attempt_manifest": _record(spec.source_attempt / "attempt_manifest.json"),
            "original_infrastructure_record": _record(spec.source_attempt / "infrastructure_invalid.json"),
            "original_fault": {"error_type": infra["error_type"], "error": infra["error"], "stage": infra["stage"]},
            "state_capture": _record(spec.source_attempt / "state_capture/state_capture.json"),
            "recomputed_predicate": {
                "predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
                "definition": "one-step final detached release inside the requested 45-degree cone",
                "requested_success": bool(row["success"]),
                "failure_category": row["failure_category"],
                "actions_executed": row["actions_executed"],
            },
            "source_provenance": source_provenance,
            "code": code,
            "registration_sha256": bundle.registration_sha256,
            "queue_sha256": bundle.queue_sha256,
            "candidate_sha256": bundle.candidate_sha256,
            "outputs": {
                "simulator_export": {"path": str(destination / "simulator_export.json"), "bytes": export_path.stat().st_size, "sha256": sha256_file(export_path)},
                "raw_episode": {"path": str(destination / "raw_episode.jsonl"), "bytes": (staging / "raw_episode.jsonl").stat().st_size, "sha256": sha256_file(staging / "raw_episode.jsonl")},
                "raw_episode_manifest": {"path": str(destination / "raw_episode.jsonl.manifest.json"), "bytes": (staging / "raw_episode.jsonl.manifest.json").stat().st_size, "sha256": sha256_file(staging / "raw_episode.jsonl.manifest.json")},
            },
            "no_shard_manifest_written": True,
        }
        _write_json(staging / "offline_recovery_manifest.json", recovery)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _require(_inventory(spec.source_attempt) == before, "retained source changed after atomic recovery close")
    return {
        "name": spec.name,
        "cell_id": spec.cell_id,
        "success": row["success"],
        "failure_category": row["failure_category"],
        "destination": str(destination),
        "recovery_manifest_sha256": sha256_file(destination / "offline_recovery_manifest.json"),
        "raw_episode_sha256": sha256_file(destination / "raw_episode.jsonl"),
        "simulator_export_sha256": sha256_file(destination / "simulator_export.json"),
    }


def select_specs(specs: Iterable[RecoverySpec], names: Iterable[str]) -> tuple[RecoverySpec, ...]:
    by_name = {spec.name: spec for spec in specs}
    requested = tuple(names)
    _require(requested and len(requested) == len(set(requested)), "recovery names must be nonempty and unique")
    _require(set(requested) <= set(by_name), f"recovery name is outside allowlist: {set(requested) - set(by_name)}")
    return tuple(by_name[name] for name in requested)
