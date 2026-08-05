#!/usr/bin/env python3
"""Fail-closed GR00T N1.7 adapter for v3 DROID Phase A.

This module performs no model inference.  It validates the committed queue,
the pinned v2 integration sources, a live-runtime identity manifest, and a
separate fixed-observation release-gate artifact.  It can then emit the exact
command for :mod:`robolab_bridge` or compile a retained raw capture into the
shared v3 behavioral/infrastructure JSONL schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "groot_n17_droid_vla"
STUDY_ID = "vla_wam_language_steerability_v3"
PHASE = "A_direct_command_matched_pairs"
QUEUE_SCHEMA = "vla-wam-shared-v3-phase-a-cells-v1"
QUEUE_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_a_cells.jsonl")
QUEUE_MANIFEST_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_a_cells_manifest.json"
)
RUNTIME_SCHEMA = "vla-wam-shared-v3-groot-runtime-identity-v1"
GATE_SCHEMA = "vla-wam-shared-v3-groot-release-gate-v1"
CAPTURE_SCHEMA = "vla-wam-shared-v3-groot-state-capture-v1"
INFRA_CAPTURE_SCHEMA = "vla-wam-shared-v3-groot-infrastructure-capture-v1"

PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
FROZEN_CHECKPOINT = {
    "id": "nvidia/GR00T-N1.7-DROID",
    "revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
}
FROZEN_CHECKPOINT_FILES = {
    "model-00001-of-00002.safetensors":
        "68d885c9684bb7d4781389873e4b7d33202b5618e70a83f2e78187a5fb839202",
    "model-00002-of-00002.safetensors":
        "aa4c6e553ea8454500354352368bcbb7e4f0fb32a9816b20d5b25c231f13a8fd",
    "model.safetensors.index.json":
        "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59",
    "config.json":
        "b20d22636bdaf49436de49c5e7e5fc65203f7a4b88384eef426000100be57d1e",
    "processor_config.json":
        "4b5c3bab3f148ff47ba903714c3247403c754f806ff2354a73acdfa2102a66fb",
    "statistics.json":
        "127832f7df25cda15da4ba6be81737f96b65673d0f892f9fc1bce1bc062fa858",
}
FROZEN_ISAAC_GROOT_COMMIT = "b9955401d50c92a29258732e3ad6ccd579f1bdc0"
FROZEN_ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
FROZEN_SOURCES = {
    "experiments/groot_droid/v2_robolab_gate.py":
        "bb242769954469680efc5d53fdf3524240abca374f344ac02bd77785a7035c24",
    "experiments/groot_droid/v2_robolab_client.py":
        "81631cfa56cf979a9f8e1bd5d177ad119ee162891ccff0b47bbc16dc31e0b2eb",
    "experiments/groot_droid/v2_seeded_server.py":
        "993f84a4476bba9f70d1da549d96f5f1ad518a814a079d6204b581413c23686e",
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py":
        "9c4d90be770266bac3ba5242b743098348c565ee622179b5e88fa2af0c4891bc",
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py":
        "ba0eb879590960c57976dd1b749c4ebbd3e86054e152ca5af014ac1bc2b6d02a",
}
HEX64 = set("0123456789abcdef")


class AdapterError(ValueError):
    """A frozen contract, queue, runtime, or output mismatch."""


def _fail(message: str) -> None:
    raise AdapterError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                _fail(f"blank queue line at {path}:{number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                _fail(f"{path}:{number} must contain a JSON object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read queue {path}: {error}") from error
    return rows


def _validate_frozen_sources(study_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_SOURCES.items():
        path = study_root / relative
        if not path.is_file():
            _fail(f"missing frozen v2 integration source: {path}")
        observed[relative] = sha256_file(path)
        if observed[relative] != expected:
            _fail(
                f"frozen v2 integration source hash mismatch for {relative}: "
                f"expected {expected}, observed {observed[relative]}"
            )
    return observed


def checkpoint_contract_sha256() -> str:
    encoded = json.dumps(FROZEN_CHECKPOINT_FILES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def adapter_contract_sha256(study_root: Path) -> str:
    contract = {
        "model_id": MODEL_ID,
        "queue_sha256": sha256_file(study_root / QUEUE_RELATIVE),
        "frozen_v2_source_sha256": FROZEN_SOURCES,
        "checkpoint_contract_sha256": checkpoint_contract_sha256(),
        "behavioral_schema_version": "vla-wam-shared-v3-raw-episode-v1",
        "infrastructure_schema_version":
            "vla-wam-shared-v3-infrastructure-attempt-v1",
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_authorized_pair(study_root: Path, seed: int) -> list[dict[str, Any]]:
    """Return the exact LEFT/RIGHT rows for one launchable GR00T pair."""

    if seed not in range(8303, 8330):
        _fail("GR00T v3 Phase-A new seeds are exactly 8303-8329")
    queue = study_root / QUEUE_RELATIVE
    manifest_path = study_root / QUEUE_MANIFEST_RELATIVE
    manifest = _load_object(manifest_path)
    digest = sha256_file(queue)
    if manifest.get("queue_file") != str(QUEUE_RELATIVE):
        _fail("Phase-A manifest queue_file is not frozen")
    if manifest.get("queue_sha256") != digest:
        _fail("Phase-A queue hash does not match its committed manifest")
    if manifest.get("study_id") != STUDY_ID:
        _fail("Phase-A manifest study_id mismatch")

    selected = [
        row for row in _load_jsonl(queue)
        if row.get("model_id") == MODEL_ID and row.get("environment_seed") == seed
    ]
    if len(selected) != 2:
        _fail(f"expected exactly two registered GR00T rows for seed {seed}")
    by_relation = {row.get("relation"): row for row in selected}
    if set(by_relation) != {"left", "right"}:
        _fail("registered pair must contain one LEFT and one RIGHT row")
    for relation, row in by_relation.items():
        expected_cell = f"v3:droid:{MODEL_ID}:seed{seed}:{relation}"
        expected_pair = f"v3:droid:{MODEL_ID}:seed{seed}"
        expected_reset = f"v3:droid_robolab:neutral_reset:environment_seed_{seed}"
        checks = {
            "schema_version": QUEUE_SCHEMA,
            "study_id": STUDY_ID,
            "arena": "droid_robolab",
            "phase": PHASE,
            "model_id": MODEL_ID,
            "cell_id": expected_cell,
            "pair_id": expected_pair,
            "environment_seed": seed,
            "sampling_seed": seed,
            "replicate": 0,
            "status": "authorized_new",
            "execution_status": "authorized_after_all_registered_release_gates",
            "prompt_family": "direct_command",
            "prompt": PROMPTS[relation],
            "reset_identity": expected_reset,
            "success_predicate_id":
                "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                _fail(f"queue mismatch for {expected_cell}.{key}")
        prompt_hash = hashlib.sha256(PROMPTS[relation].encode()).hexdigest()
        if row.get("prompt_sha256") != prompt_hash:
            _fail(f"prompt hash mismatch for {expected_cell}")
        required = row.get("required_raw_outputs")
        if not isinstance(required, list) or required[:3] != [
            "viewport_video", "executed_action_trace", "raw_result_jsonl"
        ]:
            _fail(f"raw-output contract mismatch for {expected_cell}")
        runtime = row.get("runtime_identity_requirement")
        if not isinstance(runtime, dict) or runtime.get("left_right_must_match") is not True:
            _fail(f"runtime matching contract absent for {expected_cell}")
    return [by_relation["left"], by_relation["right"]]


def validate_runtime_identity(
    study_root: Path, path: Path, *, check_live_repositories: bool = False
) -> dict[str, Any]:
    identity = _load_object(path)
    expected_scalars = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "isaac_groot_commit": FROZEN_ISAAC_GROOT_COMMIT,
        "robolab_commit": FROZEN_ROBOLAB_COMMIT,
        "checkpoint_identifier": FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": FROZEN_CHECKPOINT["revision"],
        "open_loop_horizon": 8,
        "embodiment_tag": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    }
    for key, expected in expected_scalars.items():
        if identity.get(key) != expected:
            _fail(f"runtime identity mismatch for {key}")
    for key in (
        "external_repository_diff_hash", "checkpoint_sha256",
        "environment_lock_hash", "adapter_contract_hash",
        "isaac_groot_dir_status_sha256", "robolab_dir_status_sha256",
    ):
        if not _is_sha256(identity.get(key)):
            _fail(f"runtime identity requires lowercase SHA-256 field {key}")
    for key in ("simulator_version", "renderer_backend", "runtime_id"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            _fail(f"runtime identity requires non-empty {key}")
    observed_sources = _validate_frozen_sources(study_root)
    if identity.get("frozen_v2_source_sha256") != observed_sources:
        _fail("runtime identity frozen_v2_source_sha256 mismatch")
    queue_sha = sha256_file(study_root / QUEUE_RELATIVE)
    if identity.get("phase_a_queue_sha256") != queue_sha:
        _fail("runtime identity Phase-A queue hash mismatch")
    if identity["checkpoint_sha256"] != checkpoint_contract_sha256():
        _fail("runtime identity checkpoint contract hash mismatch")
    if identity.get("checkpoint_files") != FROZEN_CHECKPOINT_FILES:
        _fail("runtime identity checkpoint file hashes mismatch")
    if identity["adapter_contract_hash"] != adapter_contract_sha256(study_root):
        _fail("runtime identity adapter contract hash mismatch")
    if identity["external_repository_diff_hash"] != identity["isaac_groot_dir_status_sha256"]:
        _fail("external repository diff hash must equal the Isaac-GR00T status hash")
    if check_live_repositories:
        for label, directory_key, commit_key in (
            ("Isaac-GR00T", "isaac_groot_dir", "isaac_groot_commit"),
            ("RoboLab", "robolab_dir", "robolab_commit"),
        ):
            directory = Path(str(identity.get(directory_key, "")))
            if not directory.is_dir():
                _fail(f"runtime identity {label} directory is unavailable")
            observed = subprocess.check_output(
                ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
            ).strip()
            if observed != identity[commit_key]:
                _fail(f"live {label} commit does not match runtime identity")
            status = subprocess.check_output(
                ["git", "-C", str(directory), "status", "--porcelain=v1"], text=True
            )
            status_hash = hashlib.sha256(status.encode()).hexdigest()
            expected_status = identity.get(f"{directory_key}_status_sha256")
            if expected_status != status_hash:
                _fail(f"live {label} diff/status hash does not match runtime identity")
        checkpoint_path = Path(str(identity.get("checkpoint_path", "")))
        if not checkpoint_path.is_dir():
            _fail("runtime identity checkpoint_path is unavailable")
        for relative, expected in FROZEN_CHECKPOINT_FILES.items():
            file_path = checkpoint_path / relative
            if not file_path.is_file() or sha256_file(file_path) != expected:
                _fail(f"live checkpoint file hash mismatch for {relative}")
        lock_path = Path(str(identity.get("environment_lock_path", "")))
        if not lock_path.is_file():
            _fail("runtime identity environment_lock_path is unavailable")
        if sha256_file(lock_path) != identity["environment_lock_hash"]:
            _fail("live environment lock hash mismatch")
    return identity


def validate_release_gate(
    path: Path, *, queue_sha256: str, runtime_identity_sha256: str
) -> dict[str, Any]:
    gate = _load_object(path)
    expected = {
        "schema_version": GATE_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "phase": PHASE,
        "phase_a_queue_sha256": queue_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "left_prompt": PROMPTS["left"],
        "right_prompt": PROMPTS["right"],
        "model_blind_neutral_reset_fixture_passed": True,
        "raw_video_action_jsonl_write_passed": True,
        "fixed_observation_exact_repeat_passed": True,
        "fixed_observation_left_right_prompt_sensitivity_passed": True,
        "behavioral_release": True,
    }
    for key, wanted in expected.items():
        if gate.get(key) != wanted:
            _fail(f"GR00T release gate mismatch for {key}")
    repeat_rms = gate.get("left_exact_repeat_rms")
    sensitivity_rms = gate.get("left_right_action_rms")
    if type(repeat_rms) not in {int, float} or float(repeat_rms) != 0.0:
        _fail("GR00T exact-repeat gate must have zero RMS")
    if (
        type(sensitivity_rms) not in {int, float}
        or not math.isfinite(float(sensitivity_rms))
        or float(sensitivity_rms) <= 0.0
    ):
        _fail("GR00T prompt-sensitivity gate requires positive LEFT/RIGHT RMS")
    for key in ("left_action_sha256", "right_action_sha256", "gate_artifact_sha256"):
        if not _is_sha256(gate.get(key)):
            _fail(f"GR00T release gate requires {key}")
    if gate["left_action_sha256"] == gate["right_action_sha256"]:
        _fail("GR00T LEFT and RIGHT gate actions must differ")
    return gate


def preflight(
    study_root: Path,
    seed: int,
    runtime_identity_path: Path,
    release_gate_path: Path,
    *,
    check_live_repositories: bool = False,
) -> dict[str, Any]:
    cells = load_authorized_pair(study_root, seed)
    runtime = validate_runtime_identity(
        study_root, runtime_identity_path,
        check_live_repositories=check_live_repositories,
    )
    queue_sha = sha256_file(study_root / QUEUE_RELATIVE)
    runtime_sha = sha256_file(runtime_identity_path)
    gate = validate_release_gate(
        release_gate_path,
        queue_sha256=queue_sha,
        runtime_identity_sha256=runtime_sha,
    )
    return {
        "status": "ready",
        "seed": seed,
        "cell_ids": [cell["cell_id"] for cell in cells],
        "pair_id": cells[0]["pair_id"],
        "phase_a_queue_sha256": queue_sha,
        "runtime_identity": runtime,
        "runtime_identity_sha256": runtime_sha,
        "release_gate_sha256": sha256_file(release_gate_path),
        "release_gate": gate,
    }


def bridge_command(
    study_root: Path,
    seed: int,
    runtime_identity_path: Path,
    release_gate_path: Path,
    output_dir: Path,
    action_trace_dir: Path,
    remote_host: str,
    remote_port: int,
) -> list[str]:
    return [
        sys.executable,
        str(study_root / "experiments/v3/groot_droid/robolab_bridge.py"),
        "--study-root", str(study_root),
        "--environment-seed", str(seed),
        "--sampling-seed-base", str(seed),
        "--runtime-identity", str(runtime_identity_path),
        "--release-gate", str(release_gate_path),
        "--state-capture-dir", str(output_dir / "state_capture"),
        "--action-trace-dir", str(action_trace_dir),
        "--remote-host", remote_host,
        "--remote-port", str(remote_port),
        "--open-loop-horizon", "8",
        "--instruction-controller", "static",
        "--condition", "both",
        "--output-dir", str(output_dir),
        "--num-envs", "1",
        "--num-runs", "1",
        "--video-mode", "viewport",
        "--disable-subtask",
    ]


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"required raw artifact is missing: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def _in_cone(sample: dict[str, Any], relation: str) -> bool:
    obj = sample["object_xyz"]
    ref = sample["reference_xyz"]
    forward = float(obj[0]) - float(ref[0])
    lateral = float(obj[1]) - float(ref[1])
    margin = lateral if relation == "left" else -lateral
    distance = math.hypot(forward, lateral)
    return distance > 1e-8 and margin / distance >= math.cos(math.radians(45.0))


def _event_timeline(capture: dict[str, Any], relation: str) -> list[dict[str, Any]]:
    samples = capture["samples"]
    initial_z = float(samples[0]["object_xyz"][2])
    pickup = _first_sustained([
        float(sample["object_xyz"][2]) - initial_z >= 0.03 for sample in samples
    ])
    requested = next(
        (index for index, sample in enumerate(samples) if _in_cone(sample, relation)),
        None,
    )
    opposite_relation = "right" if relation == "left" else "left"
    opposite = next(
        (index for index, sample in enumerate(samples) if _in_cone(sample, opposite_relation)),
        None,
    )
    contact = capture.get("first_contact_step")
    events = [{"event": "episode_start", "action_step": 0}]
    for name, step in (
        ("first_contact", contact),
        ("verified_pickup", pickup),
        ("requested_region_entry", requested),
        ("opposite_region_entry", opposite),
    ):
        if step is not None:
            events.append({"event": name, "action_step": int(step)})
    end = int(capture["actions_executed"])
    events.append({"event": "episode_end", "action_step": end})
    order = {
        "episode_start": 0,
        "first_contact": 1,
        "verified_pickup": 2,
        "requested_region_entry": 3,
        "opposite_region_entry": 4,
        "episode_end": 5,
    }
    return sorted(events, key=lambda event: (event["action_step"], order[event["event"]]))


def build_behavioral_record(
    study_root: Path,
    cell: dict[str, Any],
    capture: dict[str, Any],
    runtime_identity_path: Path,
    video_path: Path,
    action_trace_path: Path,
    raw_jsonl_path: Path,
) -> dict[str, Any]:
    """Build and validate one shared-schema behavioral record."""

    relation = cell["relation"]
    expected_capture = {
        "schema_version": CAPTURE_SCHEMA,
        "registered_cell_id": cell["cell_id"],
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "prompt": cell["prompt"],
        "requested_relation": relation,
    }
    for key, expected in expected_capture.items():
        if capture.get(key) != expected:
            _fail(f"state capture mismatch for {cell['cell_id']}.{key}")
    samples = capture.get("samples")
    if not isinstance(samples, list) or not samples:
        _fail("state capture samples must be a non-empty array")
    actions_executed = capture.get("actions_executed")
    if type(actions_executed) is not int or actions_executed < 0:
        _fail("state capture actions_executed must be a non-negative integer")
    if len(samples) != actions_executed + 1:
        _fail("state capture must retain initial plus every post-action state")
    if capture.get("behavioral_result_valid_candidate") is not True:
        _fail("partial/unreconciled state capture cannot enter a behavioral denominator")
    if capture.get("action_cap") != 450:
        _fail("GR00T DROID Phase A uses the frozen 450-action cap")
    if capture.get("requested_success") is True:
        if capture.get("right_censored") is not False:
            _fail("a frozen success termination is not right-censored")
    elif (
        capture.get("right_censored") is not True
        or actions_executed != capture.get("action_cap")
    ):
        _fail("a behavioral failure must run to the frozen action cap")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("action_step") != index:
            _fail("state capture action_step sequence is not contiguous")

    runtime = validate_runtime_identity(study_root, runtime_identity_path)
    runtime_sha = sha256_file(runtime_identity_path)
    steps = []
    for sample in samples:
        step = {
            "action_step": sample["action_step"],
            "object_xyz": sample["object_xyz"],
            "reference_xyz": sample["reference_xyz"],
            "grippers_open": sample["grippers_open"],
        }
        if "contact_detected" in sample:
            step["contact_detected"] = sample["contact_detected"]
        steps.append(step)
    record: dict[str, Any] = {
        "schema_version": "vla-wam-shared-v3-raw-episode-v1",
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": STUDY_ID,
        "arena": "droid_robolab",
        "registered_cell_id": cell["cell_id"],
        "attempt_id": capture["attempt_id"],
        "model_id": MODEL_ID,
        "pair_id": cell["pair_id"],
        "prompt": cell["prompt"],
        "prompt_family": cell["prompt_family"],
        "predicate_id": cell["success_predicate_id"],
        "reset_id": cell["reset_identity"],
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "requested_relation": relation,
        "requested_success": capture["requested_success"],
        "failure_stage": capture["frozen_failure_stage"],
        "frozen_failure_stage": capture["frozen_failure_stage"],
        "failure_taxonomy": "transport_failed",
        "measurement_frame": "robot_base_object_minus_reference_xyz_m",
        "measurement_frame_description": (
            "Object and reference XYZ samples are expressed in the frozen robot-base frame; "
            "forward is object-minus-reference x and lateral is object-minus-reference y, "
            "with positive lateral denoting robot LEFT."
        ),
        "checkpoint": dict(FROZEN_CHECKPOINT),
        "runtime_identity": {"id": runtime["runtime_id"], "sha256": runtime_sha},
        "artifacts": {
            "viewport_video": _file_record(video_path),
            "executed_action_trace": _file_record(action_trace_path),
            "raw_result_jsonl": {
                "path": str(raw_jsonl_path),
                "integrity_scope": "batch_manifest_after_close",
            },
        },
        "steps": steps,
        "actions_executed": actions_executed,
        "action_cap": capture["action_cap"],
        "right_censored": capture["right_censored"],
        "first_contact_step": capture.get("first_contact_step"),
        "first_contact_unavailable_reason": capture.get(
            "first_contact_unavailable_reason"
        ),
        "final_detached_release": capture["final_detached_release"],
        "wall_time_s": capture["wall_time_s"],
        "operational_wall_time_valid": capture["operational_wall_time_valid"],
        "event_timeline": _event_timeline(capture, relation),
    }
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import (  # type: ignore
        derive_failure_taxonomy,
        derive_initial_state_sha256,
        derive_measurements,
        validate_behavioral_record,
    )

    record["initial_state_sha256"] = derive_initial_state_sha256(record)
    measurements = derive_measurements(record)
    record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
    return validate_behavioral_record(record)


def build_infrastructure_record(
    study_root: Path,
    cell: dict[str, Any],
    capture: dict[str, Any],
    runtime_identity_path: Path,
    video_path: Path | None,
    action_trace_path: Path | None,
    raw_jsonl_path: Path,
) -> dict[str, Any]:
    """Build one schema-valid, denominator-excluded infrastructure record."""

    if capture.get("schema_version") != INFRA_CAPTURE_SCHEMA:
        _fail("unexpected GR00T infrastructure capture schema")
    for key, expected in (
        ("registered_cell_id", cell["cell_id"]),
        ("environment_seed", cell["environment_seed"]),
        ("policy_seed", cell["sampling_seed"]),
        ("prompt", cell["prompt"]),
        ("requested_relation", cell["relation"]),
    ):
        if capture.get(key) != expected:
            _fail(f"infrastructure capture mismatch for {cell['cell_id']}.{key}")
    runtime = validate_runtime_identity(study_root, runtime_identity_path)
    timeline = capture.get("event_timeline")
    if not isinstance(timeline, list) or not timeline:
        _fail("infrastructure capture requires a non-empty event_timeline")
    artifacts: dict[str, Any] = {
        "raw_result_jsonl": {
            "path": str(raw_jsonl_path),
            "integrity_scope": "batch_manifest_after_close",
        }
    }
    if video_path is not None:
        artifacts["viewport_video"] = _file_record(video_path)
    if action_trace_path is not None:
        artifacts["executed_action_trace"] = _file_record(action_trace_path)
    record = {
        "schema_version": "vla-wam-shared-v3-infrastructure-attempt-v1",
        "record_type": "infrastructure_attempt",
        "behavioral_result_valid": False,
        "classification": capture["classification"],
        "study_id": STUDY_ID,
        "arena": "droid_robolab",
        "registered_cell_id": cell["cell_id"],
        "attempt_id": capture["attempt_id"],
        "model_id": MODEL_ID,
        "pair_id": cell["pair_id"],
        "prompt": cell["prompt"],
        "prompt_family": cell["prompt_family"],
        "predicate_id": cell["success_predicate_id"],
        "reset_id": cell["reset_identity"],
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "measurement_frame": "robot_base_object_minus_reference_xyz_m",
        "measurement_frame_description": (
            "Object and reference XYZ samples are expressed in the frozen robot-base frame; "
            "forward is object-minus-reference x and lateral is object-minus-reference y, "
            "with positive lateral denoting robot LEFT."
        ),
        "checkpoint": dict(FROZEN_CHECKPOINT),
        "runtime_identity": {
            "id": runtime["runtime_id"],
            "sha256": sha256_file(runtime_identity_path),
        },
        "artifacts": artifacts,
        "stage": capture["stage"],
        "error": capture["error"],
        "log_hash": capture["log_hash"],
        "runtime_intervention": capture["runtime_intervention"],
        "repair_attempt_id": capture.get("repair_attempt_id"),
        "event_timeline": timeline,
    }
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import validate_infrastructure_record  # type: ignore

    return validate_infrastructure_record(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("preflight", "plan"):
        command = subparsers.add_parser(name)
        command.add_argument("--study-root", type=Path, required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--runtime-identity", type=Path, required=True)
        command.add_argument("--release-gate", type=Path, required=True)
        command.add_argument("--check-live-repositories", action="store_true")
    plan = subparsers.choices["plan"]
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--action-trace-dir", type=Path, required=True)
    plan.add_argument("--remote-host", required=True)
    plan.add_argument("--remote-port", type=int, default=5555)
    for name in ("compile-behavioral", "compile-infrastructure"):
        command = subparsers.add_parser(name)
        command.add_argument("--study-root", type=Path, required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--relation", choices=["left", "right"], required=True)
        command.add_argument("--runtime-identity", type=Path, required=True)
        command.add_argument("--capture", type=Path, required=True)
        command.add_argument("--output-jsonl", type=Path, required=True)
        command.add_argument(
            "--video", type=Path, required=name == "compile-behavioral"
        )
        command.add_argument(
            "--action-trace", type=Path, required=name == "compile-behavioral"
        )
    args = parser.parse_args()

    root = args.study_root.resolve()
    if args.mode in {"compile-behavioral", "compile-infrastructure"}:
        cells = {cell["relation"]: cell for cell in load_authorized_pair(root, args.seed)}
        capture = _load_object(args.capture)
        builder = (
            build_behavioral_record
            if args.mode == "compile-behavioral"
            else build_infrastructure_record
        )
        result = builder(
            root, cells[args.relation], capture, args.runtime_identity,
            args.video, args.action_trace, args.output_jsonl,
        )
        sys.path.insert(0, str(root / "tools"))
        from vla_wam_v3_episode_schema import write_jsonl  # type: ignore

        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        manifest = write_jsonl(args.output_jsonl, [result])
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    result = preflight(
        root, args.seed, args.runtime_identity, args.release_gate,
        check_live_repositories=args.check_live_repositories,
    )
    if args.mode == "plan":
        result["command"] = bridge_command(
            root, args.seed, args.runtime_identity,
            args.release_gate, args.output_dir, args.action_trace_dir,
            args.remote_host, args.remote_port,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
