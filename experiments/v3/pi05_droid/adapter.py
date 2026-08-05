#!/usr/bin/env python3
"""Fail-closed powered-v3 adapter for current-stack pi0.5 DROID Phase A.

The module performs no inference.  It validates the committed Phase-A queue,
the unchanged V2-A010 action/runtime contracts, a hash-bearing live identity,
and a fresh v3 exact-repeat/prompt-sensitivity release artifact.  Only then can
it emit the complete matched LEFT/RIGHT bridge command or compile retained raw
captures into the shared v3 behavioral/infrastructure JSONL schemas.
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


MODEL_ID = "pi05_current_stack_droid"
STUDY_ID = "vla_wam_language_steerability_v3"
PHASE = "A_direct_command_matched_pairs"
QUEUE_SCHEMA = "vla-wam-shared-v3-phase-a-cells-v1"
QUEUE_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_a_cells.jsonl")
QUEUE_MANIFEST_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_a_cells_manifest.json")
CHECKPOINT_MANIFEST_RELATIVE = Path(
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "pi05_current_stack_checkpoint_manifest.json"
)
RUNTIME_SCHEMA = "vla-wam-shared-v3-pi05-current-runtime-identity-v1"
GATE_SCHEMA = "vla-wam-shared-v3-pi05-current-release-gate-v1"
CAPTURE_SCHEMA = "vla-wam-shared-v3-pi05-current-state-capture-v1"
INFRA_CAPTURE_SCHEMA = "vla-wam-shared-v3-pi05-current-infrastructure-capture-v1"

PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
FROZEN_OPENPI_COMMIT = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
FROZEN_ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
FROZEN_CONFIG = "pi05_droid_jointpos_polaris"
FROZEN_CHECKPOINT_MANIFEST_SHA256 = (
    "f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca"
)
FROZEN_CHECKPOINT = {
    "id": FROZEN_CONFIG,
    "revision": f"v2a010-manifest-{FROZEN_CHECKPOINT_MANIFEST_SHA256}",
}
FROZEN_V2_SOURCES = {
    "experiments/pi05_current_stack/v2a010_serve_policy.py":
        "cd415e3a98da977f395242c24bb8f3d3187eb4cc3bf53c5dc659d190e6934051",
    "experiments/pi05_current_stack/v2a010_robolab_client.py":
        "de633d7219cc20da7fc7b87b6831139073eed135bbb7662e346c9e0912d89141",
    "experiments/pi05_current_stack/v2a010_fixed_observation_probe.py":
        "87c6c5a748db8a52221acbb43df3b15038ec3807cc14217dd062db160d94c328",
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py":
        "9c4d90be770266bac3ba5242b743098348c565ee622179b5e88fa2af0c4891bc",
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py":
        "ba0eb879590960c57976dd1b749c4ebbd3e86054e152ca5af014ac1bc2b6d02a",
}
ADAPTER_SOURCES = (
    "experiments/v3/pi05_droid/adapter.py",
    "experiments/v3/pi05_droid/client.py",
    "experiments/v3/pi05_droid/robolab_bridge.py",
)
HEX64 = set("0123456789abcdef")


class AdapterError(ValueError):
    """The queue, runtime, release gate, or retained evidence is inconsistent."""


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
        _fail(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                _fail(f"blank line in queue at {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                _fail(f"queue row at {path}:{number} is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read queue {path}: {error}") from error
    return rows


def _observed_source_hashes(study_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_V2_SOURCES.items():
        path = study_root / relative
        if not path.is_file():
            _fail(f"missing frozen V2-A010 source: {path}")
        observed[relative] = sha256_file(path)
        if observed[relative] != expected:
            _fail(f"frozen V2-A010 source hash mismatch: {relative}")
    return observed


def adapter_source_sha256(study_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in ADAPTER_SOURCES:
        path = study_root / relative
        if not path.is_file():
            _fail(f"missing v3 pi0.5 adapter source: {path}")
        result[relative] = sha256_file(path)
    return result


def checkpoint_contract_sha256(study_root: Path) -> str:
    path = study_root / CHECKPOINT_MANIFEST_RELATIVE
    if sha256_file(path) != FROZEN_CHECKPOINT_MANIFEST_SHA256:
        _fail("committed pi0.5 checkpoint manifest hash changed")
    manifest = _load_object(path)
    if (
        manifest.get("schema_version")
        != "vla-wam-v2a010-pi05-current-checkpoint-manifest-v1"
        or manifest.get("status") != "complete_sha256_hashed_before_model_load"
        or manifest.get("file_count") != 26
        or manifest.get("payload_bytes") != 12_434_530_510
        or len(manifest.get("files", [])) != 26
    ):
        _fail("committed pi0.5 checkpoint manifest contract changed")
    payload = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in manifest["files"]
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def adapter_contract_sha256(study_root: Path) -> str:
    contract = {
        "model_id": MODEL_ID,
        "queue_sha256": sha256_file(study_root / QUEUE_RELATIVE),
        "checkpoint_contract_sha256": checkpoint_contract_sha256(study_root),
        "frozen_v2_sources": FROZEN_V2_SOURCES,
        "adapter_sources": adapter_source_sha256(study_root),
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "robolab_commit": FROZEN_ROBOLAB_COMMIT,
        "config": FROZEN_CONFIG,
        "open_loop_horizon": 15,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_authorized_pair(study_root: Path, seed: int) -> list[dict[str, Any]]:
    """Return exactly the prospectively authorized LEFT/RIGHT rows for ``seed``."""

    if seed not in range(8303, 8330):
        _fail("pi0.5 v3 Phase-A new seeds are exactly 8303-8329")
    queue_path = study_root / QUEUE_RELATIVE
    manifest = _load_object(study_root / QUEUE_MANIFEST_RELATIVE)
    queue_sha = sha256_file(queue_path)
    if (
        manifest.get("queue_file") != str(QUEUE_RELATIVE)
        or manifest.get("queue_sha256") != queue_sha
        or manifest.get("study_id") != STUDY_ID
    ):
        _fail("committed Phase-A queue/manifest identity mismatch")
    selected = [
        row for row in _load_jsonl(queue_path)
        if row.get("model_id") == MODEL_ID and row.get("environment_seed") == seed
    ]
    if len(selected) != 2:
        _fail(f"expected exactly two pi0.5 rows for seed {seed}")
    by_relation = {row.get("relation"): row for row in selected}
    if set(by_relation) != {"left", "right"}:
        _fail("registered pi0.5 pair must contain one LEFT and one RIGHT row")
    for relation, row in by_relation.items():
        expected = {
            "schema_version": QUEUE_SCHEMA,
            "study_id": STUDY_ID,
            "arena": "droid_robolab",
            "phase": PHASE,
            "model_id": MODEL_ID,
            "cell_id": f"v3:droid:{MODEL_ID}:seed{seed}:{relation}",
            "pair_id": f"v3:droid:{MODEL_ID}:seed{seed}",
            "environment_seed": seed,
            "sampling_seed": seed,
            "replicate": 0,
            "status": "authorized_new",
            "execution_status": "authorized_after_all_registered_release_gates",
            "prompt_family": "direct_command",
            "prompt": PROMPTS[relation],
            "reset_identity": f"v3:droid_robolab:neutral_reset:environment_seed_{seed}",
            "success_predicate_id":
                "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
        }
        for key, wanted in expected.items():
            if row.get(key) != wanted:
                _fail(f"queue mismatch for seed {seed}/{relation}/{key}")
        if row.get("prompt_sha256") != hashlib.sha256(PROMPTS[relation].encode()).hexdigest():
            _fail(f"prompt hash mismatch for seed {seed}/{relation}")
        requirement = row.get("runtime_identity_requirement", {})
        if requirement.get("left_right_must_match") is not True:
            _fail("queue no longer requires identical LEFT/RIGHT runtime identity")
    return [by_relation["left"], by_relation["right"]]


def _git_status_sha256(directory: Path) -> tuple[str, str]:
    head = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain=v1"], text=True
    )
    return head, hashlib.sha256(status.encode()).hexdigest()


def validate_runtime_identity(
    study_root: Path, path: Path, *, check_live_repositories: bool = False
) -> dict[str, Any]:
    identity = _load_object(path)
    expected = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "robolab_commit": FROZEN_ROBOLAB_COMMIT,
        "openpi_config": FROZEN_CONFIG,
        "checkpoint_identifier": FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": FROZEN_CHECKPOINT["revision"],
        "checkpoint_manifest_sha256": FROZEN_CHECKPOINT_MANIFEST_SHA256,
        "open_loop_horizon": 15,
        "action_chunk_shape": [15, 8],
    }
    for key, wanted in expected.items():
        if identity.get(key) != wanted:
            _fail(f"pi0.5 runtime identity mismatch for {key}")
    for key in (
        "checkpoint_sha256", "environment_lock_hash", "adapter_contract_hash",
        "external_repository_diff_hash", "openpi_dir_status_sha256",
        "robolab_dir_status_sha256",
    ):
        if not _is_sha256(identity.get(key)):
            _fail(f"pi0.5 runtime identity requires lowercase SHA-256 {key}")
    for key in ("runtime_id", "simulator_version", "renderer_backend"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            _fail(f"pi0.5 runtime identity requires non-empty {key}")
    if identity.get("phase_a_queue_sha256") != sha256_file(study_root / QUEUE_RELATIVE):
        _fail("pi0.5 runtime identity queue hash mismatch")
    if identity["checkpoint_sha256"] != checkpoint_contract_sha256(study_root):
        _fail("pi0.5 checkpoint contract hash mismatch")
    if identity["adapter_contract_hash"] != adapter_contract_sha256(study_root):
        _fail("pi0.5 adapter contract hash mismatch")
    if identity.get("frozen_v2_source_sha256") != _observed_source_hashes(study_root):
        _fail("pi0.5 frozen V2 source hashes mismatch")
    if identity.get("adapter_source_sha256") != adapter_source_sha256(study_root):
        _fail("pi0.5 v3 adapter source hashes mismatch")
    combined = hashlib.sha256(json.dumps(
        {
            "openpi": identity["openpi_dir_status_sha256"],
            "robolab": identity["robolab_dir_status_sha256"],
        }, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if identity["external_repository_diff_hash"] != combined:
        _fail("pi0.5 combined external-repository diff hash mismatch")
    if check_live_repositories:
        for label, dir_key, commit_key, status_key in (
            ("OpenPI", "openpi_dir", "openpi_commit", "openpi_dir_status_sha256"),
            ("RoboLab", "robolab_dir", "robolab_commit", "robolab_dir_status_sha256"),
        ):
            directory = Path(str(identity.get(dir_key, "")))
            if not directory.is_dir():
                _fail(f"live {label} directory is unavailable")
            head, status_sha = _git_status_sha256(directory)
            if head != identity[commit_key] or status_sha != identity[status_key]:
                _fail(f"live {label} revision/diff differs from runtime identity")
        checkpoint_root = Path(str(identity.get("checkpoint_path", "")))
        manifest = _load_object(study_root / CHECKPOINT_MANIFEST_RELATIVE)
        if not checkpoint_root.is_dir():
            _fail("live pi0.5 checkpoint directory is unavailable")
        for row in manifest["files"]:
            file_path = checkpoint_root / row["path"]
            if (
                not file_path.is_file()
                or file_path.stat().st_size != row["bytes"]
                or sha256_file(file_path) != row["sha256"]
            ):
                _fail(f"live pi0.5 checkpoint payload mismatch: {row['path']}")
        lock = Path(str(identity.get("environment_lock_path", "")))
        if not lock.is_file() or sha256_file(lock) != identity["environment_lock_hash"]:
            _fail("live pi0.5 environment lock hash mismatch")
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
        "open_loop_horizon": 15,
        "action_shape": [15, 8],
    }
    for key, wanted in expected.items():
        if gate.get(key) != wanted:
            _fail(f"pi0.5 release gate mismatch for {key}")
    if gate.get("left_exact_repeat_bit_identical") is not True:
        _fail("pi0.5 release gate requires bit-identical LEFT repeat")
    rms = gate.get("left_right_action_rms")
    if type(rms) not in {int, float} or not math.isfinite(float(rms)) or float(rms) <= 0:
        _fail("pi0.5 release gate requires positive LEFT/RIGHT action RMS")
    for key in ("left_action_sha256", "right_action_sha256", "gate_artifact_sha256"):
        if not _is_sha256(gate.get(key)):
            _fail(f"pi0.5 release gate requires {key}")
    if gate["left_action_sha256"] == gate["right_action_sha256"]:
        _fail("pi0.5 release-gate LEFT and RIGHT action hashes must differ")
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
        "pair_id": cells[0]["pair_id"],
        "cell_ids": [row["cell_id"] for row in cells],
        "runtime_identity": runtime,
        "runtime_identity_sha256": runtime_sha,
        "release_gate": gate,
        "release_gate_sha256": sha256_file(release_gate_path),
        "phase_a_queue_sha256": queue_sha,
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
        str(study_root / "experiments/v3/pi05_droid/robolab_bridge.py"),
        "--study-root", str(study_root),
        "--environment-seed", str(seed),
        "--sampling-seed-base", str(seed),
        "--runtime-identity", str(runtime_identity_path),
        "--release-gate", str(release_gate_path),
        "--state-capture-dir", str(output_dir / "state_capture"),
        "--action-trace-dir", str(action_trace_dir),
        "--remote-host", remote_host,
        "--remote-port", str(remote_port),
        "--open-loop-horizon", "15",
        "--instruction-controller", "static",
        "--condition", "both",
        "--output-dir", str(output_dir),
        "--num-envs", "1",
        "--num-runs", "1",
        "--video-mode", "viewport",
        "--instruction-type", "default",
        "--disable-subtask",
    ]


def _file_record(path: Path, *, nonempty: bool = True) -> dict[str, Any]:
    if not path.is_file() or (nonempty and path.stat().st_size <= 0):
        _fail(f"required retained artifact is absent/empty: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_action_trace(
    path: Path, cell: dict[str, Any], actions_executed: int
) -> tuple[dict[str, Any], Path, Path]:
    trace = _load_object(path)
    expected = {
        "schema_version": "vla-wam-shared-v3-pi05-action-trace-v1",
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "environment_seed": cell["environment_seed"],
        "sampling_seed_base": cell["sampling_seed"],
        "prompt": cell["prompt"],
        "requested_relation": cell["relation"],
        "prompt_controller": "episode_static",
        "open_loop_execution_horizon": 15,
    }
    for key, wanted in expected.items():
        if trace.get(key) != wanted:
            _fail(f"pi0.5 action trace mismatch for {key}")
    entry = trace.get("executed_actions", {})
    action_path = Path(str(entry.get("path", "")))
    if (
        not action_path.is_file()
        or entry.get("sha256") != sha256_file(action_path)
        or entry.get("bytes") != action_path.stat().st_size
        or entry.get("count") != actions_executed
        or entry.get("shape") != [actions_executed, 8]
        or entry.get("dtype") != "float32"
    ):
        _fail("pi0.5 executed-action artifact/metadata mismatch")
    request_seeds = trace.get("request_sampling_seeds")
    if not isinstance(request_seeds, list) or not request_seeds or request_seeds != [
        cell["sampling_seed"] * 1000 + index for index in range(len(request_seeds))
    ]:
        _fail("pi0.5 per-request seed attestation sequence is invalid")
    chunks_entry = trace.get("returned_action_chunks", {})
    chunks_path = Path(str(chunks_entry.get("path", "")))
    if (
        not chunks_path.is_file()
        or chunks_entry.get("sha256") != sha256_file(chunks_path)
        or chunks_entry.get("bytes") != chunks_path.stat().st_size
        or chunks_entry.get("count") != len(request_seeds)
        or chunks_entry.get("shape") != [len(request_seeds), 15, 8]
        or chunks_entry.get("dtype") != "float32"
    ):
        _fail("pi0.5 returned action-chunk artifact/metadata mismatch")
    return trace, action_path, chunks_path


def _first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def _in_cone(sample: dict[str, Any], relation: str) -> bool:
    obj, ref = sample["object_xyz"], sample["reference_xyz"]
    forward = float(obj[0]) - float(ref[0])
    lateral = float(obj[1]) - float(ref[1])
    margin = lateral if relation == "left" else -lateral
    distance = math.hypot(forward, lateral)
    return distance > 1e-8 and margin / distance >= math.cos(math.radians(45))


def _event_timeline(capture: dict[str, Any], relation: str) -> list[dict[str, Any]]:
    samples = capture["samples"]
    z0 = float(samples[0]["object_xyz"][2])
    pickup = _first_sustained([
        float(sample["object_xyz"][2]) - z0 >= 0.03 for sample in samples
    ])
    requested = next((i for i, sample in enumerate(samples) if _in_cone(sample, relation)), None)
    opposite_relation = "right" if relation == "left" else "left"
    opposite = next((i for i, sample in enumerate(samples) if _in_cone(sample, opposite_relation)), None)
    events = [{"event": "episode_start", "action_step": 0}]
    for name, step in (
        ("first_contact", capture.get("first_contact_step")),
        ("verified_pickup", pickup),
        ("requested_region_entry", requested),
        ("opposite_region_entry", opposite),
    ):
        if step is not None:
            events.append({"event": name, "action_step": int(step)})
    events.append({"event": "episode_end", "action_step": int(capture["actions_executed"])})
    rank = {name: index for index, name in enumerate((
        "episode_start", "first_contact", "verified_pickup",
        "requested_region_entry", "opposite_region_entry", "episode_end",
    ))}
    return sorted(events, key=lambda event: (event["action_step"], rank[event["event"]]))


def build_behavioral_record(
    study_root: Path,
    cell: dict[str, Any],
    capture: dict[str, Any],
    runtime_identity_path: Path,
    video_path: Path,
    action_trace_metadata_path: Path,
    raw_jsonl_path: Path,
) -> dict[str, Any]:
    relation = cell["relation"]
    expected = {
        "schema_version": CAPTURE_SCHEMA,
        "registered_cell_id": cell["cell_id"],
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "prompt": cell["prompt"],
        "requested_relation": relation,
    }
    for key, wanted in expected.items():
        if capture.get(key) != wanted:
            _fail(f"pi0.5 state capture mismatch for {key}")
    samples = capture.get("samples")
    actions_executed = capture.get("actions_executed")
    if type(actions_executed) is not int or actions_executed < 0:
        _fail("pi0.5 capture requires a non-negative actions_executed")
    if not isinstance(samples, list) or len(samples) != actions_executed + 1:
        _fail("pi0.5 capture must retain initial plus every post-action state")
    if capture.get("behavioral_result_valid_candidate") is not True:
        _fail("partial pi0.5 capture cannot enter a behavioral denominator")
    if capture.get("action_cap") != 450:
        _fail("pi0.5 uses the frozen 450-action cap")
    if capture.get("requested_success") is True:
        if capture.get("right_censored") is not False:
            _fail("successful pi0.5 episode cannot be right-censored")
    elif capture.get("right_censored") is not True or actions_executed != 450:
        _fail("valid pi0.5 failure must run to the 450-action cap")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("action_step") != index:
            _fail("pi0.5 state samples are not contiguous")
    runtime = validate_runtime_identity(study_root, runtime_identity_path)
    _, actions_path, chunks_path = _validate_action_trace(
        action_trace_metadata_path, cell, actions_executed
    )
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
        "runtime_identity": {
            "id": runtime["runtime_id"],
            "sha256": sha256_file(runtime_identity_path),
        },
        "artifacts": {
            "viewport_video": _file_record(video_path),
            "executed_action_trace": _file_record(actions_path),
            "returned_action_chunks": _file_record(chunks_path),
            "action_trace_manifest": _file_record(action_trace_metadata_path),
            "raw_result_jsonl": {
                "path": str(raw_jsonl_path),
                "integrity_scope": "batch_manifest_after_close",
            },
        },
        "steps": steps,
        "actions_executed": actions_executed,
        "action_cap": 450,
        "right_censored": capture["right_censored"],
        "first_contact_step": capture.get("first_contact_step"),
        "first_contact_unavailable_reason": capture.get("first_contact_unavailable_reason"),
        "final_detached_release": capture["final_detached_release"],
        "wall_time_s": capture["wall_time_s"],
        "operational_wall_time_valid": capture["operational_wall_time_valid"],
        "event_timeline": _event_timeline(capture, relation),
        "controller_contract": {
            "open_loop_horizon": 15,
            "prompt_controller": "episode_static",
            "failure_early_stopping": False,
        },
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
    raw_jsonl_path: Path,
    *,
    video_path: Path | None = None,
    action_trace_path: Path | None = None,
) -> dict[str, Any]:
    if capture.get("schema_version") != INFRA_CAPTURE_SCHEMA:
        _fail("unexpected pi0.5 infrastructure capture schema")
    for key, wanted in (
        ("registered_cell_id", cell["cell_id"]),
        ("environment_seed", cell["environment_seed"]),
        ("policy_seed", cell["sampling_seed"]),
        ("prompt", cell["prompt"]),
        ("requested_relation", cell["relation"]),
    ):
        if capture.get(key) != wanted:
            _fail(f"pi0.5 infrastructure capture mismatch for {key}")
    runtime = validate_runtime_identity(study_root, runtime_identity_path)
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
        "event_timeline": capture["event_timeline"],
    }
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import validate_infrastructure_record  # type: ignore
    return validate_infrastructure_record(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("preflight", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--study-root", type=Path, required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--runtime-identity", type=Path, required=True)
        command.add_argument("--release-gate", type=Path, required=True)
        command.add_argument("--check-live-repositories", action="store_true")
    plan = commands.choices["plan"]
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--action-trace-dir", type=Path, required=True)
    plan.add_argument("--remote-host", required=True)
    plan.add_argument("--remote-port", type=int, default=8001)
    behavioral = commands.add_parser("compile-behavioral")
    for argument in ("study-root", "runtime-identity", "capture", "video", "action-trace", "output-jsonl"):
        behavioral.add_argument(f"--{argument}", type=Path, required=True)
    behavioral.add_argument("--seed", type=int, required=True)
    behavioral.add_argument("--relation", choices=["left", "right"], required=True)
    infrastructure = commands.add_parser("compile-infrastructure")
    for argument in ("study-root", "runtime-identity", "capture", "output-jsonl"):
        infrastructure.add_argument(f"--{argument}", type=Path, required=True)
    infrastructure.add_argument("--video", type=Path)
    infrastructure.add_argument("--action-trace", type=Path)
    infrastructure.add_argument("--seed", type=int, required=True)
    infrastructure.add_argument("--relation", choices=["left", "right"], required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    if args.mode.startswith("compile-"):
        cell = {row["relation"]: row for row in load_authorized_pair(root, args.seed)}[args.relation]
        capture = _load_object(args.capture)
        if args.mode == "compile-behavioral":
            result = build_behavioral_record(
                root, cell, capture, args.runtime_identity, args.video,
                args.action_trace, args.output_jsonl,
            )
        else:
            result = build_infrastructure_record(
                root, cell, capture, args.runtime_identity, args.output_jsonl,
                video_path=args.video, action_trace_path=args.action_trace,
            )
        sys.path.insert(0, str(root / "tools"))
        from vla_wam_v3_episode_schema import write_jsonl  # type: ignore
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(write_jsonl(args.output_jsonl, [result]), indent=2, sort_keys=True))
        return
    result = preflight(
        root, args.seed, args.runtime_identity, args.release_gate,
        check_live_repositories=args.check_live_repositories,
    )
    if args.mode == "plan":
        result["command"] = bridge_command(
            root, args.seed, args.runtime_identity, args.release_gate,
            args.output_dir, args.action_trace_dir, args.remote_host, args.remote_port,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
