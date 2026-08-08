from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004 import cosmos_runtime as cr


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _cell(model_id: str, *, seed: int, candidate_sha: str) -> dict:
    relation = "left"
    level = 1.0
    short = "nano" if model_id == "cosmos3_nano_policy_droid" else "edge"
    spec = cr.MODEL_SPECS[model_id]
    runtime = {
        "action_cap": 450,
        "action_horizon": 32,
        "checkpoint": spec["checkpoint"],
        "checkpoint_revision": spec["checkpoint_revision"],
        "phase_a_runtime_identity_sha256": spec["phase_a_runtime_identity_sha256"],
    }
    if short == "nano":
        runtime.update(
            server_repository_commit=spec["server_repository_commit"],
            robolab_commit=spec["robolab_commit"],
        )
    else:
        runtime["checkpoint_sha256"] = spec["checkpoint_sha256"]
    return {
        "schema_version": "vla-wam-shared-v3e004-cell-v1",
        "study_id": cr.STUDY_ID,
        "amendment_id": cr.AMENDMENT_ID,
        "arena": "droid_robolab",
        "model_id": model_id,
        "cell_id": f"v3e004:{short}:seed{seed}:s100:left",
        "matched_pair_id": f"v3e004:{short}:seed{seed}:s100",
        "environment_seed": seed,
        "sampling_seed": seed,
        "symmetry_level_s": level,
        "relation": relation,
        "prompt": cr.PROMPTS[relation],
        "prompt_sha256": hashlib.sha256(cr.PROMPTS[relation].encode()).hexdigest(),
        "static_episode_prompt": True,
        "execution_mode": "new_behavioral_episode",
        "layout_candidate_sha256": candidate_sha,
        "success_predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
        "runtime_identity_requirement": runtime,
    }


def _repo(tmp_path: Path) -> tuple[Path, str, dict[str, dict]]:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)

    candidate = b'{"schema_version":"test-candidate"}\n'
    candidate_sha = hashlib.sha256(candidate).hexdigest()
    rows = {
        "nano": _cell(
            "cosmos3_nano_policy_droid", seed=9920, candidate_sha=candidate_sha
        ),
        "edge": _cell(
            "cosmos3_edge_policy_droid", seed=9400, candidate_sha=candidate_sha
        ),
    }
    queue = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows.values()
    )
    queue_sha = hashlib.sha256(queue).hexdigest()
    registration = {
        "schema_version": "vla-wam-shared-v3e004-registration-v1",
        "study_id": cr.STUDY_ID,
        "amendment_id": cr.AMENDMENT_ID,
        "status": "prospectively_registered_zero_e004_model_requests_or_behavioral_episodes",
        "model_request_count_before_registration": 0,
        "behavioral_episode_count_before_registration": 0,
        "success_predicates_frozen": True,
        "queue": {
            "path": cr.QUEUE_RELATIVE,
            "sha256": queue_sha,
            "bytes": len(queue),
            "rows": len(rows),
        },
        "layout": {
            "candidate_path": cr.CANDIDATE_RELATIVE,
            "candidate_sha256": candidate_sha,
        },
    }
    _write(root / cr.CANDIDATE_RELATIVE, candidate)
    _write(root / cr.QUEUE_RELATIVE, queue)
    _write(
        root / cr.REGISTRATION_RELATIVE,
        json.dumps(registration, indent=2, sort_keys=True).encode() + b"\n",
    )
    for relative in cr.SOURCE_RELATIVES:
        _write(root / relative, (relative + "\n").encode())
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "register"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return root, commit, rows


def _runtime(model_id: str) -> dict:
    spec = cr.MODEL_SPECS[model_id]
    payload = {
        "schema_version": "vla-wam-shared-v3e004-test-runtime-v1",
        "study_id": cr.STUDY_ID,
        "model_id": model_id,
        "checkpoint_identifier": spec["checkpoint"],
        "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": spec["server_repository_commit"],
        "external_repository_diff_hash": cr.EMPTY_SHA256,
        "simulator_repository_commit": spec["robolab_commit"],
        "simulator_repository_diff_hash": cr.EMPTY_SHA256,
        "phase_a_runtime_identity_sha256": spec["phase_a_runtime_identity_sha256"],
        "environment_lock_sha256": "1" * 64,
    }
    payload["runtime_identity_sha256"] = cr.sha256_bytes(cr.canonical_json_bytes(payload))
    return payload


def _gate(path: Path, *, candidate_sha: str, cell: cr.AuthorizedCell, live: bool) -> None:
    value = {
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_sha256": candidate_sha,
        "model_id": cell.model_id,
        "registered_cell_id": cell.cell_id,
    }
    if live:
        value["scene"] = {"symmetry_level_s": cell.symmetry_level}
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_commit_binding_accepts_only_exact_new_registered_cells(tmp_path: Path) -> None:
    root, commit, rows = _repo(tmp_path)
    bundle = cr.load_registration_bundle(root, registration_commit=commit)
    assert bundle.cell(rows["nano"]["cell_id"]).seed == 9920  # registered extension seed
    assert bundle.cell(rows["edge"]["cell_id"]).seed == 9400
    with pytest.raises(cr.CosmosRuntimeError, match="not a new registered"):
        bundle.cell("v3e004:nano:seed9921:s100:left")

    with (root / cr.QUEUE_RELATIVE).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(cr.CosmosRuntimeError, match="differs from commit"):
        cr.load_registration_bundle(root, registration_commit=commit)


def test_request_response_hashes_bind_cell_session_input_and_output(tmp_path: Path) -> None:
    root, commit, rows = _repo(tmp_path)
    bundle = cr.load_registration_bundle(root, registration_commit=commit)
    cell = bundle.cell(rows["nano"]["cell_id"])
    runtime = _runtime(cell.model_id)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    gates = {}
    for label in ("static_layout", "live_camera_reset", "raw_write", "renderer"):
        path = tmp_path / f"{label}.json"
        _gate(path, candidate_sha=bundle.candidate_sha256, cell=cell, live=label == "live_camera_reset")
        gates[label] = path
    session = cr.build_session_manifest(
        bundle=bundle,
        cell=cell,
        runtime=runtime,
        runtime_manifest_path=runtime_path,
        session_id="nano-session-1",
        attempt_id="attempt01",
        gate_paths=gates,
        initial_state_sha256="2" * 64,
    )
    session_path = tmp_path / "sessions" / "session.json"
    session_path.parent.mkdir()
    session_path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")
    session_file_sha = cr.sha256_file(session_path)

    native_request = {
        "prompt": cell.row["prompt"],
        "sampling_seed": cell.seed,
        "observation/image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
    }
    request = cr.add_request_envelope(
        native_request,
        bundle=bundle,
        cell=cell,
        runtime=runtime,
        session=session,
        session_manifest_path=session_path,
        session_manifest_sha256=session_file_sha,
        request_index=0,
        action_step_start=0,
    )
    observed_cell, observed_session, _, native = cr.validate_request_envelope(
        request,
        bundle=bundle,
        runtime=runtime,
        model_id=cell.model_id,
        expected_request_index=0,
        session_root=session_path.parent,
    )
    assert observed_cell.cell_id == cell.cell_id
    assert observed_session["session_id"] == "nano-session-1"
    assert cr.hash_value(native) == request["model_input_sha256"]

    native_output = {
        "action": np.zeros((32, 8), dtype=np.float32),
        "video": np.zeros((33, 2, 2, 3), dtype=np.uint8),
    }
    response = cr.add_response_envelope(
        native_output,
        cell=cell,
        runtime=runtime,
        session=session,
        request=request,
    )
    verified = cr.validate_response_envelope(
        response,
        cell=cell,
        runtime=runtime,
        session=session,
        pending_request=request,
    )
    assert cr.hash_value(verified) == response["model_output_sha256"]

    # The OpenPI websocket transport injects timing only after the policy
    # response has been hash-bound.  Transport metadata must not change the
    # native model-output digest observed by the client.
    transported = {
        **response,
        "server_timing": {"infer_ms": 12.5, "prev_total_ms": 13.0},
    }
    transported_verified = cr.validate_response_envelope(
        transported,
        cell=cell,
        runtime=runtime,
        session=session,
        pending_request=request,
    )
    assert transported_verified == verified
    assert cr.hash_value(transported_verified) == response["model_output_sha256"]

    response["action"][0, 0] = 1.0
    with pytest.raises(cr.CosmosRuntimeError, match="model_output_sha256"):
        cr.validate_response_envelope(
            response,
            cell=cell,
            runtime=runtime,
            session=session,
            pending_request=request,
        )


def test_server_cli_keeps_model_environments_and_prompt_transport_separate() -> None:
    nano = cr.MODEL_SPECS["cosmos3_nano_policy_droid"]
    nano_cli = [
        "--checkpoint-path", nano["checkpoint_path"],
        "--hf-revision", nano["checkpoint_revision"],
        "--port", "18011",
        "--domain-name", "droid_lerobot",
        "--decode-video",
        "--action-chunk-size", "32",
        "--action-dim", "8",
        "--action-space", "joint_pos",
    ]
    assert cr.ensure_exact_server_cli("cosmos3_nano_policy_droid", nano_cli)[
        "environment_id"
    ] != cr.MODEL_SPECS["cosmos3_edge_policy_droid"]["environment_id"]

    edge = cr.MODEL_SPECS["cosmos3_edge_policy_droid"]
    edge_cli = [
        "--checkpoint-path", edge["checkpoint_path"],
        "--port", "18010",
        "--decode-video",
        "--format-prompt-as-json", "True",
        "--action-chunk-size", "32",
        "--action-dim", "8",
        "--action-space", "joint_pos",
    ]
    cr.ensure_exact_server_cli("cosmos3_edge_policy_droid", edge_cli)
    with pytest.raises(cr.CosmosRuntimeError, match="JSON prompt"):
        cr.ensure_exact_server_cli(
            "cosmos3_edge_policy_droid",
            [token for token in edge_cli if token not in {"--format-prompt-as-json", "True"}],
        )


def test_client_records_verified_input_output_and_session_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubCosmos3Client:
        def __init__(self, **_: object) -> None:
            pass

        def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
            return {**extracted_obs, "prompt": instruction}

        def reset(self, *, env_id: int | None = None) -> None:
            del env_id

    policies = types.ModuleType("policies")
    cosmos = types.ModuleType("policies.cosmos3")
    client_module = types.ModuleType("policies.cosmos3.client")
    client_module.Cosmos3Client = StubCosmos3Client
    monkeypatch.setitem(sys.modules, "policies", policies)
    monkeypatch.setitem(sys.modules, "policies.cosmos3", cosmos)
    monkeypatch.setitem(sys.modules, "policies.cosmos3.client", client_module)
    sys.modules.pop("experiments.v3.phase_e.symmetric_layout_cohort_v3e004.cosmos_client", None)
    sys.modules.pop("v2_robolab_client", None)
    client_code = importlib.import_module(
        "experiments.v3.phase_e.symmetric_layout_cohort_v3e004.cosmos_client"
    )

    root, commit, rows = _repo(tmp_path)
    bundle = cr.load_registration_bundle(root, registration_commit=commit)
    cell = bundle.cell(rows["nano"]["cell_id"])
    runtime = _runtime(cell.model_id)
    runtime_path = tmp_path / "runtime-client.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    gates = {}
    for label in ("static_layout", "live_camera_reset", "raw_write", "renderer"):
        path = tmp_path / f"client-{label}.json"
        _gate(path, candidate_sha=bundle.candidate_sha256, cell=cell, live=label == "live_camera_reset")
        gates[label] = path
    session = cr.build_session_manifest(
        bundle=bundle,
        cell=cell,
        runtime=runtime,
        runtime_manifest_path=runtime_path,
        session_id="client-session",
        attempt_id="attempt01",
        gate_paths=gates,
        initial_state_sha256="3" * 64,
    )
    session_path = tmp_path / "client-session.json"
    session_path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")
    actions = tmp_path / "actions"
    futures = tmp_path / "futures"
    live = client_code.E004CosmosClient(
        bundle=bundle,
        cell=cell,
        runtime=runtime,
        session_manifest_path=session_path,
        sampling_seed_base=cell.seed,
        action_trace_dir=actions,
        future_trace_dir=futures,
    )
    live.prompt = cell.row["prompt"]
    request = live._pack_request(
        {
            "sampling_seed": cell.seed,
            "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        cell.row["prompt"],
    )
    native_output = {
        "action": np.zeros((32, 8), dtype=np.float32),
        "video": np.zeros((33, 2, 2, 3), dtype=np.uint8),
    }
    response = cr.add_response_envelope(
        native_output,
        cell=cell,
        runtime=runtime,
        session=session,
        request=request,
    )
    action = live._unpack_response(response)
    assert action.shape == (32, 8)
    record = live.request_records[-1]
    assert record["registered_cell_id"] == cell.cell_id
    assert record["session_sha256"] == session["session_sha256"]
    assert record["model_input_sha256"] == request["model_input_sha256"]
    assert record["model_output_sha256"] == response["model_output_sha256"]
