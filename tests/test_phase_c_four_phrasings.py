from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.v3.phase_c_four_phrasings.build_registration import (
    OUTPUT_RELATIVE,
    build_cells,
    build_gate_requests,
    materialize,
)
from experiments.v3.phase_c_four_phrasings.contract import (
    EXPERIMENT_ID,
    MODEL_CONTRACTS,
    PROMPT_FORMS,
    PROMPTS,
    SEEDS,
    ContractError,
    canonical_json_bytes,
    prompt_sha256,
    randomized_conditions,
    select_whole_seed_blocks,
    sha256_file,
    validate_cells,
    validate_release_manifest,
)
from experiments.v3.phase_c_four_phrasings.fixed_observation_gate import (
    GateError,
    evaluate_records,
)
from experiments.v3.phase_c_four_phrasings.live_fixed_observation import _array_record
from experiments.v3.phase_c_four_phrasings.raw_write_preflight import run as run_write_preflight
from experiments.v3.phase_c_four_phrasings.groot_behavioral_contract import (
    TASK_SPECS,
    prompt_condition,
    validate_live_output_contract,
    validate_live_task_registration,
    validate_task_sources,
)
from experiments.v3.phase_c_four_phrasings.cosmos_behavioral_contract import (
    validate_seed_block as validate_cosmos_seed_block,
    validate_live_output_contract as validate_cosmos_live_output_contract,
    validate_live_task_registration as validate_cosmos_live_task_registration,
)
from experiments.v3.phase_c_four_phrasings.runner import build_plan


ROOT = Path(__file__).resolve().parents[1]


def test_exact_480_cell_registry_and_randomization() -> None:
    rows = build_cells()
    assert len(rows) == 480
    assert Counter(row["model_id"] for row in rows) == Counter(
        {model_id: 160 for model_id in MODEL_CONTRACTS}
    )
    blocks: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        blocks[(row["model_id"], row["seed"])].append(row)
        exact = PROMPTS[row["prompt_family"]][row["relation"]]
        assert row["prompt"] == exact
        assert row["prompt_sha256"] == prompt_sha256(exact)
        assert row["environment_seed"] == row["sampling_seed"] == row["seed"]
    assert set(blocks) == {(model, seed) for model in MODEL_CONTRACTS for seed in SEEDS}
    for (model, seed), block in blocks.items():
        ordered = sorted(block, key=lambda row: row["within_seed_execution_order"])
        assert [row["within_seed_execution_order"] for row in ordered] == list(range(1, 9))
        assert [(row["prompt_family"], row["relation"]) for row in ordered] == randomized_conditions(model, seed)


def test_materialization_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    materialize(ROOT, first)
    materialize(ROOT, second)
    names = {
        "prospective_phase_c_v3c001_registration.json",
        "phase_c_v3c001_cells.jsonl",
        "phase_c_v3c001_fixed_observation_requests.jsonl",
        "phase_c_v3c001_manifest.json",
    }
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_fixed_observation_request_registry() -> None:
    rows = build_gate_requests()
    assert len(rows) == 36
    counts = Counter((row["model_id"], row["prompt_family"]) for row in rows)
    assert set(counts.values()) == {3}
    assert set(counts) == {(model, form) for model in MODEL_CONTRACTS for form in PROMPT_FORMS}
    for row in rows:
        assert row["behavioral_episode"] is False
        assert row["prompt_sha256"] == prompt_sha256(row["prompt"])


def _gate_records(model_id: str) -> list[dict]:
    records = []
    future_required = MODEL_CONTRACTS[model_id]["fixed_observation_future_required"]
    for form in PROMPT_FORMS:
        for condition, relation, actions, future in (
            ("left", "left", [[0.0, 1.0]], [[[0.0, 1.0]]]),
            ("left_exact_repeat", "left", [[0.0, 1.0]], [[[0.0, 1.0]]]),
            ("right", "right", [[1.0, 1.0]], [[[1.0, 1.0]]]),
        ):
            prompt = PROMPTS[form][relation]
            record = {
                "model_id": model_id,
                "prompt_family": form,
                "condition": condition,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256(prompt),
                "observation_sha256": "a" * 64,
                "sampling_seed": 17,
                "actions": actions,
            }
            if future_required:
                record["decoded_future"] = future
            records.append(record)
    return records


@pytest.mark.parametrize("model_id", tuple(MODEL_CONTRACTS))
def test_fixed_observation_gate_requires_repeat_and_sensitivity(model_id: str) -> None:
    records = _gate_records(model_id)
    report = evaluate_records(records, model_id=model_id)
    assert report["passed"] is True
    assert report["behavioral_episode_count"] == 0
    broken = [dict(record) for record in records]
    broken[1]["actions"] = [[0.0, 2.0]]
    assert evaluate_records(broken, model_id=model_id)["passed"] is False


def test_cosmos_future_gate_is_mandatory_but_groot_is_action_only() -> None:
    cosmos = _gate_records("cosmos3_edge_policy_droid")
    del cosmos[0]["decoded_future"]
    with pytest.raises(GateError):
        evaluate_records(cosmos, model_id="cosmos3_edge_policy_droid")
    groot = _gate_records("groot_n17_droid_vla")
    assert evaluate_records(groot, model_id="groot_n17_droid_vla")["passed"] is True


def test_fixed_observation_gate_accepts_hash_bound_npy_artifacts(tmp_path: Path) -> None:
    records = _gate_records("cosmos3_edge_policy_droid")
    for index, record in enumerate(records):
        actions = tmp_path / f"actions-{index}.npy"
        future = tmp_path / f"future-{index}.npy"
        record["actions"] = _array_record(actions, __import__("numpy").asarray(record["actions"], dtype="float32"))
        record["decoded_future"] = _array_record(future, __import__("numpy").asarray(record["decoded_future"], dtype="uint8"))
    assert evaluate_records(records, model_id="cosmos3_edge_policy_droid")["passed"] is True


def test_raw_write_preflight_is_model_blind_and_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWriter:
        def __init__(self, path: str, *_: object) -> None:
            self.path = Path(path)

        def isOpened(self) -> bool:
            return True

        def write(self, _: object) -> None:
            self.path.write_bytes(b"model-blind-mp4-writer-proof")

        def release(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoWriter=FakeWriter,
        VideoWriter_fourcc=lambda *_: 0,
    ))
    output = tmp_path / "write"
    report = run_write_preflight(model_id="groot_n17_droid_vla", output_dir=output)
    assert report["passed"] is True
    assert report["model_request_count"] == report["behavioral_episode_count"] == 0
    assert set(report["outputs"]) == {
        "simulator_viewport_video", "executed_actions", "state_trace", "behavioral_jsonl"
    }
    with pytest.raises(ValueError, match="already exists"):
        run_write_preflight(model_id="groot_n17_droid_vla", output_dir=output)


def test_groot_prompt_routing_uses_exact_bytes_and_retains_all_task_sources() -> None:
    assert {
        prompt_condition(PROMPTS[form][relation])
        for form in PROMPT_FORMS
        for relation in ("left", "right")
    } == set(TASK_SPECS)
    with pytest.raises(ContractError, match="not one exact"):
        prompt_condition("Put it left, not right.")
    hashes = validate_task_sources(ROOT)
    assert len(hashes) == 8 and all(len(value) == 64 for value in hashes.values())
    for relative in hashes:
        source = (ROOT / relative).read_text()
        assert " as _base" in source
        assert " import RubiksCube" not in source


def test_groot_live_registration_and_output_paths_fail_closed(tmp_path: Path) -> None:
    cells = []
    registration_cells = []
    for order, ((form, relation), (filename, task_name)) in enumerate(TASK_SPECS.items(), 1):
        raw = tmp_path / f"cell-{order}"
        prompt = PROMPTS[form][relation]
        cell = {
            "registered_cell_id": f"v3c001:droid:groot_n17_droid_vla:seed8500:{form}:{relation}",
            "within_seed_execution_order": order,
            "task_name": task_name,
            "task_file": str(ROOT / "experiments/v3/phase_c_four_phrasings/groot_task_files" / filename),
            "prompt": prompt,
            "prompt_family": form,
            "relation": relation,
            "raw_cell_directory": str(raw),
            "required_outputs": {
                "behavioral_jsonl": str(raw / "episode.jsonl"),
                "executed_actions": str(raw / "executed_actions.npy"),
                "simulator_viewport_video": str(raw / "viewport.mp4"),
                "state_trace": str(raw / "state_trace.jsonl"),
                "decoded_future": "required_when_exposed_by_runtime",
            },
        }
        cells.append(cell)
        registration_cells.append({
            "registered_cell_id": cell["registered_cell_id"],
            "within_seed_execution_order": order,
            "task_name": task_name,
            "prompt": prompt,
            "left_predicate_at_reset": False,
            "right_predicate_at_reset": False,
            "model_requests": 0,
            "actions_executed": 0,
        })
        assert validate_live_output_contract(cell) == {
            "behavioral_jsonl": raw / "episode.jsonl",
            "executed_actions": raw / "executed_actions.npy",
            "simulator_viewport_video": raw / "viewport.mp4",
            "state_trace": raw / "state_trace.jsonl",
        }
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_bytes(canonical_json_bytes({"seed": 8500, "cells": cells}))
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(canonical_json_bytes({
        "schema_version": "vla-wam-shared-v3c-groot-live-task-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": "groot_n17_droid_vla",
        "seed": 8500,
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "executed_action_count": 0,
        "renderer_initialized": True,
        "matched_reset_tolerance_m": 0.003,
        "max_cube_position_spread_m": 0.0,
        "max_bowl_position_spread_m": 0.0,
        "bridge_preflight": {
            "path": str(bridge_path),
            "sha256": sha256_file(bridge_path),
        },
        "cells": registration_cells,
    }))
    validate_live_task_registration(
        bridge_preflight_path=bridge_path,
        task_registration_path=registration_path,
    )
    bad = json.loads(registration_path.read_text())
    bad["cells"][0]["prompt"] += " changed"
    registration_path.write_bytes(canonical_json_bytes(bad))
    with pytest.raises(ContractError, match="cell mismatch"):
        validate_live_task_registration(
            bridge_preflight_path=bridge_path,
            task_registration_path=registration_path,
        )


def test_cosmos_live_registration_and_future_paths_fail_closed(tmp_path: Path) -> None:
    model_id = "cosmos3_edge_policy_droid"
    cells = []
    registration_cells = []
    for order, ((form, relation), (filename, task_name)) in enumerate(TASK_SPECS.items(), 1):
        raw = tmp_path / f"cosmos-cell-{order}"
        prompt = PROMPTS[form][relation]
        cell = {
            "registered_cell_id": f"v3c001:droid:{model_id}:seed8500:{form}:{relation}",
            "within_seed_execution_order": order,
            "task_name": task_name,
            "task_file": str(
                ROOT / "experiments/v3/phase_c_four_phrasings/groot_task_files" / filename
            ),
            "prompt": prompt,
            "prompt_family": form,
            "relation": relation,
            "raw_cell_directory": str(raw),
            "required_outputs": {
                "behavioral_jsonl": str(raw / "episode.jsonl"),
                "executed_actions": str(raw / "executed_actions.npy"),
                "simulator_viewport_video": str(raw / "viewport.mp4"),
                "state_trace": str(raw / "state_trace.jsonl"),
                "decoded_future": "required_when_exposed_by_runtime",
            },
        }
        cells.append(cell)
        registration_cells.append({
            "registered_cell_id": cell["registered_cell_id"],
            "within_seed_execution_order": order,
            "task_name": task_name,
            "prompt": prompt,
            "left_predicate_at_reset": False,
            "right_predicate_at_reset": False,
            "model_requests": 0,
            "actions_executed": 0,
        })
        outputs = validate_cosmos_live_output_contract(cell)
        assert outputs["decoded_future_directory"] == raw / "decoded_futures"
        assert outputs["action_future_metadata"] == raw / "action_future_trace.json"

    bridge_path = tmp_path / "cosmos-bridge.json"
    bridge_path.write_bytes(canonical_json_bytes({
        "model_id": model_id,
        "seed": 8500,
        "cells": cells,
    }))
    registration_path = tmp_path / "cosmos-registration.json"
    registration_path.write_bytes(canonical_json_bytes({
        "schema_version": "vla-wam-shared-v3c-cosmos-live-task-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "seed": 8500,
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "executed_action_count": 0,
        "renderer_initialized": True,
        "matched_reset_tolerance_m": 0.003,
        "max_cube_position_spread_m": 0.0,
        "max_bowl_position_spread_m": 0.0,
        "bridge_preflight": {
            "path": str(bridge_path),
            "sha256": sha256_file(bridge_path),
        },
        "cells": registration_cells,
    }))
    validate_cosmos_live_task_registration(
        bridge_preflight_path=bridge_path,
        task_registration_path=registration_path,
    )
    bad = json.loads(registration_path.read_text())
    bad["max_cube_position_spread_m"] = 0.01
    registration_path.write_bytes(canonical_json_bytes(bad))
    with pytest.raises(ContractError, match="resets are not matched"):
        validate_cosmos_live_task_registration(
            bridge_preflight_path=bridge_path,
            task_registration_path=registration_path,
        )


def _release(
    model_id: str,
    registration_sha: str,
    proof_path: Path,
    *,
    passed: bool = True,
) -> dict:
    contract = MODEL_CONTRACTS[model_id]
    proof = {"proof_path": str(proof_path), "proof_sha256": sha256_file(proof_path)}
    return {
        "schema_version": "vla-wam-shared-v3c-four-phrasings-release-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "registration_manifest_sha256": registration_sha,
        "runtime_identity": {
            "semantic_sha256": contract["phase_a_runtime_identity_sha256"],
            "checkpoint": contract["checkpoint"],
            "checkpoint_revision": contract["checkpoint_revision"],
            "path": str(proof_path),
            "file_sha256": sha256_file(proof_path),
        },
        "phase_a_direct_release": {
            "passed": True,
            "runtime_identity_match": True,
            **proof,
        },
        "gates": {
            "prompt_byte_hash": {"passed": True, **proof},
            "fixed_observation_exact_repeat": {"passed": True, **proof},
            "fixed_observation_prompt_only_sensitivity": {
                "passed": passed,
                "prompt_forms": list(PROMPT_FORMS),
                **proof,
            },
            "raw_video_action_jsonl_state_write": {"passed": True, **proof},
        },
        "behavioral_release": passed,
    }


def test_release_contract_fails_closed_and_lanes_keep_seed_blocks(tmp_path: Path) -> None:
    registration_manifest = ROOT / OUTPUT_RELATIVE / "phase_c_v3c001_manifest.json"
    registration_sha = sha256_file(registration_manifest)
    model_id = "groot_n17_droid_vla"
    proof_path = tmp_path / "retained-proof.json"
    proof_path.write_text("{}\n")
    release_path = tmp_path / "release.json"
    release_path.write_bytes(
        canonical_json_bytes(_release(model_id, registration_sha, proof_path))
    )
    released = validate_release_manifest(
        release_path, model_id=model_id, registration_manifest_sha256=registration_sha
    )
    assert released.model_id == model_id
    release_path.write_bytes(
        canonical_json_bytes(
            _release(model_id, registration_sha, proof_path, passed=False)
        )
    )
    with pytest.raises(ContractError, match="remains unreleased"):
        validate_release_manifest(
            release_path, model_id=model_id, registration_manifest_sha256=registration_sha
        )

    rows = validate_cells(build_cells())
    selected = select_whole_seed_blocks(rows, model_id=model_id, lane_index=1, lane_count=3)
    counts = Counter(row["seed"] for row in selected)
    assert counts and set(counts.values()) == {8}
    assert not any(row["model_id"] != model_id for row in selected)


def test_cosmos_seed_preflight_consumes_one_complete_released_block(tmp_path: Path) -> None:
    model_id = "cosmos3_edge_policy_droid"
    registration_manifest = ROOT / OUTPUT_RELATIVE / "phase_c_v3c001_manifest.json"
    registration_sha = sha256_file(registration_manifest)
    proof_path = tmp_path / "retained-proof.json"
    proof_path.write_text("{}\n")
    release_path = tmp_path / "release.json"
    release_path.write_bytes(
        canonical_json_bytes(_release(model_id, registration_sha, proof_path))
    )
    plan = build_plan(
        cells_path=ROOT / OUTPUT_RELATIVE / "phase_c_v3c001_cells.jsonl",
        registration_manifest_path=registration_manifest,
        release_manifest_path=release_path,
        raw_root=tmp_path / "raw",
        model_id=model_id,
        lane_index=0,
        lane_count=1,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan))
    report = validate_cosmos_seed_block(
        study_root=ROOT,
        execution_plan=plan_path,
        release_manifest=release_path,
        registration_manifest=registration_manifest,
        model_id=model_id,
        seed=8500,
    )
    assert report["passed"] is True
    assert len(report["cells"]) == 8
    assert [
        (cell["prompt_family"], cell["relation"]) for cell in report["cells"]
    ] == randomized_conditions(model_id, 8500)
    Path(report["cells"][0]["raw_cell_directory"]).mkdir(parents=True)
    with pytest.raises(ContractError, match="refusing to overwrite"):
        validate_cosmos_seed_block(
            study_root=ROOT,
            execution_plan=plan_path,
            release_manifest=release_path,
            registration_manifest=registration_manifest,
            model_id=model_id,
            seed=8500,
        )


def test_cosmos_queue_is_single_client_and_serial() -> None:
    source = (
        ROOT
        / "experiments/v3/phase_c_four_phrasings/run_cosmos_phase_c_queue.sh"
    ).read_text()
    assert 'single_client_lock="$gate_root/single_client_server.lock"' in source
    assert 'if ! mkdir "$single_client_lock"' in source
    assert 'trap cleanup_single_client_lock EXIT' in source
    assert "trap 'exit 130' INT TERM" in source
    assert 'for seed in $(seq "$seed_start" "$seed_end"); do' in source
    assert not any(line.rstrip().endswith(" &") for line in source.splitlines())


def test_cosmos_live_client_requires_registered_seed_echo() -> None:
    source = (
        ROOT / "experiments/v3/phase_c_four_phrasings/cosmos_live_bridge.py"
    ).read_text()
    assert 'if server_seed != bridge["seed"]:' in source
    assert "server did not echo the registered Phase-C sampling seed" in source
    overlay = (
        ROOT / "experiments/v3/phase_c_four_phrasings/serve_phase_c_cosmos.py"
    ).read_text()
    assert "with _request_lock:" in overlay
    assert 'seed = obs.get("sampling_seed")' in overlay
    assert '"sampling_seed": seed' in overlay


def test_committed_materialization_matches_builder() -> None:
    committed = ROOT / OUTPUT_RELATIVE
    cells = [json.loads(line) for line in (committed / "phase_c_v3c001_cells.jsonl").read_text().splitlines()]
    validate_cells(cells)
    manifest = json.loads((committed / "phase_c_v3c001_manifest.json").read_text())
    assert manifest["counts"]["behavioral_cells"] == 480
    assert manifest["behavioral_release"] is False
