from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.droid_behavioral_contract import (
    MODEL_SPECS,
    bind_runtime_identity,
    model_spec,
    simulator_export_envelope,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.episode_compiler import EXPORT_SCHEMA
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (
    RuntimeContractError,
    load_runtime_bundle,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


def _bundle():
    registration = ARTIFACT_ROOT / "registration.json"
    queue = ARTIFACT_ROOT / "queue.jsonl"
    candidate = ARTIFACT_ROOT / "layout/candidate.json"
    return load_runtime_bundle(
        registration_path=registration,
        registration_sha256=sha256_file(registration),
        queue_path=queue,
        queue_sha256=sha256_file(queue),
        candidate_path=candidate,
        candidate_sha256=sha256_file(candidate),
    )


@pytest.mark.parametrize(
    ("cell_id", "port", "policy_id"),
    (
        ("v3e004:pi05:seed9400:s100:left", 8001, "pi05_v2a010_current"),
        ("v3e004:nano:seed9400:s100:left", 18011, "cosmos3_nano_v2"),
        ("v3e004:dreamzero:seed9400:s100:left", 5101, "dreamzero_v2"),
        ("v3e004:edge:seed9400:s100:left", 18010, "cosmos3_v2"),
    ),
)
def test_model_dispatch_is_bound_to_registered_runtime(cell_id: str, port: int, policy_id: str) -> None:
    cell = _bundle().cell(cell_id)
    spec = model_spec(cell, endpoint_port=port)
    assert spec.policy_id == policy_id
    assert spec.action_dim == 8
    assert spec.action_cap == 450


def test_cosmos_port_mismatch_fails_closed() -> None:
    cell = _bundle().cell("v3e004:nano:seed9400:s100:left")
    with pytest.raises(RuntimeContractError, match="endpoint port"):
        model_spec(cell, endpoint_port=18010)


def test_runtime_wrapper_binds_source_lane_and_cell(tmp_path: Path) -> None:
    bundle = _bundle()
    cell = bundle.cell("v3e004:pi05:seed9400:s100:left")
    source = tmp_path / "source_runtime.json"
    source.write_text(json.dumps({"model_id": cell.model_id}) + "\n", encoding="utf-8")
    lane = tmp_path / "lane_release.json"
    lane.write_text(json.dumps({"passed": True}) + "\n", encoding="utf-8")
    output = tmp_path / "bound_runtime.json"
    value = bind_runtime_identity(
        cell=cell,
        bundle=bundle,
        source_path=source,
        source_expected_sha256=sha256_file(source),
        lane_release_path=lane,
        lane_release_sha256=sha256_file(lane),
        output_path=output,
    )
    assert value["registered_cell_sha256"] == cell.row_sha256
    assert value["runtime_identity_requirement"] == cell.row["runtime_identity_requirement"]
    assert json.loads(output.read_text(encoding="utf-8")) == value


def test_canonical_export_preserves_exact_prompt_and_raw_fields() -> None:
    bundle = _bundle()
    cell = bundle.cell("v3e004:nano:seed9400:s100:right")
    steps = [
        {
            "action_step": 0,
            "object_xyz": [0.0, 0.0, 0.1],
            "reference_xyz": [0.1, 0.0, 0.1],
            "grippers_open": True,
            "object_grabbed": False,
        },
        {
            "action_step": 1,
            "object_xyz": [0.0, -0.1, 0.1],
            "reference_xyz": [0.1, 0.0, 0.1],
            "grippers_open": True,
            "object_grabbed": False,
        },
    ]
    export = simulator_export_envelope(
        cell=cell,
        bundle=bundle,
        steps=steps,
        requested_success=False,
        right_censored=False,
        final_detached_release=True,
        live_gate={"path": "/tmp/gate", "sha256": "0" * 64, "bytes": 1},
        runtime_identity={"path": "/tmp/runtime", "sha256": "1" * 64, "bytes": 1},
        executed_action_trace={"path": "/tmp/actions", "sha256": "2" * 64, "bytes": 1},
        viewport_video={"path": "/tmp/video", "sha256": "3" * 64, "bytes": 1},
        future_evidence=None,
        future_evidence_status="exposed_and_retained",
    )
    assert export["schema_version"] == EXPORT_SCHEMA
    assert export["prompt"] == "Put the Rubik's cube to the right of the bowl."
    assert export["prompt_sha256"] == cell.row["prompt_sha256"]
    assert export["instruction_controller"] == "static_episode_prompt"
    assert export["steps"] == steps
    assert export["actions_executed"] == 1


def test_every_droid_model_has_one_dispatch_spec() -> None:
    bundle = _bundle()
    assert {cell.model_id for cell in bundle.cells if cell.row["arena"] == "droid_robolab"} == set(MODEL_SPECS)
