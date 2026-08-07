from __future__ import annotations

import json
from pathlib import Path

from tools.release_pi05_v3d001_stochastic_queue import (
    PHASE_D_SHA256,
    PROMPTS,
    REGISTRATION_SHA256,
    SCOPE_CORRECTION_SHA256,
    build,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


class Args:
    study_root: Path
    eligibility_report: Path
    eligibility_manifest: Path
    output_dir: Path


def test_corrected_phase_d_queue_is_27_by_2_by_8(tmp_path: Path) -> None:
    report = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-eligibility-result-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "registration_id": "V3-D001",
        "model_id": "pi05_current_stack_droid",
        "status": "eligible_effective_sampling_seed",
        "passed": True,
        "model_request_count": 32,
        "behavioral_episode_count": 0,
        "registration_sha256": REGISTRATION_SHA256,
        "phase_d_registry_sha256": PHASE_D_SHA256,
        "scope_correction_sha256": SCOPE_CORRECTION_SHA256,
        "sampling_seed_indices": list(range(8)),
        "exact_prompts": PROMPTS,
        "direction_metrics": {"left": {"passed": True}, "right": {"passed": True}},
    }
    report_path = tmp_path / "eligibility_report.json"
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-eligibility-manifest-v1",
        "status": "eligible_effective_sampling_seed",
        "model_request_count": 32,
        "behavioral_episode_count": 0,
        "files": [{"path": str(report_path.resolve()), "sha256": sha256_file(report_path)}],
    }
    manifest_path = tmp_path / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    args = Args()
    args.study_root = ROOT
    args.eligibility_report = report_path
    args.eligibility_manifest = manifest_path
    args.output_dir = tmp_path / "release"
    result = build(args)
    rows = [json.loads(line) for line in Path(result["queue"]).read_text().splitlines()]
    assert len(rows) == 432
    assert len({row["cell_id"] for row in rows}) == 432
    assert {row["environment_seed"] for row in rows} == set(range(8303, 8330))
    assert {row["shared_policy_sampling_seed_index"] for row in rows} == set(range(8))
    assert {row["requested_relation"] for row in rows} == {"left", "right"}
    assert all(row["behavioral_status"] == "authorized_not_launched" for row in rows)
