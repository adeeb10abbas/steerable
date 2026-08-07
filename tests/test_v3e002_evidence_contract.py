import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/reference_controller_symmetry_v3e002"


def test_e002_results_are_complete_and_all_failures_are_classified():
    results = json.loads((BASE / "results.json").read_text())
    assert results["behavioral_episode_count"] == 108
    assert results["model_request_count"] == 0
    assert results["infrastructure_invalid_count"] == 0
    assert results["gate"]["selected_depth_m"] == 0.1
    for row in results["cells"].values():
        assert row["episodes"] == 27
        assert row["failure_categories"]["pick_failed"] == 27


def test_e002_manifest_binds_the_actual_runner():
    manifest = json.loads((BASE / "evidence_manifest.json").read_text())
    runner = manifest["execution_provenance"]["runner"]
    assert runner["path"] == "experiments/v3/phase_e/reference_controller_runner.py"
    assert len(runner["sha256"]) == 64
    assert runner["model_requests"] == 0
    assert len(manifest["execution_provenance"]["lane_invocations"]) == 4
