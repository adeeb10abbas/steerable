#!/usr/bin/env python3
"""Fail-closed validator for V3-E002 results and provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/reference_controller_symmetry_v3e002"
REG = BASE / "registration.json"
GATE = BASE / "model_blind_gate/ik_gate_control_final.json"
RESULTS = BASE / "results.json"
MEMO = BASE / "DECISION_MEMO.md"
MANIFEST = BASE / "evidence_manifest.json"
CELLS = ("control:left", "control:right", "position_mirrored:left", "position_mirrored:right")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert reg["schema_version"] == "vla-wam-shared-v3e002-registration-v1"
    assert reg["status"] == "registered_before_inference"
    assert reg["model_blind"] is True
    assert reg["learned_model_request_count"] == 0
    assert len(reg["queue"]) == 108
    assert len({row["cell_id"] for row in reg["queue"]}) == 108
    for item in reg["parent_bindings"]:
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path

    gate = json.loads(GATE.read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["selected_depth_m"] == 0.1
    assert gate["candidate_sha256"] == "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88"

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert results["status"] == "complete"
    assert results["model_request_count"] == 0
    assert results["behavioral_episode_count"] == 108
    assert results["infrastructure_invalid_count"] == 0
    assert results["gate"]["selected_depth_m"] == 0.1
    result_cells = {key.replace("/", ":"): value for key, value in results["cells"].items()}
    assert set(result_cells) == set(CELLS)
    for cell in CELLS:
        row = result_cells[cell]
        assert row["episodes"] == 27
        assert row["successes"] == 0
        assert row["failure_categories"]["pick_failed"] == 27
        assert sum(row["failure_categories"].values()) == 27
    assert len(results["matched_pairs"]) == 54
    assert all("interaction" in results["interactions"][field] for field in results["interactions"])
    source_entries = {item["path"]: item for item in results["source_files"]}
    assert len(source_entries) == 4
    for rel, item in source_entries.items():
        path = ROOT / rel
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "vla-wam-shared-v3e002-evidence-manifest-v2"
    provenance = manifest["execution_provenance"]
    runner = provenance["runner"]
    runner_path = ROOT / runner["path"]
    assert runner_path.is_file()
    assert runner["bytes"] == runner_path.stat().st_size
    assert runner["sha256"] == sha256(runner_path)
    assert runner["model_requests"] == 0
    assert provenance["runtime"]["robolab_commit"] == "0aef241fb088ca21bb4ebd24448940ed56620d17"
    assert provenance["runtime"]["selected_depth_m"] == 0.1
    assert len(provenance["lane_invocations"]) == 4
    assert {item["condition"] for item in provenance["lane_invocations"]} == set(CELLS)

    memo = MEMO.read_text(encoding="utf-8")
    assert "108 valid episodes" in memo
    assert "zero learned-model requests" in memo
    assert "pick_failed" in memo and "0/27 each" in memo
    assert "0.100 m" in memo

    expected = {item["path"]: item for item in manifest["files"]}
    for path in (REG, GATE, RESULTS, MEMO):
        rel = str(path.relative_to(ROOT))
        assert rel in expected, rel
        assert expected[rel]["bytes"] == path.stat().st_size
        assert expected[rel]["sha256"] == sha256(path)
    for path in BASE.glob("episodes/*.jsonl"):
        rel = str(path.relative_to(ROOT))
        assert rel in expected
        assert expected[rel]["bytes"] == path.stat().st_size
        assert expected[rel]["sha256"] == sha256(path)

    print(json.dumps({"status": "valid", "registration_sha256": sha256(REG),
                      "results_sha256": sha256(RESULTS), "behavioral_episodes": 108,
                      "learned_model_requests": 0, "selected_depth_m": 0.1}, indent=2))


if __name__ == "__main__":
    main()
