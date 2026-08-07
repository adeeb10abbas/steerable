#!/usr/bin/env python3
"""Fail-closed validator for the hash-closed V3-E001 post-processing report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001"
REG = BASE / "registration.json"
REPORT = BASE / "results/compiled_results.json"
MEMO = BASE / "DECISION_MEMO.md"
MANIFEST = BASE / "evidence_manifest.json"
MODELS = ("pi05_current_stack_droid", "cosmos3_nano_policy_droid", "dreamzero_droid_action_cfg")
LAYOUTS = ("control", "position_mirrored")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert reg["schema_version"] == "vla-wam-shared-v3e001-registration-v1"
    assert reg["status"] == "registered_before_inference"
    assert reg["design"]["total_model_requests"] == 336
    assert reg["design"]["behavioral_episode_count"] == 0
    assert reg["policy_sampling_seeds"] == list(range(9400, 9427))
    assert set(reg["models"]) == set(MODELS)
    assert REG.is_file() and sha256(REG)
    for item in reg["parent_bindings"]:
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path

    assert REPORT.is_file(), REPORT
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["behavioral_episode_count"] == 0
    assert report["registered_model_request_count"] == 336
    assert report["model_request_count"] == report["valid_record_count"] == 336
    assert report["exact_repeat_comparison_count"] == 12
    repeat = report["exact_repeat_summary"]
    assert repeat == {"all_12_complete": True, "all_shape_equal": True,
                      "all_action_sha256_equal": True, "all_bit_identical": True,
                      "all_rms_zero": True}
    assert report["raw_invalid_row_count"] >= report["unique_invalid_attempt_count"] >= 0
    assert report["duplicate_invalid_row_count"] == report["raw_invalid_row_count"] - report["unique_invalid_attempt_count"]
    assert report["invalid_source_file_count"] == len({x["path"] for x in report["source_files"] if x["sha256"] in {i["source_sha256"] for i in report.get("invalid_attempts", [])}}) if report.get("invalid_attempts") else report["invalid_source_file_count"] >= 0
    assert report["invalid_source_file_count"] >= report["unique_invalid_source_hash_count"] >= 0
    assert len(report["source_files"]) > 0
    assert len({x["path"] for x in report["source_files"]}) == len(report["source_files"])
    for source in report["source_files"]:
        assert isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64
        assert int(source.get("bytes", -1)) >= 0
    assert len(report["exact_repeat_comparisons"]) == 12
    for comparison in report["exact_repeat_comparisons"]:
        assert comparison["sampling_seed"] == 9400
        assert comparison["action_shape_equal"] is True
        assert comparison["action_sha256_equal"] is True
        assert comparison["np_array_equal_bit_identity"] is True
        assert comparison["numerical_rms"] == 0.0

    for model in MODELS:
        for layout in LAYOUTS:
            row = report["metrics"][f"{model}/{layout}"]
            assert row["status"] == "complete"
            assert row["model_request_rows"] == 56
            assert row["base_request_rows"] == 54
            assert row["exact_repeat_request_rows"] == 2
            native = row["native_full_returned_action_chunk"]
            assert native["status"] == "available"
            assert len(native["matched_prompt_effect_rms"]) == 27
            assert len(native["per_dimension_rms_by_seed"]) == 27
            assert len(native["per_dimension_rms_mean"]) > 0
            assert native["paired_systematic_distribution_shift"]["permutation_replicates"] >= 100000
            assert row["executable_prefix"]["status"] in {"available", "unavailable"}
            if row["executable_prefix"]["status"] == "unavailable":
                assert row["executable_prefix"].get("reason")
            assert row["semantic_fk"]["status"] in {"available", "unavailable"}
            assert row["semantic_fk"].get("reason") or row["semantic_fk"]["status"] == "available"
            interaction = row["layout_interaction"]
            assert len(interaction["all_27_seed_effects"]) == 27
            assert interaction["paired_bootstrap_95_ci"]["replicates"] == 20000
            assert "exact_two_sided_sign_test" in interaction

    assert MEMO.is_file() and MANIFEST.is_file()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["amendment_id"] == "V3-E001"
    expected = {item["path"]: item for item in manifest["files"]}
    for path in (REG, REPORT, MEMO):
        rel = str(path.relative_to(ROOT))
        assert rel in expected, rel
        assert expected[rel]["bytes"] == path.stat().st_size
        assert expected[rel]["sha256"] == sha256(path)
    print(json.dumps({"status": "valid", "registration_sha256": sha256(REG),
                      "compiled_results_sha256": sha256(REPORT), "requests": 336,
                      "exact_repeat_comparisons": 12,
                      "raw_invalid_rows": report["raw_invalid_row_count"],
                      "unique_invalid_attempts": report["unique_invalid_attempt_count"],
                      "behavioral_episodes": 0}, indent=2))


if __name__ == "__main__":
    main()
