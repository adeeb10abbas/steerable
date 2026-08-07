import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001/results/compiled_results.json"


def test_e001_has_all_exact_repeat_comparisons_and_real_vectors():
    report = json.loads(REPORT.read_text())
    assert report["exact_repeat_comparison_count"] == 12
    assert report["exact_repeat_summary"]["all_bit_identical"] is True
    assert all(x["numerical_rms"] == 0.0 for x in report["exact_repeat_comparisons"])
    for row in report["metrics"].values():
        assert row["model_request_rows"] == 56
        native = row["native_full_returned_action_chunk"]
        assert len(native["per_dimension_rms_mean"]) > 1
        assert len(native["per_dimension_rms_by_seed"]) == 27
        assert row["layout_interaction"]["paired_bootstrap_95_ci"]["replicates"] == 20000


def test_e001_invalid_attempts_are_deduplicated_and_provenanced():
    report = json.loads(REPORT.read_text())
    assert report["raw_invalid_row_count"] >= report["unique_invalid_attempt_count"]
    assert report["duplicate_invalid_row_count"] == report["raw_invalid_row_count"] - report["unique_invalid_attempt_count"]
    assert report["unique_invalid_source_hash_count"] >= 1
    assert all(len(source["sha256"]) == 64 for source in report["source_files"])
