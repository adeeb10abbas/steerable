import hashlib
import json

from experiments.workshops.spatial_grounding_v1.compile import (
    bootstrap_mean,
    censoring_bounds,
    compile_manifests,
    equivalence,
    compile_registered_queue,
    compile_primary_statistics,
    render_neutral_coverage_table,
    render_paper_export_plan,
    marginal_sign_bounds,
    holm_adjust,
    holm_adjust_primary,
    paired_signflip,
)


def _manifest(tmp_path, *, result, release_hash="release"):
    root = tmp_path / f"release-{len(list(tmp_path.glob('release-*')))}"
    directory = root / "attempts" / "cell-1" / "attempt-1"
    directory.mkdir(parents=True)
    artifact = directory / "result.json"
    artifact.write_text(json.dumps(result, sort_keys=True))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({
        "complete": True,
        "release_hashes": {"release": release_hash},
        "artifacts": {"result.json": {"sha256": digest}},
        "result": result,
    }))
    pointer = root / "cells" / "cell-1.complete.json"
    pointer.parent.mkdir()
    pointer.write_text(json.dumps({
        "release_id": "release-1",
        "cell_id": "cell-1",
        "attempt_id": "attempt-1",
        "result": {"path": "attempts/cell-1/attempt-1/result.json", "sha256": digest},
        "manifest_path": "attempts/cell-1/attempt-1/manifest.json",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }))
    return pointer


def test_compile_accepts_only_hash_bound_complete_manifests(tmp_path):
    first = _manifest(tmp_path, result={"status": "valid_success", "S": 1})
    second = _manifest(tmp_path, result={"status": "technical_invalid", "technical_cause": "renderer"})
    compiled = compile_manifests([first, second], expected_release_hashes={"release": "release"})
    assert len(compiled.rows) == 2
    assert len(compiled.valid_rows) == 1
    assert compiled.technical_missing == 1


def test_reproducible_statistics_and_frozen_bounds():
    assert bootstrap_mean([1.0, 2.0, 3.0], draws=100, seed=1) == bootstrap_mean([1.0, 2.0, 3.0], draws=100, seed=1)
    assert paired_signflip([1.0, -1.0], draws=100, seed=1) == paired_signflip([1.0, -1.0], draws=100, seed=1)
    assert censoring_bounds([0.0, 1.0], 1, lower=-1.0, upper=1.0) == (0.0, 2 / 3)
    assert equivalence((-0.05, 0.05), (-0.01, 0.01))


def test_holm_adjustment_is_monotone():
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.8})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_primary_holm_always_accounts_for_six_tests():
    adjusted = holm_adjust_primary({"N3-LAT": 0.01})
    assert set(adjusted) == {"N3-LAT", "N3-HEIGHT", "N3-DIST", "D1-LAT", "D1-HEIGHT", "D1-DIST"}


def test_registered_queue_preserves_unrun_cells_and_neutral_output():
    queue = "experiments/workshops/spatial_grounding_v1/spec/planned_cells.csv"
    compiled = compile_registered_queue(queue, [], expected_release_hashes={})
    assert len(compiled.ledger) == 1044
    assert compiled.complete is False
    assert len(compiled.missing_cell_ids) == 1044
    assert "not_run" in render_neutral_coverage_table(compiled)
    assert compiled.confirmation_estimates
    assert all(row["status"] == "incomplete" for row in compiled.confirmation_estimates)
    assert all(row["status"] == "unavailable" for row in compile_primary_statistics(compiled))
    plan = render_paper_export_plan(compiled)
    assert plan["status"] == "not_run_or_incomplete"
    assert plan["figures"] == []


def test_sign_bounds_and_strict_ninety_percent_equivalence():
    assert marginal_sign_bounds([1.0], 1, lower=-1.0, upper=1.0)["sign"] == "undetermined"
    assert marginal_sign_bounds([2.0], 0, lower=-1.0, upper=1.0)["sign"] == "positive"
    assert not equivalence((-0.10, 0.05), (-0.01, 0.01))
    assert not equivalence((-0.01, 0.01), (-0.02, 0.02))
    assert equivalence((-0.09, 0.09), (-0.019, 0.019))
