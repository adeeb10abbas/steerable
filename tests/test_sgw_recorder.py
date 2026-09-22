import json
from pathlib import Path

from experiments.workshops.spatial_grounding_v1.contract import load_release
from experiments.workshops.spatial_grounding_v1.recorder import AttemptRecorder, next_attempt_number
from tests.test_sgw_contract import make_release


def test_completion_is_no_overwrite_and_model_failure_is_complete(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))
    cell = release.partition("N3", "LAT", "P")[0]
    first = AttemptRecorder(release, cell, "attempt-001")
    first.begin()
    for directory in ("actions", "states", "observations", "videos"):
        path = first.path / directory / "raw"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retained")
    assert first.complete({"status": "valid_model_failure", "failure_reason": "wrong_side",
                           "executed_action_count": 450, "safety_terminated": False})
    second = AttemptRecorder(release, cell, "attempt-002")
    second.begin()
    for directory in ("actions", "states", "observations", "videos"):
        path = second.path / directory / "raw"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retained")
    assert not second.complete({"status": "valid_success", "executed_action_count": 450,
                                "safety_terminated": False})
    pointer = json.loads((release.root.parent / "cells" / "cell-0.complete.json").read_text())
    assert pointer["attempt_id"] == "attempt-001"


def test_technical_attempts_remain_durable_without_publication(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))
    cell = release.partition("N3", "LAT", "P")[0]
    recorder = AttemptRecorder(release, cell, "attempt-001")
    recorder.begin()
    assert not recorder.complete({"status": "technical_invalid", "technical_cause": "disk full"})
    assert not (release.root.parent / "cells" / "cell-0.complete.json").exists()
    assert next_attempt_number(release, cell) == 2
