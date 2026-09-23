import hashlib
import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.main_p_fixture_assignment import (
    FIXTURE_ID, build_assignment, materialize_release_fixture, write_assignment,
)


QUEUE = Path("experiments/workshops/spatial_grounding_v1/spec/planned_cells.csv")


def _evidence(path, *, passes=6):
    value = {
        "candidate_id": FIXTURE_ID,
        "status": "verified_all_six_pass",
        "passed_checks": passes,
        "model_requests": 0,
        "behavioral_episodes": 0,
    }
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_assignment_preserves_existing_six_p_cells_and_keeps_release_pending(tmp_path):
    capture = tmp_path / "capture.json"
    capture.write_text("{}")
    proof = tmp_path / "qualification-verification.json"
    assignment = build_assignment(
        planned_queue=QUEUE, translated_capture=capture,
        translated_capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        qualification_verification=proof, qualification_verification_sha256=_evidence(proof),
    )
    cells = assignment["main_p_frozen_queue"]["cells"]
    assert [row["cell_id"] for row in cells] == [
        "LAT-P01-N3-I-POS", "LAT-P01-N3-D-POS", "LAT-P01-N3-C-POS",
        "LAT-P01-N3-C-NEG", "LAT-P01-N3-D-NEG", "LAT-P01-N3-I-NEG",
    ]
    assert assignment["runtime_release"]["release_permitted"] is False
    assert assignment["duplicate_reservation"]["later_d_c_duplicate_within_tolerance_reuse_forbidden"] is True


def test_assignment_rejects_changed_evidence_or_incomplete_scripted_checks(tmp_path):
    capture = tmp_path / "capture.json"
    capture.write_text("{}")
    proof = tmp_path / "qualification-verification.json"
    capture_hash = hashlib.sha256(capture.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="all six"):
        build_assignment(
            planned_queue=QUEUE, translated_capture=capture, translated_capture_sha256=capture_hash,
            qualification_verification=proof, qualification_verification_sha256=_evidence(proof, passes=5),
        )
    proof_hash = _evidence(proof)
    capture.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="translated capture"):
        build_assignment(
            planned_queue=QUEUE, translated_capture=capture, translated_capture_sha256=capture_hash,
            qualification_verification=proof, qualification_verification_sha256=proof_hash,
        )


def test_assignment_is_immutable_on_write(tmp_path):
    capture = tmp_path / "capture.json"
    capture.write_text("{}")
    proof = tmp_path / "qualification-verification.json"
    output = tmp_path / "assignment.json"
    write_assignment(
        output=output, planned_queue=QUEUE, translated_capture=capture,
        translated_capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        qualification_verification=proof, qualification_verification_sha256=_evidence(proof),
    )


def test_materialization_requires_prequalified_time_map_and_keeps_runtime_pending(tmp_path):
    capture = tmp_path / "capture.json"
    capture.write_text("{}")
    proof = tmp_path / "qualification-verification.json"
    assignment_path = tmp_path / "assignment.json"
    write_assignment(
    output=assignment_path, planned_queue=QUEUE, translated_capture=capture,
    translated_capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
    qualification_verification=proof, qualification_verification_sha256=_evidence(proof),
    )
    fixture = tmp_path / "qualified.json"
    fixture.write_text(json.dumps({"status": "qualified", "layouts": {}, "time_maps": {"N3": "t" * 64}}))
    output = tmp_path / "fixtures.json"
    value = materialize_release_fixture(
    assignment_path=assignment_path, qualified_fixture_input=fixture, output=output,
    )
    assert value["layouts"]["LAT-P01"]["fixture_sha256"] == value["main_p_fixture_materialization"]["assignment_sha256"]
    assert value["layouts"]["LAT-P01"]["release_permitted"] is False
    with pytest.raises(FileExistsError):
        write_assignment(
            output=output, planned_queue=QUEUE, translated_capture=capture,
            translated_capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
            qualification_verification=proof, qualification_verification_sha256=_evidence(proof),
        )
