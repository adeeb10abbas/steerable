from experiments.workshops.spatial_grounding_v1.historical_layout_dedup import dedup_ledger


def _candidate():
    return {"candidate_id": "LAT-CANDIDATE-001", "object_poses": {
        "rubiks_cube": {"position_m": [0.4, 0, .1], "quaternion_wxyz": [1, 0, 0, 0]},
        "bowl": {"position_m": [.5, 0, .1], "quaternion_wxyz": [1, 0, 0, 0]}},
        "metadata": {"scoring_center_offsets_root_local_m": {"rubiks_cube": [0, .01, 0], "bowl": [0, 0, 0]}}}


def test_dedup_blocks_without_historical_evidence():
    ledger = dedup_ledger({"candidates": [_candidate()] * 100}, None, proposal_sha256="a" * 64)
    assert ledger["release_permitted"] is False
    assert ledger["status"] == "blocked_missing_historical_layout_evidence"


def test_dedup_detects_hash_bound_actor_root_duplicate():
    record = {"layout_id": "old", "sha256": "b" * 64, "object_root_poses": _candidate()["object_poses"],
              "scoring_center_offsets_root_local_m": _candidate()["metadata"]["scoring_center_offsets_root_local_m"]}
    ledger = dedup_ledger({"candidates": [_candidate()] * 100}, {"layouts": [record]}, proposal_sha256="a" * 64)
    assert ledger["status"] == "blocked_historical_duplicate_detected"
    assert len(ledger["matches"]) == 100
