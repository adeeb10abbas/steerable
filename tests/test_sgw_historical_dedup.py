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
    assert ledger["release_permitted"] is False


def test_dedup_blocks_empty_registry_without_claiming_complete_coverage():
    ledger = dedup_ledger({"candidates": [_candidate()] * 100}, {"layouts": []}, proposal_sha256="a" * 64)

    assert ledger["status"] == "blocked_missing_historical_layout_evidence"
    assert ledger["coverage"]["exhaustive_source_verified"] is False
    assert "complete" not in ledger["coverage"]
    assert ledger["release_permitted"] is False


def test_dedup_blocks_arbitrary_nonmatching_subset_even_with_syntactic_binding():
    record = {
        "layout_id": "nonmatching-subset",
        "sha256": "b" * 64,
        "source": {"path": "unverified/history.json", "sha256": "b" * 64},
        "object_root_poses": {
            "rubiks_cube": {"position_m": [0.8, 0, .1], "quaternion_wxyz": [1, 0, 0, 0]},
            "bowl": {"position_m": [.9, 0, .1], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "scoring_center_offsets_root_local_m": _candidate()["metadata"]["scoring_center_offsets_root_local_m"],
    }
    ledger = dedup_ledger({"candidates": [_candidate()] * 100}, {"layouts": [record]}, proposal_sha256="a" * 64)

    assert ledger["status"] == "blocked_unverified_historical_layout_coverage"
    assert ledger["matches"] == []
    assert ledger["release_permitted"] is False


def test_dedup_rejects_malformed_offsets_without_masking_valid_duplicate():
    duplicate = {
        "layout_id": "duplicate",
        "sha256": "b" * 64,
        "object_root_poses": _candidate()["object_poses"],
        "scoring_center_offsets_root_local_m": _candidate()["metadata"]["scoring_center_offsets_root_local_m"],
    }
    malformed = {
        "layout_id": "malformed",
        "sha256": "not-a-sha",
        "source": {"path": "fake.json", "sha256": "not-a-sha"},
        "object_root_poses": _candidate()["object_poses"],
        "scoring_center_offsets_root_local_m": {"rubiks_cube": [float("nan"), 0, 0], "bowl": []},
    }
    ledger = dedup_ledger(
        {"candidates": [_candidate()] * 100},
        {"layouts": [malformed, duplicate]},
        proposal_sha256="a" * 64,
    )

    assert ledger["status"] == "blocked_historical_duplicate_detected"
    assert len(ledger["matches"]) == 100
    assert ledger["release_permitted"] is False
    assert any("complete cube/bowl" in reason for reason in ledger["missing_evidence"])
