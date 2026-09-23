import hashlib
import io
import json
import tarfile

import pytest

from experiments.workshops.spatial_grounding_v1.historical_layout_coverage_manifest import (
    _probe_archive,
    compile_manifest,
)


def test_manifest_accounts_for_all_45_inventory_sources():
    result = compile_manifest()
    assert result["counts"]["total"] == 45
    assert result["counts"]["arena_excluded"] == 0
    assert result["counts"]["comparable_geometry_eligible"] == 0
    assert sum(
        result["counts"][key]
        for key in (
            "already_covered",
            "source_inspection_needed",
            "requires_named_hash_anchored_pvc_payloads",
        )
    ) == 45
    assert result["release_authorization"] is False
    assert result["counts"]["already_covered"] == 0
    assert result["counts"]["recovered_root_only_rows"] == 4
    assert result["counts"]["recovered_geometry_rows"] == 0
    assert all(
        record["source_hash_status"] == "verified_against_inventory"
        for record in result["records"]
    )
    assert all(record["inspection_conclusion"] for record in result["records"])
    assert result["producer_semantics"]["V3-A-phase-a-groot"]["pinned_robolab_commit"] == (
        "0aef241fb088ca21bb4ebd24448940ed56620d17"
    )
    assert result["producer_semantics"]["V3-A-phase-a-groot"]["cohort_wide_exclusion_proven"] is False


def test_payload_requests_are_source_anchored_and_centers_not_promoted():
    result = compile_manifest()
    requests = [
        payload
        for record in result["records"]
        for payload in record["required_pvc_payloads"]
    ]
    assert requests
    assert all(payload["source_json_pointer"].startswith("/") for payload in requests)
    assert all(
        payload["expected_sha256"] is None
        or len(payload["expected_sha256"]) == 64
        for payload in requests
    )
    assert all(
        payload["hash_binding_status"]
        in {"hash_anchored", "unanchored_path_only"}
        for payload in requests
    )
    assert result["prospective_neutral_centers"]["conservative_exclusion_candidates"] == []
    assert all(
        row["semantic_status"] == "measured_root_only"
        and row["blocker"]
        for row in result["recovered_root_only_evidence"]
    )
    assert all(
        request["request_scope"] == "exact_file_only"
        for request in result["indispensable_external_requests"]
    )
    for request in result["indispensable_external_requests"]:
        if request["selection_reason"].startswith("minimal_semantics_probe"):
            continue
        assert result["external_probe_results"]["status"] == "loaded_hash_verified"
        assert request["source_json_pointer"] == "/child_report"
        assert any(
            probe["cohort_id"] == request["cohort_id"]
            and probe.get("state_payload_request", {}).get("path") == request["path"]
            and probe["state_payload_request"]["sha256"] == request["expected_sha256"]
            for probe in result["external_probe_results"]["probes"]
        )


@pytest.mark.parametrize("corrupt,wrong_source", [(True, False), (False, True), (True, True), (False, False)])
def test_probe_binding_verifies_payload_and_source_independently(tmp_path, corrupt, wrong_source):
    payload = b'{"action_step":0,"object_xyz":[1,2,3],"reference_xyz":[1,2,3]}'
    manifest = {
        "job_uid": "test",
        "model_requests": 0,
        "records": [{
            "cohort_id": "V3-A-phase-a-groot",
            "export_path": "00-payload.jsonl",
            "path": "/data/wrong/source.jsonl" if wrong_source else "/data/expected/source.jsonl",
            "expected_sha256": hashlib.sha256(payload).hexdigest(),
            "actual_sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }
    archive = tmp_path / "probe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        manifest_bytes = json.dumps(manifest).encode()
        actual_payload = payload + b"x" if corrupt else payload
        for name, data in (("manifest.json", manifest_bytes), ("00-payload.jsonl", actual_payload)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    result = _probe_archive(
        archive,
        {("V3-A-phase-a-groot", "/data/expected/source.jsonl", manifest["records"][0]["expected_sha256"])},
    )
    if corrupt or wrong_source:
        assert result["status"] == "rejected_payload_binding"
        assert result["errors"]
        assert result["probes"] == []
    else:
        assert result["status"] == "loaded_hash_verified"
        assert result["errors"] == []
        assert len(result["probes"]) == 1


def test_explicit_missing_archive_does_not_silently_disappear(tmp_path):
    with pytest.raises(FileNotFoundError):
        _probe_archive(tmp_path / "missing.tar.gz", set())
