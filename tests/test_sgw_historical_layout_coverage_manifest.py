from experiments.workshops.spatial_grounding_v1.historical_layout_coverage_manifest import (
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
            "locally_recoverable",
            "requires_named_hash_anchored_pvc_payloads",
        )
    ) == 45
    assert result["release_authorization"] is False


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
    assert result["prospective_neutral_centers"]["conservative_exclusion_candidates"] == []
