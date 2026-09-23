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
    assert all(
        request["selection_reason"].startswith("minimal_semantics_probe")
        for request in result["indispensable_external_requests"]
    )
