from experiments.workshops.spatial_grounding_v1.historical_layout_coverage import (
    compile_coverage,
)


def test_additional_sources_are_verified_but_not_promoted_to_geometry():
    coverage = compile_coverage()
    assert coverage["coverage_status"].startswith("incomplete_")
    assert coverage["release_authorization"] is False
    assert coverage["configured_root_only_rows"]
    assert coverage["comparable_geometry_rows"] == []
    assert any(item["source_id"] == "v3e005-scene-candidate" for item in coverage["blockers"])
    assert all(source["git_blob_sha1"] for source in coverage["sources"])


def test_root_only_rows_require_explicit_coordinates():
    coverage = compile_coverage()
    for row in coverage["configured_root_only_rows"]:
        assert len(row["configured_position_robot_base_m"]) == 3
        assert row["status"] == "configured_root_only"
        assert "quaternion" in row["blocker"]
