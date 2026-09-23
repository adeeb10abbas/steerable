import json
from pathlib import Path

from experiments.workshops.spatial_grounding_v1.historical_layout_inventory import (
    _index,
    _path_evidence,
    build_inventory,
)


def test_inventory_is_additive_incomplete_and_includes_b002_sources():
    inventory = build_inventory()
    assert inventory["schema_version"].endswith("-v2")
    assert inventory["coverage_status"] == "incomplete_unresolved_historical_layout_coverage"
    assert inventory["release_authorization"] is False
    assert "robotwin" in inventory["arena_exclusion"]
    ids = {cohort["cohort_id"] for cohort in inventory["cohorts"]}
    assert "V3-B002-pi05-position-reflection" in ids
    assert "V3-B002-pi05-release-gate" in ids
    assert inventory["unresolved_coverage"]
    path = Path(__file__).parents[1] / (
        "artifacts/workshops/spatial_grounding_v1/infrastructure/"
        "historical-droid-layout-source-inventory-v2.json"
    )
    assert json.loads(path.read_text()) == inventory


def test_json_pointer_index_uses_rfc6901_escaping_and_bounded_hash_lookup():
    index = _index({"a/b": {"x~y": "value"}})
    assert index["/a~1b/x~0y"] == "value"
    inventory = build_inventory()
    assert all(cohort["layout_ids"] == [] for cohort in inventory["cohorts"])
    assert all("source_sha256" in cohort for cohort in inventory["cohorts"])
    paths = _path_evidence(_index({
        "a/b": {"path": "/data/droid/reset.json", "sha256": "f" * 64},
        "other_arena": {"path": "/data/robotwin/reset.json", "sha256": "e" * 64},
    }))
    assert paths == [{
        "path": "/data/droid/reset.json",
        "source_json_pointer": "/a~1b/path",
        "expected_sha256": "f" * 64,
    }]
