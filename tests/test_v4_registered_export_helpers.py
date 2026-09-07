"""Tests for registered export helper utilities."""

from tools.v4_registered_export_helpers import (
    build_coverage_metadata,
    summarize_outcome_composition,
)


def test_summarize_outcome_composition_preserves_labels() -> None:
    rows = [
        {"success": False, "outcome": {"failure_label": "no_grasp"}},
        {"success": False, "outcome": {"failure_label": "transport_incomplete"}},
        {"success": True},
    ]
    assert summarize_outcome_composition(rows) == {
        "success": 1,
        "no_grasp": 1,
        "transport_incomplete": 1,
    }


def test_build_coverage_metadata_partial() -> None:
    payload = build_coverage_metadata(accepted=548, planned=768, compile_id="20260908k")
    assert payload["export_status"] == "partial"
    assert payload["coverage_label"] == "548/768 (71.4%)"
