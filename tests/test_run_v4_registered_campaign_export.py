"""Tests for registered V4 campaign export helpers."""

from __future__ import annotations

from tools.run_v4_registered_campaign_export import (
    build_campaign_blocked_scope,
    build_campaign_tables,
    render_campaign_scope_figure,
)
from tools.v4_registered_export_helpers import build_coverage_metadata


def test_build_campaign_blocked_scope_includes_horizontal_squeeze() -> None:
    payload = build_campaign_blocked_scope(
        c7_blocked_scope={
            "campaign_scope_revision": {
                "original_planned_policy_episodes": 17664,
                "achievable_policy_episodes": 2304,
                "scientifically_blocked_episodes": 15360,
                "pre_repair_c7_excluded_episodes": 279,
            },
            "not_estimable_or_blocked": {"C2": "blocked"},
        },
        horizontal_blocked_scope={
            "scientific_finding": {
                "classification": "information_gate_squeeze",
                "summary": "No ladder scale satisfies both constraints.",
                "scale_ladder_rejections": [
                    {"scale": 2.0, "binding_constraint": "empty_goals"},
                    {"scale": 0.5, "binding_constraint": "information_gate"},
                ],
                "information_gate_squeeze_at_0p5": {"removed_area_pct_range": [13.7, 19.9]},
            }
        },
        family_status={
            "C7": {
                **build_coverage_metadata(accepted=548, planned=768, compile_id="20260908k"),
                "outcome_composition": {"no_grasp": 544, "transport_incomplete": 4},
            }
        },
    )
    assert payload["scientifically_blocked_episodes"] == 15360
    assert payload["pre_repair_c7_excluded_episodes"] == 279
    assert payload["horizontal_scale_squeeze"]["classification"] == "information_gate_squeeze"
    assert payload["criteria_amended"] is False
    assert "C1" in payload["not_estimable_or_blocked"]


def test_build_campaign_tables_and_figure(tmp_path) -> None:
    campaign_blocked = {
        "original_planned_policy_episodes": 17664,
        "achievable_policy_episodes": 2304,
        "scientifically_blocked_episodes": 15360,
        "pre_repair_c7_excluded_episodes": 279,
        "blocked_breakdown": {
            "C1_C3_C4_horizontal": 9728,
            "C2_reference_binding": 4096,
            "C5_vertical": 768,
        },
        "not_estimable_or_blocked": {"C1": "blocked", "C2": "blocked", "C5": "blocked"},
        "horizontal_scale_squeeze": {
            "scale_ladder_rejections": [{"scale": 0.5, "binding_constraint": "information_gate", "detail": "64/128"}]
        },
    }
    family_rollups = {
        "C7": {
            "coverage": build_coverage_metadata(accepted=548, planned=768, compile_id="20260908k"),
            "outcome_composition": {"no_grasp": 544, "transport_incomplete": 4},
            "fixture": "object_pair",
            "platform": "isaac_droid",
            "evidence_phase": "confirmatory_partial",
        }
    }
    tables = build_campaign_tables(
        campaign_blocked=campaign_blocked,
        c7_audit={"validation": {"accepted_unique": 548, "valid_success_records": 0, "valid_failure_records": 548}},
        c7_primary_rows=[{"estimand_id": "H1", "status": "not_estimable"}],
        family_rollups=family_rollups,
        campaign_export_status="partial",
    )
    assert len(tables["scope_summary.csv"]) >= 10
    assert tables["c7_outcome_composition.csv"] == [
        {"failure_label": "no_grasp", "count": 544},
        {"failure_label": "transport_incomplete", "count": 4},
    ]
    assert tables["blocked_families.csv"][0]["blocked_episodes"] == 9728
    figure_path = tmp_path / "campaign_scope.svg"
    render_campaign_scope_figure(rows=tables["scope_summary.csv"], out_path=figure_path, export_status="partial")
    assert figure_path.is_file()
    assert "V4 registered campaign scope" in figure_path.read_text(encoding="utf-8")
