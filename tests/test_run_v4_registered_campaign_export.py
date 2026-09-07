"""Tests for registered V4 campaign export helpers."""

from __future__ import annotations

from tools.run_v4_registered_campaign_export import (
    build_campaign_blocked_scope,
    build_campaign_tables,
    render_campaign_scope_figure,
)


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
        horizontal_evidence_memo={},
        c7_ledger_rows=541,
        c7_planned=768,
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
    tables = build_campaign_tables(
        campaign_blocked=campaign_blocked,
        c7_audit={"validation": {"accepted_unique": 541, "valid_success_records": 0, "valid_failure_records": 541}},
        c7_primary_rows=[{"estimand_id": "H1", "status": "not_estimable"}],
    )
    assert len(tables["scope_summary.csv"]) == 8
    assert tables["blocked_families.csv"][0]["blocked_episodes"] == 9728
    figure_path = tmp_path / "campaign_scope.svg"
    render_campaign_scope_figure(rows=tables["scope_summary.csv"], out_path=figure_path)
    assert figure_path.is_file()
    assert "V4 registered campaign scope" in figure_path.read_text(encoding="utf-8")
