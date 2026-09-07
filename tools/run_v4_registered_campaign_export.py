#!/usr/bin/env python3
"""Build the registered V4 campaign export from family ledgers and blocked-scope slices."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_C7_MANIFEST = (
    ROOT / "artifacts/online_correction_v4/setup/c7_confirmatory/queue.frozen.jsonl"
)
HORIZONTAL_SLICE = (
    ROOT / "artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908"
)
FROZEN_ANALYSIS = ROOT / "artifacts/online_correction_v4/frozen_analysis_manifest.json"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_campaign_blocked_scope(
    *,
    c7_blocked_scope: dict,
    horizontal_blocked_scope: dict,
    horizontal_evidence_memo: dict,
    c7_ledger_rows: int,
    c7_planned: int,
) -> dict:
    campaign = dict(c7_blocked_scope.get("campaign_scope_revision") or {})
    horizontal_finding = horizontal_blocked_scope.get("scientific_finding") or {}
    squeeze = horizontal_finding.get("information_gate_squeeze_at_0p5") or {}
    campaign["horizontal_geometry_repair_note"] = horizontal_finding.get(
        "summary",
        campaign.get("horizontal_geometry_repair_note"),
    )
    campaign["horizontal_scale_squeeze"] = {
        "classification": horizontal_finding.get("classification"),
        "scale_ladder_rejections": horizontal_finding.get("scale_ladder_rejections"),
        "information_gate_at_0p5": squeeze,
        "evidence_slice": str(HORIZONTAL_SLICE.relative_to(ROOT)),
    }
    not_estimable = dict(c7_blocked_scope.get("not_estimable_or_blocked") or {})
    not_estimable["C1"] = (
        "scientifically blocked with C3/C4: registered fixture design squeeze — "
        "geometry repair v2 restored physical feasibility (3072/3072 path checks at scale 0.5) "
        "but no ladder scale satisfies both large-displacement goal non-emptiness and the frozen "
        "20% shrinking-area information threshold; scale 0.5 fails information gate on 64/128 "
        "seeds (all sign=-1, removed area 13.7–19.9%); receipt "
        "20260908_horizontal_g3_gate_decision_g3r20260908g.json; "
        "horizontal_geometry_repair_v2/20260908 evidence slice"
    )
    return {
        "schema_version": "v4-registered-campaign-blocked-scope-v1",
        "original_planned_policy_episodes": campaign.get("original_planned_policy_episodes", 17664),
        "achievable_policy_episodes": campaign.get("achievable_policy_episodes", 2304),
        "scientifically_blocked_episodes": campaign.get("scientifically_blocked_episodes", 15360),
        "achievable_breakdown": campaign.get("achievable_breakdown"),
        "blocked_breakdown": campaign.get("blocked_breakdown"),
        "disclosed_setup_repairs": campaign.get("disclosed_setup_repairs"),
        "pre_repair_c7_excluded_episodes": campaign.get("pre_repair_c7_excluded_episodes", 279),
        "horizontal_geometry_repair_note": campaign.get("horizontal_geometry_repair_note"),
        "horizontal_scale_squeeze": campaign.get("horizontal_scale_squeeze"),
        "family_status": {
            "C7": {
                "accepted_ledger_rows": c7_ledger_rows,
                "planned_episodes": c7_planned,
                "pre_repair_excluded": campaign.get("pre_repair_c7_excluded_episodes", 279),
            },
            "C6": {
                "planned_episodes": 768,
                "status": "achievable; confirmatory dispatch in progress (Agent B)",
            },
            "C8": {
                "planned_episodes": 768,
                "status": "achievable; G7 pilot dispatched; confirmatory after G8 (Agent A)",
            },
        },
        "not_estimable_or_blocked": not_estimable,
        "scientific_blockers": c7_blocked_scope.get("scientific_blockers"),
        "horizontal_evidence": {
            "manifest": "artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908/evidence_manifest.json",
            "blocked_scope": "artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908/blocked_scope.json",
            "evidence_memo": "artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908/paper/evidence_memo.json",
        },
        "criteria_amended": False,
    }


def build_campaign_tables(
    *,
    campaign_blocked: dict,
    c7_audit: dict,
    c7_primary_rows: list[dict],
) -> dict[str, list[dict]]:
    validation = c7_audit.get("validation") or {}
    scope_summary = [
        {"metric": "original_planned_policy_episodes", "value": campaign_blocked.get("original_planned_policy_episodes", 17664)},
        {"metric": "achievable_policy_episodes", "value": campaign_blocked.get("achievable_policy_episodes", 2304)},
        {"metric": "scientifically_blocked_episodes", "value": campaign_blocked.get("scientifically_blocked_episodes", 15360)},
        {"metric": "pre_repair_c7_excluded_episodes", "value": campaign_blocked.get("pre_repair_c7_excluded_episodes", 279)},
        {"metric": "c7_accepted_valid_unique", "value": validation.get("accepted_unique")},
        {"metric": "c7_valid_success_records", "value": validation.get("valid_success_records")},
        {"metric": "c7_valid_failure_records", "value": validation.get("valid_failure_records")},
        {"metric": "criteria_amended", "value": campaign_blocked.get("criteria_amended", False)},
    ]
    breakdown = campaign_blocked.get("blocked_breakdown") or {}
    not_estimable = campaign_blocked.get("not_estimable_or_blocked") or {}
    blocked_families = []
    for block_key, families in (
        ("C1_C3_C4_horizontal", ("C1", "C3", "C4")),
        ("C2_reference_binding", ("C2",)),
        ("C5_vertical", ("C5",)),
    ):
        blocked_families.append(
            {
                "block_class": block_key,
                "blocked_episodes": breakdown.get(block_key),
                "families": ",".join(families),
                "status": not_estimable.get(families[0], ""),
            }
        )
    squeeze_rows = [
        {
            "scale": item.get("scale"),
            "binding_constraint": item.get("binding_constraint"),
            "detail": item.get("detail"),
        }
        for item in (campaign_blocked.get("horizontal_scale_squeeze") or {}).get(
            "scale_ladder_rejections"
        )
        or []
    ]
    estimand_status = [{**row, "analysis_scope": "registered_primary_contrast_registry"} for row in c7_primary_rows]
    return {
        "scope_summary.csv": scope_summary,
        "blocked_families.csv": blocked_families,
        "scale_ladder_squeeze.csv": squeeze_rows,
        "campaign_primary_results.csv": estimand_status,
    }


def render_campaign_scope_figure(*, rows: list[dict], out_path: Path) -> None:
    planned = next((row["value"] for row in rows if row["metric"] == "original_planned_policy_episodes"), 17664)
    achievable = next((row["value"] for row in rows if row["metric"] == "achievable_policy_episodes"), 2304)
    blocked = next((row["value"] for row in rows if row["metric"] == "scientifically_blocked_episodes"), 15360)
    excluded = next((row["value"] for row in rows if row["metric"] == "pre_repair_c7_excluded_episodes"), 279)
    width = 720
    height = 220
    total = max(int(planned), 1)
    bar_h = 36
    y0 = 80
    x = 40
    parts = []
    for label, value, color in (
        ("achievable", achievable, "#2a9d8f"),
        ("blocked", blocked, "#e76f51"),
        ("pre-repair excluded (C7)", excluded, "#f4a261"),
    ):
        seg_w = int((width - 80) * (int(value) / total))
        parts.append(f'<rect x="{x}" y="{y0}" width="{seg_w}" height="{bar_h}" fill="{color}" />')
        parts.append(f'<text x="{x + 4}" y="{y0 + 22}" font-size="12" fill="#111">{label}: {value}</text>')
        x += seg_w
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<text x="20" y="28" font-size="16" font-weight="600">V4 registered campaign scope</text>'
        f'<text x="20" y="52" font-size="12" fill="#444">Planned policy episodes: {planned}</text>'
        + "".join(parts)
        + "</svg>\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def build_campaign_evidence_memo(
    *,
    c7_memo: dict,
    horizontal_memo: dict,
    c7_export_manifest: dict,
    c7_audit: dict,
) -> dict:
    horizontal_narrative = horizontal_memo.get("paper_narrative") or {}
    paragraphs = list(horizontal_narrative.get("paragraphs") or [])
    validation = c7_audit.get("validation") or {}
    campaign_paragraphs = [
        "Registered campaign scope: 17,664 planned policy episodes; 2,304 achievable "
        "(C6, C7, C8 confirmatory families × 768); 15,360 scientifically blocked "
        "(C1/C3/C4 horizontal information-gate squeeze 9,728; C2 reference_binding "
        "information gate 4,096; C5 vertical IK reachability 768). No eligibility "
        "criterion, threshold, or scale ladder was amended to recover blocked scope.",
        "Three disclosed setup repairs bound this export: (1) NaturalGraspDetector trigger "
        "observation wiring now uses finger-contact coupling instead of robot-base pose; "
        "(2) eef_tool_length_m=0.14 flange-versus-fingerpad offset applied uniformly across "
        "Isaac fixtures; (3) second_stack SimplerEnv observation and sampling defect repaired "
        "with control-boundary sampling and provisioned render libraries.",
        "279 pre-repair C7 episodes are excluded from all behavioral claims; only a ledger "
        "compile against the confirmatory manifest separates repaired-path accepted rows from "
        "infrastructure-invalid pre-repair attempts.",
        *paragraphs,
    ]
    if c7_memo.get("intervention_trigger_positive_control"):
        campaign_paragraphs.append(
            "C7 behavioral evidence under verified repaired NaturalGraspDetector timing: "
            f"{c7_memo['intervention_trigger_positive_control'].get('status')} — "
            f"{c7_memo['intervention_trigger_positive_control'].get('finding')}. "
            f"Compiled ledger: {validation.get('accepted_unique', 'n/a')} accepted valid, "
            f"{validation.get('valid_success_records', 0)} successes, "
            f"{validation.get('valid_failure_records', 0)} valid failures."
        )
    campaign_paragraphs.append(
        "C2 primary reference-selectivity (H) remains not estimable; the homogeneous G3 gate "
        "finalized at 128/128 as a scientific block with computation-correct information-gate "
        "rejection (4096 episodes). C8 G7 engineering pilot is dispatched; confirmatory lock "
        "and 768-episode dispatch follow G8 and Agent A receipt handoff."
    )
    return {
        "schema_version": "v4-registered-campaign-evidence-memo-v1",
        "frozen_analysis_manifest": str(FROZEN_ANALYSIS.relative_to(ROOT)),
        "headline": (
            "Achievable confirmatory scope is 2,304 episodes; 15,360 blocked by registered "
            "fixture geometry with disclosed setup repairs and no criterion amendments"
        ),
        "paper_narrative": {
            "headline": horizontal_narrative.get("headline"),
            "paragraphs": campaign_paragraphs,
        },
        "c7_export": {
            "tag": c7_export_manifest.get("tag"),
            "accepted_ledger_sha256": (c7_export_manifest.get("accepted_ledger") or {}).get("sha256"),
            "accepted_valid_unique": validation.get("accepted_unique"),
            "valid_success_records": validation.get("valid_success_records"),
            "valid_failure_records": validation.get("valid_failure_records"),
        },
        "horizontal_slice": str(HORIZONTAL_SLICE.relative_to(ROOT)),
        "limitations": [
            "Blocked families carry qualification receipts only; no policy episodes were dispatched.",
            "279 pre-repair C7 episodes remain excluded from behavioral claims.",
            "Refresh this export by recompiling the C7 ledger from PVC attempts and re-running this tool.",
            "C6 and C8 confirmatory family tables land when Agent B and Agent A complete dispatch.",
        ],
        "not_estimable_primary_estimands": [
            "C1/C3/C4 primary wording and reference-selectivity contrasts",
            "C2 reference-selectivity primary (H)",
            "C5 vertical family (IK reachability block)",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c7-ledger", type=Path, required=True)
    parser.add_argument("--c7-manifest", type=Path, default=DEFAULT_C7_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/results/registered",
    )
    parser.add_argument("--tag", type=str, default="20260908")
    args = parser.parse_args(argv)

    c7_ledger = args.c7_ledger.resolve()
    if not c7_ledger.is_file():
        raise SystemExit(f"missing C7 accepted ledger: {c7_ledger}")

    out_root = args.out.resolve() / args.tag
    if out_root.exists():
        import shutil

        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    c7_tag = f"{args.tag}c7"
    c7_out = out_root / "families" / "C7"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_v4_c7_partial_analysis_export.py"),
            "--results",
            str(c7_ledger),
            "--manifest",
            str(args.c7_manifest.resolve()),
            "--config",
            str(args.config.resolve()),
            "--out",
            str(c7_out),
            "--tag",
            c7_tag,
        ],
        check=True,
        cwd=ROOT,
    )

    c7_export_manifest = load_json(c7_out / c7_tag / "results_export_manifest.json")
    c7_blocked_scope = load_json(c7_out / c7_tag / "blocked_scope.json")
    c7_memo_path = c7_out / c7_tag / "paper" / "evidence_memo.json"
    c7_memo = load_json(c7_memo_path) if c7_memo_path.is_file() else {}
    c7_audit_path = c7_out / c7_tag / "tables" / "audit_report.json"
    c7_audit = load_json(c7_audit_path) if c7_audit_path.is_file() else {}
    c7_primary_path = c7_out / c7_tag / "tables" / "primary_results.csv"
    c7_primary_rows: list[dict] = []
    if c7_primary_path.is_file():
        with c7_primary_path.open(encoding="utf-8", newline="") as handle:
            c7_primary_rows = list(csv.DictReader(handle))

    horizontal_blocked = load_json(HORIZONTAL_SLICE / "blocked_scope.json")
    horizontal_memo = load_json(HORIZONTAL_SLICE / "paper" / "evidence_memo.json")

    c7_rows = sum(1 for line in c7_ledger.read_text(encoding="utf-8").splitlines() if line.strip())
    c7_planned = int(c7_blocked_scope.get("planned_c7_episodes") or 768)

    campaign_blocked = build_campaign_blocked_scope(
        c7_blocked_scope=c7_blocked_scope,
        horizontal_blocked_scope=horizontal_blocked,
        horizontal_evidence_memo=horizontal_memo,
        c7_ledger_rows=c7_rows,
        c7_planned=c7_planned,
    )
    blocked_path = out_root / "blocked_scope.json"
    blocked_path.write_text(json.dumps(campaign_blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    campaign_memo = build_campaign_evidence_memo(
        c7_memo=c7_memo,
        horizontal_memo=horizontal_memo,
        c7_export_manifest=c7_export_manifest,
        c7_audit=c7_audit,
    )
    paper_dir = out_root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    memo_path = paper_dir / "evidence_memo.json"
    memo_path.write_text(json.dumps(campaign_memo, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tables_dir = out_root / "tables"
    campaign_tables = build_campaign_tables(
        campaign_blocked=campaign_blocked,
        c7_audit=c7_audit,
        c7_primary_rows=c7_primary_rows,
    )
    table_artifacts: dict[str, dict] = {}
    for name, rows in campaign_tables.items():
        table_path = tables_dir / name
        write_csv(table_path, rows)
        table_artifacts[name] = artifact(table_path)

    audit_report = {
        "schema_version": "v4-registered-campaign-audit-v1",
        "frozen_analysis_manifest": str(FROZEN_ANALYSIS.relative_to(ROOT)),
        "c7_family_audit": c7_audit,
        "ledger_rows_compiled": c7_rows,
        "ledger_source": str(c7_ledger.relative_to(ROOT)),
        "pvc_complete_note": (
            "Ledger compile against confirmatory manifest is required to separate "
            "279 pre-repair infrastructure-invalid C7 episodes from repaired-path rows."
        ),
    }
    audit_path = tables_dir / "audit_report.json"
    audit_path.write_text(json.dumps(audit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_artifacts["audit_report.json"] = artifact(audit_path)

    figures_dir = out_root / "figures"
    scope_figure = figures_dir / "campaign_scope.svg"
    render_campaign_scope_figure(rows=campaign_tables["scope_summary.csv"], out_path=scope_figure)
    figures_manifest_path = figures_dir / "figures_manifest.json"
    figures_manifest = {
        "schema_version": "v4-registered-campaign-figures-v1",
        "campaign_scope": artifact(scope_figure),
        "c7_family_figures": str((c7_out / c7_tag / "figures").relative_to(ROOT)),
    }
    figures_manifest_path.write_text(json.dumps(figures_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    results_manifest_path = tables_dir / "results_manifest.json"
    results_manifest = {
        "schema_version": "v4-registered-campaign-results-manifest-v1",
        "tag": args.tag,
        "tables": table_artifacts,
        "figures_manifest": artifact(figures_manifest_path),
        "c7_family_tables": str((c7_out / c7_tag / "tables").relative_to(ROOT)),
    }
    results_manifest_path.write_text(json.dumps(results_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    results_stub = paper_dir / "RESULTS.md"
    results_stub.write_text(
        "# V4 registered campaign results\n\n"
        + "\n\n".join(campaign_memo["paper_narrative"]["paragraphs"])
        + "\n\n## Campaign tables and figures\n\n"
        + f"- Tables: `{tables_dir.relative_to(ROOT)}/`\n"
        + f"- Figures: `{figures_dir.relative_to(ROOT)}/`\n"
        + f"- C7 family export: `families/C7/{c7_tag}/`\n"
        + f"- Horizontal squeeze slice: `{HORIZONTAL_SLICE.relative_to(ROOT)}/`\n",
        encoding="utf-8",
    )

    validation = c7_audit.get("validation") or {}
    manifest = {
        "schema_version": "v4-registered-campaign-export-manifest-v1",
        "tag": args.tag,
        "output_root": str(out_root.relative_to(ROOT)),
        "c7_family_export": {
            "path": str((c7_out / c7_tag).relative_to(ROOT)),
            "manifest_sha256": sha256_file(c7_out / c7_tag / "results_export_manifest.json"),
        },
        "horizontal_slice": str(HORIZONTAL_SLICE.relative_to(ROOT)),
        "blocked_scope": artifact(blocked_path),
        "evidence_memo": artifact(memo_path),
        "results_md": artifact(results_stub),
        "tables": {"results_manifest": artifact(results_manifest_path), **table_artifacts},
        "figures": {
            "manifest": artifact(figures_manifest_path),
            "campaign_scope": artifact(scope_figure),
        },
        "frozen_analysis_manifest": artifact(FROZEN_ANALYSIS),
        "c7_ledger_rows": c7_rows,
        "c7_planned_episodes": c7_planned,
        "c7_valid_success_records": validation.get("valid_success_records"),
    }
    manifest_path = out_root / "registered_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
