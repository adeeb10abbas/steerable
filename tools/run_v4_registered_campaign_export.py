#!/usr/bin/env python3
"""Build the registered V4 campaign export from family ledgers and blocked-scope slices."""

from __future__ import annotations

import argparse
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
    payload = {
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
            "C6": {"planned_episodes": 768, "status": "achievable; confirmatory dispatch in progress"},
            "C8": {"planned_episodes": 768, "status": "achievable; 24-episode G7 pilot dispatched (8 lanes × 3); confirmatory after G8"},
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
    return payload


def build_campaign_evidence_memo(
    *,
    c7_memo: dict,
    horizontal_memo: dict,
    c7_export_manifest: dict,
) -> dict:
    horizontal_narrative = horizontal_memo.get("paper_narrative") or {}
    paragraphs = list(horizontal_narrative.get("paragraphs") or [])
    campaign_paragraphs = [
        "Registered campaign scope: 17,664 planned policy episodes; 2,304 achievable "
        "(C6, C7, C8 confirmatory families × 768); 15,360 scientifically blocked "
        "(C1/C3/C4 horizontal information-gate squeeze 9,728; C2 reference_binding "
        "information gate 4,096; C5 vertical IK reachability 768). No eligibility "
        "criterion, threshold, or scale ladder was amended to recover blocked scope.",
        *paragraphs,
    ]
    if c7_memo.get("intervention_trigger_positive_control"):
        campaign_paragraphs.append(
            "C7 behavioral evidence under verified repaired NaturalGraspDetector timing: "
            f"{c7_memo['intervention_trigger_positive_control'].get('status')} — "
            f"{c7_memo['intervention_trigger_positive_control'].get('finding')}."
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
            "accepted_ledger_sha256": (
                (c7_export_manifest.get("accepted_ledger") or {}).get("sha256")
            ),
        },
        "horizontal_slice": str(HORIZONTAL_SLICE.relative_to(ROOT)),
        "limitations": list(horizontal_memo.get("limitations") or [])
        + list(c7_memo.get("limitations") or []),
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
    export_cmd = [
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
    ]
    subprocess.run(export_cmd, check=True, cwd=ROOT)

    c7_export_manifest = load_json(c7_out / c7_tag / "results_export_manifest.json")
    c7_blocked_scope = load_json(c7_out / c7_tag / "blocked_scope.json")
    c7_memo_path = c7_out / c7_tag / "paper" / "evidence_memo.json"
    c7_memo = load_json(c7_memo_path) if c7_memo_path.is_file() else {}

    horizontal_blocked_path = HORIZONTAL_SLICE / "blocked_scope.json"
    horizontal_memo_path = HORIZONTAL_SLICE / "paper" / "evidence_memo.json"
    horizontal_blocked = load_json(horizontal_blocked_path)
    horizontal_memo = load_json(horizontal_memo_path)

    c7_rows = sum(
        1
        for line in c7_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    c7_planned = int(c7_blocked_scope.get("planned_c7_episodes") or 768)

    campaign_blocked = build_campaign_blocked_scope(
        c7_blocked_scope=c7_blocked_scope,
        horizontal_blocked_scope=horizontal_blocked,
        horizontal_evidence_memo=horizontal_memo,
        c7_ledger_rows=c7_rows,
        c7_planned=c7_planned,
    )
    blocked_path = out_root / "blocked_scope.json"
    blocked_path.write_text(
        json.dumps(campaign_blocked, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    campaign_memo = build_campaign_evidence_memo(
        c7_memo=c7_memo,
        horizontal_memo=horizontal_memo,
        c7_export_manifest=c7_export_manifest,
    )
    paper_dir = out_root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    memo_path = paper_dir / "evidence_memo.json"
    memo_path.write_text(
        json.dumps(campaign_memo, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results_stub = paper_dir / "RESULTS.md"
    paragraphs = campaign_memo["paper_narrative"]["paragraphs"]
    results_stub.write_text(
        "# V4 registered campaign results\n\n"
        + "\n\n".join(paragraphs)
        + "\n\n## C7 family export\n\n"
        + f"Tables and figures: `families/C7/{c7_tag}/`\n",
        encoding="utf-8",
    )

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
        "frozen_analysis_manifest": artifact(FROZEN_ANALYSIS),
        "c7_ledger_rows": c7_rows,
        "c7_planned_episodes": c7_planned,
    }
    manifest_path = out_root / "registered_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
