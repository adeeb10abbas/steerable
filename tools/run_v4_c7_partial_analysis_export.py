#!/usr/bin/env python3
"""Re-run C7 partial analysis export from a PVC-compiled accepted ledger."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.analysis import load_manifest  # noqa: E402

DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_MANIFEST = ROOT / "artifacts/online_correction_v4/queue.jsonl"
GRASP_POSITIVE_CONTROL_AUDIT = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_c7_grasp_detector_positive_control_audit.json"
)
LIVE_POSITIVE_CONTROL = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_object_pair_natural_grasp_live_positive_control_g3ngp20260908m.json"
)
TRIGGER_WIRING_RECLASSIFICATION = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_c7_natural_grasp_trigger_wiring_reclassification.json"
)
NATURAL_GRASP_LIVE_CONTROL_REGISTRY = (
    ROOT / "artifacts/online_correction_v4/setup/natural_grasp_live_control_registry.json"
)
REPAIRED_POSITIVE_CONTROL = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_object_pair_natural_grasp_live_positive_control_g3ngp20260908p.json"
)
GRASP_OPPORTUNITY_TIMING_AUDIT = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_c7_grasp_opportunity_timing_audit.json"
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_accepted_rows(results_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_trigger_defect_reclassified_results(
    accepted_rows: list[dict],
    *,
    reclassification_path: Path,
) -> list[dict]:
    reclassification = json.loads(reclassification_path.read_text(encoding="utf-8"))
    reason_text = (
        "inoperative_natural_grasp_trigger_wiring: object_kinematic_state() fed "
        "NaturalGraspDetector robot-base pose and object_grabbed() contact; live "
        "positive control confirmed blocking setup defect before repair verification"
    )
    rows: list[dict] = []
    for source in accepted_rows:
        episode_id = source.get("episode_id")
        if not isinstance(episode_id, str):
            continue
        attempt_id = source.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            attempt_id = "reclassified_trigger_wiring"
        rows.append(
            {
                "episode_id": episode_id,
                "attempt_id": attempt_id,
                "status": "infra_invalid",
                "reason": reason_text,
                "family": source.get("family", "C7"),
                "config_sha256": source.get("config_sha256"),
                "infrastructure_failure": {
                    "schema_version": "v4-c7-infrastructure-invalid-trigger-wiring-v1",
                    "failure_class": "inoperative_natural_grasp_trigger_wiring",
                    "reclassification_path": str(reclassification_path.relative_to(ROOT)),
                    "live_positive_control_path": reclassification.get("live_positive_control", {}).get(
                        "path"
                    ),
                    "defect_summary": reclassification.get("defect", {}).get("summary"),
                    "paper_boundary": reclassification.get("paper_boundary"),
                },
                "provenance": {
                    "source_accepted_ledger_row": True,
                    "prior_interpretation": "valid_no_grasp_under_inoperative_trigger",
                },
            }
        )
    return rows


def build_scoped_manifest(
    *,
    queue_path: Path,
    accepted_rows: list[dict],
    family: str = "C7",
    allow_empty: bool = False,
) -> tuple[list[dict], dict]:
    queue_by_id = {row["episode_id"]: row for row in load_manifest(queue_path)}
    scoped: list[dict] = []
    missing: list[str] = []
    for row in accepted_rows:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str):
            continue
        manifest_row = queue_by_id.get(episode_id)
        if manifest_row is None:
            missing.append(episode_id)
            continue
        if manifest_row.get("family") != family:
            continue
        bound = dict(manifest_row)
        config_sha = row.get("config_sha256")
        if isinstance(config_sha, str):
            bound["config_sha256"] = config_sha
        scoped.append(bound)
    if missing:
        raise SystemExit(
            f"accepted ledger references {len(missing)} episode IDs missing from queue manifest"
        )
    if not scoped and not allow_empty:
        raise SystemExit("no accepted rows matched the requested family in the queue manifest")
    planned_c7 = sum(1 for row in queue_by_id.values() if row.get("family") == family)
    summary = {
        "family": family,
        "accepted_unique_episodes": len({row["episode_id"] for row in scoped}),
        "accepted_rows": len(accepted_rows),
        "planned_family_episodes": planned_c7,
        "missing_family_episodes": planned_c7 - len({row["episode_id"] for row in scoped}),
        "manifest_binding": "accepted_rows_only_with_frozen_config_sha256_from_ledger",
    }
    return scoped, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="accepted_ledger.jsonl")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/results/c7_partial",
    )
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument(
        "--apply-trigger-wiring-reclassification",
        action="store_true",
        help="Reclassify source accepted rows as infra_invalid under inoperative trigger wiring",
    )
    args = parser.parse_args(argv)
    results = args.results.resolve()
    if not results.is_file():
        raise SystemExit(f"missing accepted ledger: {results}")
    source_accepted_rows = load_accepted_rows(results)
    reclassification_path = TRIGGER_WIRING_RECLASSIFICATION.resolve()
    apply_reclassification = bool(
        args.apply_trigger_wiring_reclassification and reclassification_path.is_file()
    )
    if apply_reclassification:
        analysis_results = build_trigger_defect_reclassified_results(
            source_accepted_rows,
            reclassification_path=reclassification_path,
        )
    else:
        analysis_results = source_accepted_rows
    accepted_rows = source_accepted_rows
    scoped_manifest, scope_summary = build_scoped_manifest(
        queue_path=args.manifest.resolve(),
        accepted_rows=[] if apply_reclassification else accepted_rows,
        allow_empty=apply_reclassification,
    )
    if apply_reclassification:
        scope_summary = {
            **scope_summary,
            "family": "C7",
            "accepted_unique_episodes": 0,
            "accepted_rows": len(source_accepted_rows),
            "source_accepted_unique_episodes": len(
                {row["episode_id"] for row in source_accepted_rows if row.get("episode_id")}
            ),
            "infra_invalid_trigger_wiring_episodes": len(source_accepted_rows),
            "manifest_binding": "behavioral_scope_empty_after_trigger_wiring_reclassification",
        }
    out_root = args.out.resolve() / args.tag
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    scoped_manifest_path = out_root / "scoped_c7_manifest.jsonl"
    write_jsonl(scoped_manifest_path, scoped_manifest)
    analysis_results_path = out_root / "analysis_results.jsonl"
    write_jsonl(
        analysis_results_path,
        [] if apply_reclassification else analysis_results,
    )
    reclassified_results_path = out_root / "reclassified_infra_invalid_results.jsonl"
    if apply_reclassification:
        write_jsonl(reclassified_results_path, analysis_results)
    analyze = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/analyze_online_correction_v4.py"),
            "--config",
            str(args.config.resolve()),
            "--manifest",
            str(scoped_manifest_path),
            "--results",
            str(analysis_results_path),
            "--out",
            str(out_root / "tables"),
        ],
        check=True,
        cwd=ROOT,
    )
    export = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/export_v4_paper_bundle.py"),
            "--config",
            str(args.config.resolve()),
            "--manifest",
            str(scoped_manifest_path),
            "--results",
            str(analysis_results_path),
            "--out",
            str(out_root),
        ],
        check=True,
        cwd=ROOT,
    )
    figures = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/render_v4_analysis_figures.py"),
            "--tables",
            str(out_root / "tables"),
            "--out",
            str(out_root / "figures"),
        ],
        check=True,
        cwd=ROOT,
    )
    grasp_audit = {}
    if GRASP_POSITIVE_CONTROL_AUDIT.is_file():
        grasp_audit = json.loads(
            GRASP_POSITIVE_CONTROL_AUDIT.read_text(encoding="utf-8")
        )
    live_positive = {}
    if LIVE_POSITIVE_CONTROL.is_file():
        live_positive = json.loads(LIVE_POSITIVE_CONTROL.read_text(encoding="utf-8"))
    reclassification = {}
    if apply_reclassification and reclassification_path.is_file():
        reclassification = json.loads(reclassification_path.read_text(encoding="utf-8"))
    live_control_registry = {}
    if NATURAL_GRASP_LIVE_CONTROL_REGISTRY.is_file():
        live_control_registry = json.loads(
            NATURAL_GRASP_LIVE_CONTROL_REGISTRY.read_text(encoding="utf-8")
        )
    repaired_positive = {}
    if REPAIRED_POSITIVE_CONTROL.is_file():
        repaired_positive = json.loads(REPAIRED_POSITIVE_CONTROL.read_text(encoding="utf-8"))
    timing_audit = {}
    if GRASP_OPPORTUNITY_TIMING_AUDIT.is_file():
        timing_audit = json.loads(
            GRASP_OPPORTUNITY_TIMING_AUDIT.read_text(encoding="utf-8")
        )
    timing_verdict = timing_audit.get("interpretation", {}).get("verdict")
    timing_audit_complete = timing_audit.get("status") == "complete"
    policy_no_grasp_settled = (
        not apply_reclassification
        and timing_audit_complete
        and timing_verdict == "policy_failure_under_correct_timing"
    )
    passed_live_controls = live_control_registry.get("passed_controls") or []
    rb_controls = {
        item.get("control_mode"): item
        for item in passed_live_controls
        if item.get("fixture_id") == "reference_binding" and item.get("status") == "passed"
    }
    c8_controls = {
        item.get("control_mode"): item
        for item in passed_live_controls
        if item.get("fixture_id") == "second_stack" and item.get("status") == "passed"
    }
    repair_verified = (
        "scripted_grasp" in rb_controls
        and "hold_only" in rb_controls
        and "scripted_grasp" in c8_controls
        and "hold_only" in c8_controls
    )
    trigger_finding = (
        "blocking_setup_defect_confirmed_then_repaired"
        if apply_reclassification and repair_verified
        else (
            "policy_failure_under_correct_timing"
            if policy_no_grasp_settled
            else (
                "blocking_setup_defect"
                if live_positive.get("verdict") == "blocking_setup_defect"
                else grasp_audit.get("finding", "pending_verdict")
            )
        )
    )
    trigger_status = (
        "repair_verified_object_pair_isaac"
        if apply_reclassification and repair_verified
        else (
            "timing_audit_confirms_genuine_policy_no_grasp"
            if policy_no_grasp_settled
            else (
                "blocking_setup_defect_confirmed"
                if live_positive.get("verdict") == "blocking_setup_defect"
                else "pending_live_isaac_verdict"
            )
        )
    )
    paper_caveat = (
        "Live Isaac positive control confirmed a blocking setup defect: scripted grasp stages passed but trigger_eligible never fired because object_kinematic_state() fed NaturalGraspDetector robot-base pose and object_grabbed() contact. The 279 C7 episodes collected before repair verification were run under an inoperative intervention trigger and are excluded from all behavioral claims. Post-repair live positive and negative controls passed on object_pair Isaac at study commit c9c988d; confirmatory redispatch is required before any behavioral headline."
        if apply_reclassification
        else (
            timing_audit.get("interpretation", {}).get("verdict_detail")
            if policy_no_grasp_settled
            else grasp_audit.get(
                "paper_distinction",
                "Do not present zero trigger_eligible as a settled policy finding until live detector positive control is verified.",
            )
        )
    )
    memo_path = out_root / "paper" / "evidence_memo.json"
    if memo_path.is_file():
        memo = json.loads(memo_path.read_text(encoding="utf-8"))
        memo.setdefault("limitations", [])
        memo["limitations"] = [
            item
            for item in memo["limitations"]
            if "Figures require accepted trajectory" not in item
            and "preliminary only" not in item
            and "not yet a settled policy headline" not in item
            and "pending_live_isaac_verdict" not in item
        ]
        if apply_reclassification:
            memo["limitations"].extend(
                [
                    "C7 confirmatory episodes previously compiled as accepted_valid are reclassified infrastructure-invalid under inoperative NaturalGraspDetector wiring; zero successes must not appear in behavioral claims.",
                    "279 pre-repair C7 episodes remain excluded from all behavioral claims under disclosed setup repair (eef_tool_length_m flange offset).",
                    "Original campaign allocation was 17,664 policy episodes; achievable scope after disclosed scientific blocks is 6,400 (C7 768 + C6 768 + C2 4,096 + C8 768).",
                    "C1/C3/C4 (9,728 episodes) are scientifically blocked: horizontal G3 homogeneous 128-seed wave failed shrinking-area information gates on right/behind for physical_translation_sign=-1 across half the registry.",
                    "C5 (768 episodes) is scientifically blocked: vertical fixture IK reachability finding at confirmatory geometry.",
                    "C2 primary inference remains not estimable until verified common-prefix replay is recorded after G3/G4-G6; live controls for reference_binding now pass under shared flange-offset repair.",
                    "C8 confirmatory dispatch remains blocked until G7/G8, confirmatory runtime lock, and policy runner wiring complete; SimplerEnv trigger adapter is verified independently.",
                ]
            )
        else:
            limitations = [
                "279 pre-repair C7 episodes remain excluded from all behavioral claims under disclosed setup repair (eef_tool_length_m flange offset).",
                "C2 primary inference remains not estimable until verified common-prefix replay is recorded.",
            ]
            if policy_no_grasp_settled:
                limitations.extend(
                    [
                        "Agent B timing audit confirms repaired-path confirmatory episodes run the full 61.0 s simulation window with grasp_eligible false on every control tick; zero successes reflect genuine policy no-grasp under registered timing, not episode-budget truncation.",
                        "C2 and C8 full policy dispatches (4096 and 768 episodes) remain blocked until their respective qualification gates complete.",
                    ]
                )
            else:
                limitations.extend(
                    [
                        "C7 confirmatory success rates in this partial export are preliminary only; Agent B is verifying whether zero successes reflect genuine policy behavior or an interface regression.",
                        "NaturalGraspDetector has only a synthetic G5 positive control; zero trigger_eligible across live C7 episodes is not yet a settled policy headline until the live Isaac scripted grasp verdict lands.",
                        "C2 and C8 full policy dispatches (4096 and 768 episodes) remain blocked until the live trigger positive-control question is resolved.",
                    ]
                )
            memo["limitations"].extend(limitations)
        memo["intervention_trigger_positive_control"] = {
            "audit_path": str(GRASP_POSITIVE_CONTROL_AUDIT.relative_to(ROOT))
            if GRASP_POSITIVE_CONTROL_AUDIT.is_file()
            else None,
            "timing_audit_path": str(GRASP_OPPORTUNITY_TIMING_AUDIT.relative_to(ROOT))
            if GRASP_OPPORTUNITY_TIMING_AUDIT.is_file()
            else None,
            "timing_audit_sha256": sha256_file(GRASP_OPPORTUNITY_TIMING_AUDIT)
            if GRASP_OPPORTUNITY_TIMING_AUDIT.is_file()
            else None,
            "live_positive_control_path": str(LIVE_POSITIVE_CONTROL.relative_to(ROOT))
            if LIVE_POSITIVE_CONTROL.is_file()
            else None,
            "reclassification_path": str(reclassification_path.relative_to(ROOT))
            if apply_reclassification
            else None,
            "finding": trigger_finding,
            "live_trigger_eligible_observed": live_positive.get("trigger_eligible"),
            "scripted_grasp_stages_passed": live_positive.get(
                "scripted_trajectory_passed"
            ),
            "status": trigger_status,
            "paper_caveat": paper_caveat,
        }
        if apply_reclassification:
            memo["c7_behavioral_disposition"] = {
                "accepted_valid_for_behavioral_analysis": 0,
                "infra_invalid_trigger_wiring": reclassification.get(
                    "episode_count_reclassified", len(source_accepted_rows)
                ),
                "exclude_from_policy_findings": True,
            }
        memo_path.write_text(
            json.dumps(memo, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    blocked_path = out_root / "blocked_scope.json"
    c7_status = (
        "all 279 previously accepted episodes reclassified infrastructure-invalid under inoperative NaturalGraspDetector wiring; excluded from behavioral claims; raw PVC evidence preserved"
        if apply_reclassification
        else (
            "partial export with timing-audit-backed genuine policy no-grasp on repaired-path confirmatory completes; 279 pre-repair episodes excluded"
            if policy_no_grasp_settled
            else "partial export only; preliminary — zero successes not a settled headline pending Agent B interface verification and live NaturalGraspDetector positive control"
        )
    )
    blocked_payload = {
        "schema_version": "v4-c7-partial-blocked-scope-v2",
        "accepted_c7_episodes_behavioral": 0 if apply_reclassification else scope_summary["accepted_unique_episodes"],
        "infra_invalid_c7_episodes_trigger_wiring": reclassification.get(
            "episode_count_reclassified", len(source_accepted_rows)
        )
        if apply_reclassification
        else 0,
        "source_accepted_ledger_episodes": len(source_accepted_rows),
        "planned_c7_episodes": scope_summary["planned_family_episodes"],
        "missing_c7_episodes": scope_summary["missing_family_episodes"],
        "campaign_scope_revision": {
            "original_planned_policy_episodes": 17664,
            "achievable_policy_episodes": 2304,
            "achievable_breakdown": {
                "C6": 768,
                "C7": 768,
                "C8": 768,
            },
            "scientifically_blocked_episodes": 15360,
            "blocked_breakdown": {
                "C1_C3_C4_horizontal": 9728,
                "C2_reference_binding": 4096,
                "C5_vertical": 768,
            },
            "disclosed_setup_repairs": [
                "NaturalGraspDetector trigger observation wiring repaired: object_kinematic_state() now uses finger-contact coupling instead of robot-base pose (C7)",
                "eef_tool_length_m=0.14 flange-versus-fingerpad offset applied uniformly across Isaac fixtures",
                "second_stack observation and sampling defect repaired: SimplerEnv adapter uses control-boundary sampling with provisioned LD_LIBRARY_PATH/VK_ICD_FILENAMES (C8)",
            ],
            "pre_repair_c7_excluded_episodes": 279,
            "horizontal_geometry_repair_note": (
                "Horizontal/reference_binding geometry repair restored physical feasibility "
                "yet the registered shrinking-area information gate still fails across the "
                "full 128-seed homogeneous ladder at scale 0.5 under counterbalance semantics."
            ),
        },
        "not_estimable_or_blocked": {
            "C1": "scientifically blocked with C3/C4: horizontal G3 information-gate rejection across full registered ladder at scale 0.5 (64/128 seeds fail right/behind for physical_translation_sign=-1); receipt 20260908_horizontal_g3_gate_decision_g3r20260908g.json",
            "C2": (
                "scientifically blocked: homogeneous G3 gate g3rb20260908v failed shrinking-area "
                "information gate at confirmatory scale 0.5 (96/128 seeds fail); sign=-1 "
                "0/64 pass and sign=+1 32/64 fail; front goal passes 128/128 case checks; "
                "behind fails 63/128; all 3072 path checks pass; computation audit confirmed "
                "correct gate math; primary estimand not estimable; 4096 confirmatory episodes "
                "blocked; no criterion was adjusted to recover blocked scope"
            ),
            "C3": "scientifically blocked with C1/C4: horizontal information-gate failure on registered ladder",
            "C4": "scientifically blocked with C1/C3: horizontal information-gate failure on registered ladder",
            "C5": "scientifically blocked: vertical IK reachability finding at seed 2100020000 scale 0.5; receipt 20260908_vertical_natural_grasp_live_control_finding_g3ngp20260908vp8.json",
            "C6": "achievable (768 episodes); dispatch pending qualification completion",
            "C7": c7_status,
            "C8": "achievable (768 episodes); pilot G3 path gate passed 96/96 (g3c8p20260908f); confirmatory G7/G8 release and dispatch in progress",
        },
        "scientific_blockers": {
            "horizontal_information_gate": {
                "attempt_id": "g3r20260908g",
                "receipt_path": "artifacts/online_correction_v4/qualification/20260908_horizontal_g3_gate_decision_g3r20260908g.json",
                "affected_families": ["C1", "C3", "C4"],
                "affected_episodes": 9728,
            },
            "reference_binding_information_gate": {
                "attempt_id": "g3rb20260908v",
                "receipt_path": "artifacts/online_correction_v4/qualification/20260908_reference_binding_g3_gate_decision_g3rb20260908v.json",
                "provisional_receipt_path": "artifacts/online_correction_v4/qualification/20260908_reference_binding_g3_gate_decision_g3rb20260908v_provisional.json",
                "computation_audit_path": "artifacts/online_correction_v4/qualification/20260908_information_gate_computation_audit.json",
                "affected_families": ["C2"],
                "affected_episodes": 4096,
                "primary_estimand_status": "not_estimable",
            },
            "vertical_ik_reachability": {
                "receipt_path": "artifacts/online_correction_v4/qualification/20260908_vertical_natural_grasp_live_control_finding_g3ngp20260908vp8.json",
                "affected_families": ["C5"],
                "affected_episodes": 768,
            },
        },
        "intervention_trigger_positive_control": {
            "audit_path": str(GRASP_POSITIVE_CONTROL_AUDIT.relative_to(ROOT))
            if GRASP_POSITIVE_CONTROL_AUDIT.is_file()
            else None,
            "audit_sha256": sha256_file(GRASP_POSITIVE_CONTROL_AUDIT)
            if GRASP_POSITIVE_CONTROL_AUDIT.is_file()
            else None,
            "live_positive_control_path": str(LIVE_POSITIVE_CONTROL.relative_to(ROOT))
            if LIVE_POSITIVE_CONTROL.is_file()
            else None,
            "live_positive_control_sha256": sha256_file(LIVE_POSITIVE_CONTROL)
            if LIVE_POSITIVE_CONTROL.is_file()
            else None,
            "repaired_positive_control_path": str(REPAIRED_POSITIVE_CONTROL.relative_to(ROOT))
            if REPAIRED_POSITIVE_CONTROL.is_file()
            else None,
            "repaired_positive_control_sha256": sha256_file(REPAIRED_POSITIVE_CONTROL)
            if REPAIRED_POSITIVE_CONTROL.is_file()
            else None,
            "live_controls_registry_path": str(
                NATURAL_GRASP_LIVE_CONTROL_REGISTRY.relative_to(ROOT)
            )
            if NATURAL_GRASP_LIVE_CONTROL_REGISTRY.is_file()
            else None,
            "reclassification_path": str(reclassification_path.relative_to(ROOT))
            if apply_reclassification
            else None,
            "finding": trigger_finding,
            "live_trigger_eligible_observed_pre_repair": live_positive.get(
                "trigger_eligible"
            ),
            "live_trigger_eligible_observed_post_repair": repaired_positive.get(
                "trigger_eligible"
            ),
            "scripted_grasp_stages_passed": live_positive.get(
                "scripted_trajectory_passed"
            ),
            "defect_component": reclassification.get("defect", {}).get("component"),
            "repair_status": reclassification.get("defect", {}).get("repair_status"),
            "status": trigger_status,
            "policy_dispatch_blocked": {
                "C2_confirmatory_4096": "held until homogeneous G3 gate (both translation signs), G4-G6, prefix replay, and confirmatory runtime lock",
                "C8_confirmatory_768": "held until G7/G8, confirmatory runtime lock, and second_stack policy runner wiring",
            },
            "paper_caveat": paper_caveat,
        },
    }
    blocked_path.write_text(
        json.dumps(blocked_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "v4-c7-partial-results-manifest-v1",
        "tag": args.tag,
        "scope_summary": scope_summary,
        "accepted_ledger": {
            "path": str(results),
            "sha256": sha256_file(results),
            "bytes": results.stat().st_size,
        },
        "analysis_results": {
            "path": str(analysis_results_path),
            "sha256": sha256_file(analysis_results_path),
            "bytes": analysis_results_path.stat().st_size,
            "trigger_wiring_reclassification_applied": apply_reclassification,
        },
        "scoped_manifest": {
            "path": str(scoped_manifest_path),
            "sha256": sha256_file(scoped_manifest_path),
        },
        "blocked_scope": {
            "path": str(blocked_path),
            "sha256": sha256_file(blocked_path),
        },
        "figures": json.loads(
            (out_root / "figures" / "figures_manifest.json").read_text(encoding="utf-8")
        )
        if (out_root / "figures" / "figures_manifest.json").is_file()
        else {},
        "output_root": str(out_root),
        "analyze_exit_code": analyze.returncode,
        "export_exit_code": export.returncode,
        "figures_exit_code": figures.returncode,
    }
    manifest_path = out_root / "results_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
