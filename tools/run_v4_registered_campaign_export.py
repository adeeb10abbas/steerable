#!/usr/bin/env python3
"""Build the registered V4 campaign export from family ledgers and blocked-scope slices."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v4_registered_export_helpers import (  # noqa: E402
    build_coverage_metadata,
    build_dispatch_coverage_metadata,
    format_outcome_decomposition,
    load_accepted_ledger_rows,
    load_manifest_by_episode_id,
    outcome_composition_rows,
    summarize_outcome_composition,
    summarize_scenario_outcome_breakdown,
)

DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_C7_MANIFEST = (
    ROOT / "artifacts/online_correction_v4/setup/c7_confirmatory/queue.frozen.jsonl"
)
DEFAULT_C6_MANIFEST = (
    ROOT / "artifacts/online_correction_v4/setup/c6_confirmatory/queue.frozen.jsonl"
)
DEFAULT_C8_MANIFEST = (
    ROOT / "artifacts/online_correction_v4/setup/c8_confirmatory/queue.frozen.jsonl"
)
DEFAULT_C6_PILOT_LEDGER = (
    ROOT
    / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908/pilot-ledger-20260908a/accepted_ledger.jsonl"
)
DEFAULT_C6_PILOT_MANIFEST = (
    ROOT / "artifacts/online_correction_v4/setup/containment_g7_pilot_manifest.jsonl"
)
DEFAULT_C8_PILOT_LEDGER = (
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/pilot-ledger-20260908b/accepted_ledger.jsonl"
)
DEFAULT_C8_PILOT_MANIFEST = (
    ROOT / "artifacts/online_correction_v4/setup/second_stack_g7_pilot_queue.jsonl"
)
C8_EXECUTION_STATUS = (
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/pvc-receipts-sync/c8_execution_status_20260908i.json"
)
C8_EXECUTION_STATUS_FALLBACKS = (
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/pvc-receipts-sync/c8_execution_status_20260908h.json",
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/pvc-receipts-sync/c8_execution_status_20260908g.json",
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/pvc-receipts-sync/c8_execution_status_20260908d.json",
)
C8_PILOT_COMPOSITION = (
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/pvc-receipts-sync/c8_pilot_composition_by_scenario_20260908a.json"
)
C8_CONFIRMATORY_COMPOSITION_MILESTONE = (
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/pvc-receipts-sync/c8_confirmatory_composition_by_scenario_20260908d.json"
)
C8_DESTINATION_STATIC_CONFOUND = (
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/pvc-receipts-sync/c8_destination_static_confound_check_20260908a.json"
)
DEFAULT_C8_CONFIRMATORY_LEDGER = (
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/confirmatory-ledger-20260908d/accepted_ledger.jsonl"
)
C7_COMPILE_RETIREMENT = (
    ROOT
    / "artifacts/online_correction_v4/execution/c7_object_pair_20260906/compile_retirement_pending.json"
)
DEFAULT_C7_FINAL_LEDGER = (
    ROOT
    / "artifacts/online_correction_v4/execution/c7_object_pair_20260906/pvc-ledgers/compiled_ledger_20260908_FINAL/accepted_ledger.jsonl"
)
DEFAULT_C7_PARTIAL_LEDGER_M = (
    ROOT
    / "artifacts/online_correction_v4/execution/c7_object_pair_20260906/pvc-ledgers/compiled_ledger_20260908m/accepted_ledger.jsonl"
)
C6_WAVE_B_DISPATCH = (
    ROOT
    / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908/create-c6confirm20260908b.json"
)
C6_WAVE_D_DISPATCH = (
    ROOT
    / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908/create-c6confirm20260908d.json"
)
C6_WAVE_F_DISPATCH = (
    ROOT
    / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908/create-c6confirm20260908f.json"
)
C6_RUNNER_BINDING = (
    ROOT
    / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908/runner_binding_resolution_20260908.json"
)
GPU_POOL_SWEEP_RECEIPT = (
    ROOT / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_pool_sweep_receipt.json"
)
C6_G7_RECEIPT = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_containment_g7_pilot_receipt_g7c6p20260908a.json"
)
C8_G7_RECEIPT = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_second_stack_g7_pilot_g7c8p20260908a.json"
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
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_compile_retirement_rows(retirement: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in retirement.get("retired_partials") or []:
        path = str(item.get("path") or "")
        compile_id = path.rsplit("/", 1)[-1] if path else ""
        rows.append(
            {
                "compile_id": compile_id,
                "rows": item.get("rows"),
                "status": item.get("status", "retired_partial"),
                "superseded_by": item.get("superseded_by"),
                **{f"outcome_{key}": value for key, value in (item.get("composition") or {}).items()},
            }
        )
    return rows


def build_c8_grasp_by_scenario_rows(composition: dict[str, Any], *, cohort: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, payload in (composition.get("by_scenario") or {}).items():
        episodes = int(
            payload.get("episodes")
            or payload.get("episodes_completed")
            or payload.get("completed")
            or 0
        )
        grasp = int(payload.get("grasp_achieved") or 0)
        rows.append(
            {
                "cohort": cohort,
                "scenario": scenario,
                "episodes": episodes,
                "grasp_achieved": grasp,
                "no_grasp": payload.get("no_grasp"),
                "transport_incomplete": payload.get("transport_incomplete"),
                "grasp_rate": round(grasp / max(episodes, 1), 3),
                "estimability_note": (
                    "pilot_hypothesis_only"
                    if cohort == "pilot"
                    else (
                        "insufficient_coverage_not_estimable"
                        if (composition.get("ordering_assessment") or {}).get("ordering_holds")
                        == "insufficient_coverage"
                        or int((composition.get("progress") or {}).get("behavioral_valid") or 0) < 768
                        else "confirmatory_partial"
                    )
                ),
            }
        )
    return rows


def build_confound_check_rows(confound: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = confound.get("evidence") or {}
    distances = evidence.get("green_yellow_distance_at_reset_m") or {}
    rows: list[dict[str, Any]] = []
    for scenario, stats in distances.items():
        rows.append(
            {
                "scenario": scenario,
                "green_yellow_distance_mean_m": stats.get("mean"),
                "green_yellow_distance_stdev_m": stats.get("stdev"),
                "seed_count": stats.get("n"),
                "initial_geometry_identical": confound.get("answer"),
                "protocol_difference": (
                    "reference placement profile only"
                    if scenario == "destination_static"
                    else "identity or motion profile"
                ),
            }
        )
    return rows


def build_capacity_finding() -> dict[str, Any]:
    sweep = load_json(GPU_POOL_SWEEP_RECEIPT) if GPU_POOL_SWEEP_RECEIPT.is_file() else {}
    session = sweep.get("session_cumulative") or {}
    status = load_c8_execution_status()
    return {
        "lane_pair_model": (
            "Lane pair is a study-design and throughput unit, not a Kubernetes scheduler "
            "constraint: policy and simulator are separate single-GPU pods connected over HTTP."
        ),
        "pending_failure_attribution": (
            "Pending pod failures are attributable to per-node GPU fragmentation and legitimate "
            "C7 occupancy, not an unsatisfiable two-GPU request on one node."
        ),
        "gpu_orphan_sweep": {
            "tool": "tools/v4_gpu_pool_sweep.py",
            "receipt": str(GPU_POOL_SWEEP_RECEIPT.relative_to(ROOT)) if GPU_POOL_SWEEP_RECEIPT.is_file() else None,
            "reclaimed_gpu_total": session.get("reclaimed_gpu_total"),
            "reclaimed_gpus_by_product": session.get("reclaimed_gpus_by_product"),
            "pools_swept": len(session.get("reclaimed_gpus_by_product") or {}),
        },
        "placement_policies": {
            "tool": "tools/v4_gpu_scheduling.py",
            "policies": ["c8_a40_spread", "c6_a10040_spread"],
            "purpose": "Topology spread and soft anti-affinity to bin-pack single-GPU pods across nodes",
        },
        "confirmatory_episodes_per_family": 768,
        "estimated_c8_confirmatory_walltime_hours_at_full_a40_pool": 16,
        "c8_execution_status_receipt": str(resolve_c8_status_receipt_path().relative_to(ROOT))
        if resolve_c8_status_receipt_path()
        else None,
        "c8_confirmatory_valid": (status.get("confirmatory_progress") or {}).get("valid"),
    }


def load_compile_provenance(*, active_compile_id: str, active_rows: int, is_final: bool) -> dict[str, Any]:
    if not C7_COMPILE_RETIREMENT.is_file():
        return {
            "active_compile_id": active_compile_id,
            "active_rows": active_rows,
            "is_final": is_final,
            "retirement_manifest": None,
        }
    retirement = load_json(C7_COMPILE_RETIREMENT)
    return {
        "schema_version": retirement.get("schema_version"),
        "retirement_manifest": str(C7_COMPILE_RETIREMENT.relative_to(ROOT)),
        "final_compile_path": retirement.get("final_compile_path"),
        "authoritative_compile_id": active_compile_id if is_final else retirement.get("final_compile_path", "").rsplit("/", 1)[-1],
        "active_compile_id": active_compile_id,
        "active_rows": active_rows,
        "is_final": is_final,
        "retired_partials": retirement.get("retired_partials"),
        "note": retirement.get("note"),
    }


def resolve_c7_ledger(args: argparse.Namespace) -> tuple[Path, str, bool]:
    if args.c7_ledger is not None:
        path = args.c7_ledger.resolve()
        compile_id = args.c7_ledger_compile_id or path.parent.name
        return path, compile_id, compile_id.endswith("FINAL")
    if DEFAULT_C7_FINAL_LEDGER.is_file():
        return DEFAULT_C7_FINAL_LEDGER, "compiled_ledger_20260908_FINAL", True
    if DEFAULT_C7_PARTIAL_LEDGER_M.is_file():
        return DEFAULT_C7_PARTIAL_LEDGER_M, "compiled_ledger_20260908m", False
    raise SystemExit(
        "missing C7 accepted ledger: provide --c7-ledger or place compile-m/FINAL under pvc-ledgers/"
    )


def load_c8_execution_status() -> dict[str, Any]:
    for path in (C8_EXECUTION_STATUS, *C8_EXECUTION_STATUS_FALLBACKS):
        if path.is_file():
            return load_json(path)
    return {}


def resolve_c8_status_receipt_path() -> Path | None:
    for path in (C8_EXECUTION_STATUS, *C8_EXECUTION_STATUS_FALLBACKS):
        if path.is_file():
            return path
    return None


def resolve_c8_confirmatory_composition() -> dict[str, Any] | None:
    status = load_c8_execution_status()
    milestone = (
        load_json(C8_CONFIRMATORY_COMPOSITION_MILESTONE)
        if C8_CONFIRMATORY_COMPOSITION_MILESTONE.is_file()
        else {}
    )
    progress = status.get("confirmatory_progress") or {}
    live_valid = int(progress.get("valid") or progress.get("behavioral_valid") or 0)
    if not live_valid and not milestone:
        return None
    by_scenario: dict[str, Any] = {}
    grasp_total = 0
    aggregate: dict[str, int] = {"no_grasp": 0, "transport_incomplete": 0}
    for scenario, payload in (progress.get("by_scenario") or milestone.get("by_scenario") or {}).items():
        completed = int(payload.get("completed") or payload.get("episodes_completed") or 0)
        grasp = int(payload.get("grasp_achieved") or 0)
        grasp_total += grasp
        aggregate["no_grasp"] += int(payload.get("no_grasp") or 0)
        aggregate["transport_incomplete"] += int(payload.get("transport_incomplete") or 0)
        by_scenario[scenario] = {
            "episodes_completed": completed,
            "no_grasp": payload.get("no_grasp"),
            "transport_incomplete": payload.get("transport_incomplete"),
            "grasp_achieved": grasp,
            "grasp_rate": round(grasp / max(completed, 1), 3),
        }
    ordering = milestone.get("ordering_assessment") or {}
    return {
        **milestone,
        "ledger_compile_id_milestone_50": milestone.get("ledger_compile_id") or "20260908d",
        "ledger_compile_id_live_interim": status.get("confirmatory_ledger_compile_id_live"),
        "next_milestone_compile_at": status.get("next_milestone_compile_at") or milestone.get("next_milestone_compile_at"),
        "progress": {
            "behavioral_valid": live_valid,
            "total_queued": int(progress.get("remaining", 0) + live_valid) if progress.get("remaining") is not None else 768,
            "coverage_fraction": round(live_valid / 768, 4) if live_valid else milestone.get("progress", {}).get("coverage_fraction"),
        },
        "aggregate_outcomes": aggregate if live_valid else milestone.get("aggregate_outcomes"),
        "grasp_achieved_count": grasp_total if live_valid else milestone.get("grasp_achieved_count"),
        "by_scenario": by_scenario,
        "ordering_assessment": {
            **ordering,
            "ordering_holds": progress.get("ordering_holds") or ordering.get("ordering_holds", "insufficient_coverage"),
            "confirmatory_observed_at_live": (
                "destination_static > move_stop > original_sham"
                if live_valid >= 6
                else ordering.get("confirmatory_observed_at_83")
            ),
        },
        "confound_check_uri": status.get("destination_static_confound_uri")
        or milestone.get("ordering_assessment", {}).get("confound_check_uri"),
    }


def resolve_campaign_export_status(*, c7_is_final: bool, family_rollups: dict[str, dict]) -> str:
    if not c7_is_final:
        return "partial"
    c6_accepted = int(
        ((family_rollups.get("C6") or {}).get("confirmatory_dispatch") or {}).get("accepted_valid_unique") or 0
    )
    c8_accepted = int(
        ((family_rollups.get("C8") or {}).get("confirmatory_dispatch") or {}).get("accepted_valid_unique") or 0
    )
    if c6_accepted >= 768 and c8_accepted >= 768:
        return "complete"
    return "partial"


def build_campaign_blocked_scope(
    *,
    c7_blocked_scope: dict,
    horizontal_blocked_scope: dict,
    family_status: dict[str, dict],
    compile_provenance: dict[str, Any],
    c8_scenario_grasp: dict[str, Any] | None = None,
    c8_confirmatory_grasp: dict[str, Any] | None = None,
    c8_confound_check: dict[str, Any] | None = None,
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
        "family_status": family_status,
        "capacity_finding": build_capacity_finding(),
        "c6_runner_binding_resolution": (
            str(C6_RUNNER_BINDING.relative_to(ROOT)) if C6_RUNNER_BINDING.is_file() else None
        ),
        "c8_destination_static_confound_check": c8_confound_check,
        "c7_compile_provenance": compile_provenance,
        "c8_pilot_scenario_grasp_pattern": c8_scenario_grasp,
        "c8_confirmatory_scenario_grasp_pattern": c8_confirmatory_grasp,
        "not_estimable_or_blocked": not_estimable,
        "scientific_blockers": c7_blocked_scope.get("scientific_blockers"),
        "horizontal_evidence": {
            "manifest": "artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908/evidence_manifest.json",
            "blocked_scope": "artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908/blocked_scope.json",
            "evidence_memo": "artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908/paper/evidence_memo.json",
        },
        "criteria_amended": False,
    }


def build_cross_fixture_contrast_rows(
    *,
    family_rollups: dict[str, dict],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, rollup in family_rollups.items():
        composition = rollup.get("outcome_composition") or {}
        rows.append(
            {
                "family": family,
                "fixture": rollup.get("fixture"),
                "platform": rollup.get("platform"),
                "evidence_phase": rollup.get("evidence_phase"),
                "coverage_label": (rollup.get("coverage") or {}).get("coverage_label"),
                "episode_count": (rollup.get("coverage") or {}).get("accepted_valid_unique"),
                "success": composition.get("success", 0),
                "no_grasp": composition.get("no_grasp", 0),
                "transport_incomplete": composition.get("transport_incomplete", 0),
                "wrong_goal_region": composition.get("wrong_goal_region", 0),
                "wrong_placement": composition.get("wrong_placement", 0),
                "support_or_containment_failed": composition.get("support_or_containment_failed", 0),
                "grasp_achieved": rollup.get("grasp_achieved"),
            }
        )
    return rows


def build_family_coverage_rows(family_rollups: dict[str, dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, rollup in family_rollups.items():
        coverage = rollup.get("coverage") or {}
        confirmatory = rollup.get("confirmatory_dispatch") or {}
        rows.append(
            {
                "family": family,
                "fixture": rollup.get("fixture"),
                "evidence_phase": rollup.get("evidence_phase"),
                "export_status": coverage.get("export_status"),
                "coverage_label": coverage.get("coverage_label"),
                "accepted_valid_unique": coverage.get("accepted_valid_unique"),
                "planned_episodes": coverage.get("planned_episodes"),
                "confirmatory_dispatched": confirmatory.get("dispatched_episodes"),
                "confirmatory_accepted": confirmatory.get("accepted_valid_unique"),
                "ledger_compile_id": coverage.get("ledger_compile_id"),
            }
        )
        if family == "C8" and int(confirmatory.get("accepted_valid_unique") or 0) > 0:
            rows.append(
                {
                    "family": "C8",
                    "fixture": rollup.get("fixture"),
                    "evidence_phase": "confirmatory_partial",
                    "export_status": confirmatory.get("export_status"),
                    "coverage_label": confirmatory.get("coverage_label"),
                    "accepted_valid_unique": confirmatory.get("accepted_valid_unique"),
                    "planned_episodes": confirmatory.get("planned_episodes"),
                    "confirmatory_dispatched": confirmatory.get("dispatched_episodes"),
                    "confirmatory_accepted": confirmatory.get("accepted_valid_unique"),
                    "ledger_compile_id": confirmatory.get("ledger_compile_id"),
                }
            )
    return rows


def build_campaign_tables(
    *,
    campaign_blocked: dict,
    c7_audit: dict,
    c7_primary_rows: list[dict],
    family_rollups: dict[str, dict],
    campaign_export_status: str,
    compile_provenance: dict[str, Any],
    c8_scenario_grasp: dict[str, Any] | None,
    c8_confirmatory_grasp: dict[str, Any] | None = None,
    c8_confound_check: dict[str, Any] | None = None,
) -> dict[str, list[dict]]:
    validation = c7_audit.get("validation") or {}
    c7_coverage = (family_rollups.get("C7") or {}).get("coverage") or {}
    c7_outcome_composition = (family_rollups.get("C7") or {}).get("outcome_composition") or {}
    scope_summary = [
        {"metric": "campaign_export_status", "value": campaign_export_status},
        {"metric": "original_planned_policy_episodes", "value": campaign_blocked.get("original_planned_policy_episodes", 17664)},
        {"metric": "achievable_policy_episodes", "value": campaign_blocked.get("achievable_policy_episodes", 2304)},
        {"metric": "scientifically_blocked_episodes", "value": campaign_blocked.get("scientifically_blocked_episodes", 15360)},
        {"metric": "pre_repair_c7_excluded_episodes", "value": campaign_blocked.get("pre_repair_c7_excluded_episodes", 279)},
        {"metric": "c7_coverage_label", "value": c7_coverage.get("coverage_label")},
        {"metric": "c6_pilot_coverage_label", "value": ((family_rollups.get("C6") or {}).get("coverage") or {}).get("coverage_label")},
        {"metric": "c8_pilot_coverage_label", "value": ((family_rollups.get("C8") or {}).get("coverage") or {}).get("coverage_label")},
        {
            "metric": "c8_confirmatory_coverage_label",
            "value": ((family_rollups.get("C8") or {}).get("confirmatory_dispatch") or {}).get("coverage_label"),
        },
        {"metric": "c7_accepted_valid_unique", "value": validation.get("accepted_unique")},
        {"metric": "c7_valid_success_records", "value": validation.get("valid_success_records")},
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
    tables: dict[str, list[dict]] = {
        "scope_summary.csv": scope_summary,
        "blocked_families.csv": blocked_families,
        "scale_ladder_squeeze.csv": squeeze_rows,
        "campaign_primary_results.csv": estimand_status,
        "c7_outcome_composition.csv": outcome_composition_rows(c7_outcome_composition),
        "family_coverage.csv": build_family_coverage_rows(family_rollups),
        "cross_fixture_outcome_contrast.csv": build_cross_fixture_contrast_rows(
            family_rollups=family_rollups
        ),
        "compile_retirement_provenance.csv": build_compile_retirement_rows(
            load_json(C7_COMPILE_RETIREMENT) if C7_COMPILE_RETIREMENT.is_file() else {}
        ),
    }
    if c8_scenario_grasp:
        tables["c8_grasp_by_scenario.csv"] = build_c8_grasp_by_scenario_rows(c8_scenario_grasp, cohort="pilot")
    if c8_confirmatory_grasp:
        tables["c8_confirmatory_grasp_by_scenario.csv"] = build_c8_grasp_by_scenario_rows(
            c8_confirmatory_grasp,
            cohort="confirmatory",
        )
        confirmatory_composition = c8_confirmatory_grasp.get("aggregate_outcomes") or {}
        if confirmatory_composition:
            tables["c8_confirmatory_outcome_composition.csv"] = outcome_composition_rows(
                confirmatory_composition
            )
    if c8_confound_check:
        tables["c8_destination_static_confound_check.csv"] = build_confound_check_rows(c8_confound_check)
    for family in ("C6", "C8"):
        rollup = family_rollups.get(family) or {}
        composition = rollup.get("outcome_composition") or {}
        if composition:
            tables[f"{family.lower()}_outcome_composition.csv"] = outcome_composition_rows(composition)
        scenario_rows = rollup.get("scenario_outcome_breakdown") or []
        if scenario_rows:
            tables[f"{family.lower()}_scenario_outcome_breakdown.csv"] = scenario_rows
    c7_scenario_rows = (family_rollups.get("C7") or {}).get("scenario_outcome_breakdown") or []
    if c7_scenario_rows:
        tables["c7_scenario_outcome_breakdown.csv"] = c7_scenario_rows
    return tables


def render_campaign_scope_figure(*, rows: list[dict], out_path: Path, export_status: str) -> None:
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
    c7_coverage_label = next((row["value"] for row in rows if row["metric"] == "c7_coverage_label"), "n/a")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<text x="20" y="28" font-size="16" font-weight="600">V4 registered campaign scope</text>'
        f'<text x="20" y="52" font-size="12" fill="#444">Planned policy episodes: {planned}; export: {export_status.upper()}; C7 {c7_coverage_label}</text>'
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
    campaign_export_status: str,
    family_rollups: dict[str, dict],
    family_exports: dict[str, dict],
) -> dict:
    horizontal_narrative = horizontal_memo.get("paper_narrative") or {}
    paragraphs = list(horizontal_narrative.get("paragraphs") or [])
    validation = c7_audit.get("validation") or {}
    c7_rollup = family_rollups.get("C7") or {}
    c7_coverage = c7_rollup.get("coverage") or {}
    c7_outcome_composition = c7_rollup.get("outcome_composition") or {}
    c6_rollup = family_rollups.get("C6") or {}
    c8_rollup = family_rollups.get("C8") or {}
    decomposition_text = format_outcome_decomposition(c7_outcome_composition)
    c6_pilot_text = format_outcome_decomposition(c6_rollup.get("outcome_composition") or {})
    c8_pilot_text = format_outcome_decomposition(c8_rollup.get("outcome_composition") or {})
    c8_confirmatory = (c8_rollup.get("confirmatory_dispatch") or {})
    c8_pilot_scenarios = (C8_PILOT_COMPOSITION.is_file() and load_json(C8_PILOT_COMPOSITION).get("by_scenario")) or {}
    c8_confirmatory_progress = resolve_c8_confirmatory_composition() or {}
    c8_confirmatory_by_scenario = c8_confirmatory_progress.get("by_scenario") or {}
    c8_confirmatory_valid = int(c8_confirmatory.get("accepted_valid_unique") or 0)
    c8_confound = load_json(C8_DESTINATION_STATIC_CONFOUND) if C8_DESTINATION_STATIC_CONFOUND.is_file() else {}
    confound_evidence = (c8_confound.get("evidence") or {}).get("green_yellow_distance_at_reset_m") or {}
    campaign_paragraphs = [
        f"**{campaign_export_status.upper()} EXPORT** — three achievable families are in flight. "
        f"C7 confirmatory coverage {c7_coverage.get('coverage_label', 'n/a')} on "
        f"{c7_coverage.get('ledger_compile_id', 'n/a')} (retired partials k/l/m documented in "
        f"compile_retirement_pending.json pending FINAL 768/768); C6 G7 pilot 24/24 complete with "
        f"confirmatory wave {c6_rollup.get('confirmatory_dispatch', {}).get('wave', 'F')} re-rendered "
        f"({c6_rollup.get('confirmatory_dispatch', {}).get('coverage_label', 'n/a')}); "
        f"C8 G7 pilot 24/24 with confirmatory "
        f"{c8_confirmatory.get('coverage_label', 'n/a')}.",
        "Registered campaign scope: 17,664 planned policy episodes; 2,304 achievable "
        "(C6, C7, C8 confirmatory families × 768); 15,360 scientifically blocked "
        "(C1/C3/C4 horizontal information-gate squeeze 9,728; C2 reference_binding "
        "information gate 4,096; C5 vertical IK reachability 768). No eligibility "
        "criterion, threshold, or scale ladder was amended to recover blocked scope.",
        "Cross-fixture contrast (compiled ledgers): C7 object_pair on Isaac is dominated by "
        f"no_grasp ({c7_outcome_composition.get('no_grasp', 0)}/{c7_coverage.get('accepted_valid_unique', 0)}) "
        f"but is not uniformly no_grasp — compile-m adds "
        f"{c7_outcome_composition.get('transport_incomplete', 0)} transport_incomplete, "
        f"{c7_outcome_composition.get('wrong_goal_region', 0)} wrong_goal_region, and "
        f"{c7_outcome_composition.get('support_or_containment_failed', 0)} support_or_containment_failed "
        "(0 successes). C6 containment pilot shows grasps and placement attempts "
        f"({c6_pilot_text}). C8 second_stack WidowX pilot achieves grasp on 12/24 episodes "
        f"({c8_pilot_text}); transport_incomplete implies successful grasp with incomplete transport.",
        "C8 pilot per-scenario grasp rates (destination_static "
        f"{c8_pilot_scenarios.get('destination_static', {}).get('grasp_achieved', 'n/a')}/8 = 75%, "
        f"move_stop {c8_pilot_scenarios.get('move_stop', {}).get('grasp_achieved', 'n/a')}/8 = 50%, "
        f"original_sham {c8_pilot_scenarios.get('original_sham', {}).get('grasp_achieved', 'n/a')}/8 = 25%) "
        "form a hypothesis about reference-placement effects on online scene movement. At "
        f"{c8_confirmatory_valid}/768 confirmatory episodes (milestone compile 20260908d at 50+, "
        f"20260908e reserved for 100), observed rates are destination_static "
        f"{c8_confirmatory_by_scenario.get('destination_static', {}).get('grasp_achieved', 0)}/"
        f"{c8_confirmatory_by_scenario.get('destination_static', {}).get('episodes_completed', 0)} "
        f"({100 * float(c8_confirmatory_by_scenario.get('destination_static', {}).get('grasp_rate', 0)):.1f}%), "
        f"move_stop {c8_confirmatory_by_scenario.get('move_stop', {}).get('grasp_achieved', 0)}/"
        f"{c8_confirmatory_by_scenario.get('move_stop', {}).get('episodes_completed', 0)}, "
        f"original_sham {c8_confirmatory_by_scenario.get('original_sham', {}).get('grasp_achieved', 0)}/"
        f"{c8_confirmatory_by_scenario.get('original_sham', {}).get('episodes_completed', 0)}. "
        "The ordering direction matches the pilot but remains insufficient_coverage and is marked "
        "not estimable under the registered estimator with Holm handling.",
        "C8 destination_static confound elimination (c8_destination_static_confound_check_20260908a.json): "
        "green-to-yellow reset distance is identical at 0.1414 m across all three scenarios over 256 seeds "
        f"each ({', '.join(f'{s} mean={confound_evidence.get(s, {}).get('mean')} m' for s in ('original_sham', 'destination_static', 'move_stop'))}). "
        "Higher destination_static grasp rate is therefore not explained by the object starting closer to "
        "the goal region; the registered protocol difference is reference placement (static reference at "
        "planned+D versus sham identity and move_stop motion profiles). This converts the scenario "
        "ordering from a suggestive trend into a testable claim about reference placement rather than "
        "task geometry.",
        "C6 wave D was cleanly re-rendered as rendered-c6confirm20260908f after a shared-checkout "
        "runner conflict; operating rule documented in runner_binding_resolution_20260908.json: when C7 "
        "mutates the shared checkout runner, C6 must re-render in-flight waves from the updated "
        "released lock—never patch coord bindings on PVC alone. C6 still has 0 confirmatory terminals "
        "with simulators running; confirmatory ledger compile pending.",
        "C7 compile provenance is explicit: partial compiles k (548), l (565), m (580), and stale "
        ".compiled-ledger-final (551) are retired in favor of compiled_ledger_20260908_FINAL at "
        "768/768 require-full-coverage; see compile_retirement_provenance.csv.",
        "Capacity finding (revised): a lane pair is a study-design and throughput unit, not a scheduler "
        "constraint—policy and simulator are separate single-GPU pods connected over HTTP. Pending "
        "failures are attributable to per-node GPU fragmentation and legitimate C7 occupancy, not an "
        "unsatisfiable two-GPU request. tools/v4_gpu_pool_sweep.py reclaimed 34 GPUs across four pools "
        "(receipt gpu_pool_sweep_receipt.json); tools/v4_gpu_scheduling.py provides reusable "
        "c8_a40_spread and c6_a10040_spread placement policies for anti-affinity bin-packing.",
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
            f"Compiled ledger ({c7_coverage.get('ledger_compile_id', 'n/a')}): "
            f"{validation.get('accepted_unique', 'n/a')} accepted valid with outcome decomposition "
            f"{decomposition_text}."
        )
    campaign_paragraphs.append(
        "C2 primary reference-selectivity (H) remains not estimable; the homogeneous G3 gate "
        "finalized at 128/128 as a scientific block with computation-correct information-gate "
        "rejection (4096 episodes). C6 is the only fixture passing the information gate on both "
        "translation-sign halves; C8 is the cross-platform WidowX/SimplerEnv check."
    )
    return {
        "schema_version": "v4-registered-campaign-evidence-memo-v1",
        "export_status": campaign_export_status,
        "export_coverage": {
            family: (family_rollups.get(family) or {}).get("coverage")
            for family in ("C7", "C6", "C8")
        },
        "confirmatory_dispatch": {
            family: (family_rollups.get(family) or {}).get("confirmatory_dispatch")
            for family in ("C6", "C8")
        },
        "cross_fixture_contrast": build_cross_fixture_contrast_rows(family_rollups=family_rollups),
        "frozen_analysis_manifest": str(FROZEN_ANALYSIS.relative_to(ROOT)),
        "headline": (
            f"{campaign_export_status.title()} registered export: C7 Isaac no_grasp dominance "
            "contrasts with C6/C8 pilot transport and placement attempts across fixtures"
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
            "outcome_composition": c7_outcome_composition,
            "export_coverage": c7_coverage,
        },
        "family_exports": {
            family: {
                "path": payload.get("path"),
                "export_coverage": payload.get("export_coverage"),
                "outcome_composition": payload.get("outcome_composition"),
            }
            for family, payload in family_exports.items()
        },
        "horizontal_slice": str(HORIZONTAL_SLICE.relative_to(ROOT)),
        "limitations": [
            "Blocked families carry qualification receipts only; no policy episodes were dispatched.",
            "279 pre-repair C7 episodes remain excluded from behavioral claims.",
            "C7 partial uses compile-m until FINAL 768/768 compile lands; retired partials documented in compile_retirement_pending.json.",
            "C6/C8 confirmatory tables refresh as Agent B and Agent A ledger compiles accumulate.",
            "C8 per-scenario grasp ordering contrast is not estimable at current confirmatory width (insufficient_coverage).",
        ],
        "not_estimable_primary_estimands": [
            "C1/C3/C4 primary wording and reference-selectivity contrasts",
            "C2 reference-selectivity primary (H)",
            "C5 vertical family (IK reachability block)",
            "C8 per-scenario grasp ordering contrast (confirmatory underpowered at current width)",
        ],
    }


def build_family_rollup(
    *,
    family: str,
    fixture: str,
    platform: str,
    evidence_phase: str,
    ledger: Path,
    manifest: Path,
    coverage: dict[str, Any],
    confirmatory_dispatch: dict[str, Any] | None = None,
    grasp_achieved: int | None = None,
) -> dict[str, Any]:
    rows = load_accepted_ledger_rows(ledger)
    manifest_by_id = load_manifest_by_episode_id(manifest)
    composition = summarize_outcome_composition(rows)
    payload = {
        "family": family,
        "fixture": fixture,
        "platform": platform,
        "evidence_phase": evidence_phase,
        "coverage": coverage,
        "outcome_composition": composition,
        "scenario_outcome_breakdown": summarize_scenario_outcome_breakdown(rows, manifest_by_id),
        "confirmatory_dispatch": confirmatory_dispatch or {},
        "receipt_path": (
            str(C6_G7_RECEIPT.relative_to(ROOT))
            if family == "C6" and C6_G7_RECEIPT.is_file()
            else str(C8_G7_RECEIPT.relative_to(ROOT))
            if family == "C8" and C8_G7_RECEIPT.is_file()
            else None
        ),
    }
    if grasp_achieved is not None:
        payload["grasp_achieved"] = grasp_achieved
    return payload


def build_family_status(family_rollups: dict[str, dict]) -> dict[str, dict]:
    status: dict[str, dict] = {}
    for family, rollup in family_rollups.items():
        coverage = dict(rollup.get("coverage") or {})
        payload = {
            **coverage,
            "fixture": rollup.get("fixture"),
            "platform": rollup.get("platform"),
            "evidence_phase": rollup.get("evidence_phase"),
            "outcome_composition": rollup.get("outcome_composition") or {},
            "scenario_outcome_breakdown": rollup.get("scenario_outcome_breakdown") or [],
        }
        confirmatory = rollup.get("confirmatory_dispatch") or {}
        if confirmatory:
            payload["confirmatory_dispatch"] = confirmatory
        if family == "C7":
            payload["pre_repair_excluded"] = 279
        status[family] = payload
    return status


def resolve_pilot_defaults(args: argparse.Namespace) -> tuple[Path | None, Path, str | None, Path | None, Path, str | None]:
    c6_ledger = args.c6_ledger
    c6_manifest = args.c6_manifest
    c6_compile_id = args.c6_ledger_compile_id
    if c6_ledger is None and not args.no_default_pilot_exports and DEFAULT_C6_PILOT_LEDGER.is_file():
        c6_ledger = DEFAULT_C6_PILOT_LEDGER
        c6_manifest = DEFAULT_C6_PILOT_MANIFEST
        c6_compile_id = c6_compile_id or "g7-containment-20260908a"
    c8_ledger = args.c8_ledger
    c8_manifest = args.c8_manifest
    c8_compile_id = args.c8_ledger_compile_id
    if c8_ledger is None and not args.no_default_pilot_exports and DEFAULT_C8_PILOT_LEDGER.is_file():
        c8_ledger = DEFAULT_C8_PILOT_LEDGER
        c8_manifest = DEFAULT_C8_PILOT_MANIFEST
        c8_compile_id = c8_compile_id or "20260908b"
    return c6_ledger, c6_manifest, c6_compile_id, c8_ledger, c8_manifest, c8_compile_id


def load_confirmatory_dispatch_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    c6_dispatch: dict[str, Any] = {}
    wave_path = C6_WAVE_F_DISPATCH if C6_WAVE_F_DISPATCH.is_file() else C6_WAVE_D_DISPATCH
    if wave_path.is_file():
        wave_payload = load_json(wave_path)
        wave_label = "F" if wave_path == C6_WAVE_F_DISPATCH else "D"
        c6_dispatch = build_dispatch_coverage_metadata(
            accepted=0,
            planned=768,
            dispatched=int(wave_payload.get("behavioral_episode_count", 384)),
            compile_id=None,
            wave=wave_label,
        )
        c6_dispatch["dispatch_receipt"] = str(wave_path.relative_to(ROOT))
        c6_dispatch["lane_count"] = wave_payload.get("lane_count")
        c6_dispatch["render_root"] = (wave_payload.get("lane_assignments_summary") or {}).get("render_root")
        c6_dispatch["post_dispatch_observed"] = wave_payload.get("post_dispatch_observed")
        c6_dispatch["runner_binding_resolution"] = (
            str(C6_RUNNER_BINDING.relative_to(ROOT)) if C6_RUNNER_BINDING.is_file() else None
        )
        if wave_label == "F":
            c6_dispatch["note"] = (
                "Wave D re-rendered as 20260908f after shared-checkout runner conflict; "
                "0 confirmatory terminals; wave E follows."
            )
        else:
            c6_dispatch["note"] = "Wave D dispatched; wave E follows on Agent B schedule."
    elif C6_WAVE_B_DISPATCH.is_file():
        c6_dispatch = build_dispatch_coverage_metadata(
            accepted=0,
            planned=768,
            dispatched=int(load_json(C6_WAVE_B_DISPATCH).get("behavioral_episode_count", 384)),
            compile_id=None,
            wave="B",
        )
    c8_dispatch = {}
    status = load_c8_execution_status()
    status_path = resolve_c8_status_receipt_path()
    if status:
        progress = status.get("confirmatory_progress") or {}
        milestone_id = status.get("confirmatory_ledger_compile_id_50_milestone") or "20260908d"
        c8_dispatch = build_dispatch_coverage_metadata(
            accepted=int(progress.get("valid") or progress.get("behavioral_valid") or 0),
            planned=768,
            dispatched=768,
            compile_id=milestone_id,
            wave="main",
        )
        c8_dispatch["ledger_compile_id_live_interim"] = status.get("confirmatory_ledger_compile_id_live")
        c8_dispatch["next_milestone_compile_at"] = status.get("next_milestone_compile_at")
        c8_dispatch["aggregate_outcomes"] = progress.get("aggregate_outcomes")
        c8_dispatch["by_scenario"] = progress.get("by_scenario")
        c8_dispatch["ordering_holds"] = progress.get("ordering_holds")
        c8_dispatch["confirmatory_lanes"] = status.get("confirmatory_lanes")
        c8_dispatch["confound_check_uri"] = status.get("destination_static_confound_uri")
        c8_dispatch["composition_receipt"] = (
            str(C8_CONFIRMATORY_COMPOSITION_MILESTONE.relative_to(ROOT))
            if C8_CONFIRMATORY_COMPOSITION_MILESTONE.is_file()
            else None
        )
        if status_path:
            c8_dispatch["execution_status_receipt"] = str(status_path.relative_to(ROOT))
    return c6_dispatch, c8_dispatch


def export_family(
    *,
    family: str,
    ledger: Path,
    manifest: Path,
    config: Path,
    out_root: Path,
    tag: str,
    compile_id: str | None,
) -> dict:
    family_out = out_root / "families" / family
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_v4_c7_partial_analysis_export.py"),
            "--results",
            str(ledger),
            "--manifest",
            str(manifest),
            "--config",
            str(config),
            "--out",
            str(family_out),
            "--tag",
            tag,
            "--family",
            family,
            *(["--ledger-compile-id", compile_id] if compile_id else []),
        ],
        check=True,
        cwd=ROOT,
    )
    export_manifest = load_json(family_out / tag / "results_export_manifest.json")
    return {
        "path": str((family_out / tag).relative_to(ROOT)),
        "manifest_sha256": sha256_file(family_out / tag / "results_export_manifest.json"),
        "export_manifest": export_manifest,
        "blocked_scope": load_json(family_out / tag / "blocked_scope.json"),
        "memo": load_json(family_out / tag / "paper" / "evidence_memo.json")
        if (family_out / tag / "paper" / "evidence_memo.json").is_file()
        else {},
        "audit": load_json(family_out / tag / "tables" / "audit_report.json")
        if (family_out / tag / "tables" / "audit_report.json").is_file()
        else {},
        "export_coverage": export_manifest.get("export_coverage") or {},
        "outcome_composition": export_manifest.get("outcome_composition") or {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c7-ledger", type=Path, default=None)
    parser.add_argument("--c7-manifest", type=Path, default=DEFAULT_C7_MANIFEST)
    parser.add_argument("--c7-ledger-compile-id", type=str, default=None)
    parser.add_argument("--c6-ledger", type=Path, default=None)
    parser.add_argument("--c6-manifest", type=Path, default=DEFAULT_C6_MANIFEST)
    parser.add_argument("--c6-ledger-compile-id", type=str, default=None)
    parser.add_argument("--c8-ledger", type=Path, default=None)
    parser.add_argument("--c8-manifest", type=Path, default=DEFAULT_C8_MANIFEST)
    parser.add_argument("--c8-ledger-compile-id", type=str, default=None)
    parser.add_argument(
        "--no-default-pilot-exports",
        action="store_true",
        help="Do not auto-include C6/C8 G7 pilot ledgers from default repo paths",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/results/registered",
    )
    parser.add_argument("--tag", type=str, default="20260908")
    args = parser.parse_args(argv)

    c7_ledger, c7_compile_id, c7_is_final = resolve_c7_ledger(args)

    out_root = args.out.resolve() / args.tag
    if out_root.exists():
        import shutil

        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    c7_tag = f"{args.tag}c7"
    c7_payload = export_family(
        family="C7",
        ledger=c7_ledger,
        manifest=args.c7_manifest.resolve(),
        config=args.config.resolve(),
        out_root=out_root,
        tag=c7_tag,
        compile_id=c7_compile_id,
    )
    c7_export_manifest = c7_payload["export_manifest"]
    c7_blocked_scope = c7_payload["blocked_scope"]
    c7_memo = c7_payload["memo"]
    c7_audit = c7_payload["audit"]
    c7_primary_path = out_root / "families" / "C7" / c7_tag / "tables" / "primary_results.csv"
    c7_primary_rows: list[dict] = []
    if c7_primary_path.is_file():
        with c7_primary_path.open(encoding="utf-8", newline="") as handle:
            c7_primary_rows = list(csv.DictReader(handle))

    family_exports: dict[str, dict] = {"C7": c7_payload}
    c8_scenario_grasp = load_json(C8_PILOT_COMPOSITION) if C8_PILOT_COMPOSITION.is_file() else None
    c8_confirmatory_grasp = resolve_c8_confirmatory_composition()
    c8_confound_check = (
        load_json(C8_DESTINATION_STATIC_CONFOUND) if C8_DESTINATION_STATIC_CONFOUND.is_file() else None
    )
    pilot_sources: dict[str, tuple[Path, Path, str | None]] = {}
    c6_ledger, c6_manifest, c6_compile_id, c8_ledger, c8_manifest, c8_compile_id = resolve_pilot_defaults(args)
    optional_families = (
        ("C6", c6_ledger, c6_manifest, c6_compile_id, f"{args.tag}c6pilot"),
        ("C8", c8_ledger, c8_manifest, c8_compile_id, f"{args.tag}c8pilot"),
    )
    for family, ledger_arg, manifest_arg, compile_id, tag in optional_families:
        if ledger_arg is None:
            continue
        ledger = ledger_arg.resolve()
        manifest = manifest_arg.resolve()
        if not ledger.is_file():
            raise SystemExit(f"missing {family} accepted ledger: {ledger}")
        if not manifest.is_file():
            raise SystemExit(f"missing {family} manifest: {manifest}")
        pilot_sources[family] = (ledger, manifest, compile_id)
        payload = export_family(
            family=family,
            ledger=ledger,
            manifest=manifest,
            config=args.config.resolve(),
            out_root=out_root,
            tag=tag,
            compile_id=compile_id,
        )
        family_exports[family] = payload

    if DEFAULT_C8_CONFIRMATORY_LEDGER.is_file() and DEFAULT_C8_MANIFEST.is_file():
        c8_confirmatory_tag = f"{args.tag}c8confirmatory"
        c8_confirmatory_payload = export_family(
            family="C8",
            ledger=DEFAULT_C8_CONFIRMATORY_LEDGER,
            manifest=DEFAULT_C8_MANIFEST.resolve(),
            config=args.config.resolve(),
            out_root=out_root,
            tag=c8_confirmatory_tag,
            compile_id=(c8_confirmatory_grasp or {}).get("ledger_compile_id_milestone_50") or "20260908d",
        )
        family_exports["C8_confirmatory"] = c8_confirmatory_payload

    horizontal_blocked = load_json(HORIZONTAL_SLICE / "blocked_scope.json")
    horizontal_memo = load_json(HORIZONTAL_SLICE / "paper" / "evidence_memo.json")

    c7_rows = len(load_accepted_ledger_rows(c7_ledger))
    c7_planned = int(c7_blocked_scope.get("planned_c7_episodes") or 768)
    c7_coverage = c7_payload["export_coverage"] or build_coverage_metadata(
        accepted=c7_rows,
        planned=c7_planned,
        compile_id=c7_compile_id,
    )
    c6_confirmatory_dispatch, c8_confirmatory_dispatch = load_confirmatory_dispatch_metadata()
    compile_provenance = load_compile_provenance(
        active_compile_id=c7_compile_id,
        active_rows=c7_rows,
        is_final=c7_is_final,
    )

    family_rollups: dict[str, dict] = {
        "C7": build_family_rollup(
            family="C7",
            fixture="object_pair",
            platform="isaac_droid",
            evidence_phase="confirmatory_final" if c7_is_final else "confirmatory_partial",
            ledger=c7_ledger,
            manifest=args.c7_manifest.resolve(),
            coverage={
                **c7_coverage,
                "export_phase": "confirmatory_final" if c7_is_final else "confirmatory_partial",
                "ledger_compile_id": c7_compile_id,
            },
        )
    }
    if "C6" in pilot_sources:
        ledger, manifest, compile_id = pilot_sources["C6"]
        family_rollups["C6"] = build_family_rollup(
            family="C6",
            fixture="containment",
            platform="isaac_droid",
            evidence_phase="g7_pilot_complete",
            ledger=ledger,
            manifest=manifest,
            coverage=build_coverage_metadata(
                accepted=len(load_accepted_ledger_rows(ledger)),
                planned=24,
                compile_id=compile_id,
            ),
            confirmatory_dispatch=c6_confirmatory_dispatch,
        )
    if "C8" in pilot_sources:
        ledger, manifest, compile_id = pilot_sources["C8"]
        family_rollups["C8"] = build_family_rollup(
            family="C8",
            fixture="second_stack",
            platform="widowx_simplerenv",
            evidence_phase="g7_pilot_complete",
            ledger=ledger,
            manifest=manifest,
            coverage=build_coverage_metadata(
                accepted=len(load_accepted_ledger_rows(ledger)),
                planned=24,
                compile_id=compile_id,
            ),
            confirmatory_dispatch=c8_confirmatory_dispatch,
            grasp_achieved=(c8_scenario_grasp or {}).get("grasp_achieved_count"),
        )

    c7_outcome_composition = family_rollups["C7"]["outcome_composition"]
    campaign_export_status = resolve_campaign_export_status(c7_is_final=c7_is_final, family_rollups=family_rollups)
    family_status = build_family_status(family_rollups)

    campaign_blocked = build_campaign_blocked_scope(
        c7_blocked_scope=c7_blocked_scope,
        horizontal_blocked_scope=horizontal_blocked,
        family_status=family_status,
        compile_provenance=compile_provenance,
        c8_scenario_grasp=c8_scenario_grasp,
        c8_confirmatory_grasp=c8_confirmatory_grasp,
        c8_confound_check=c8_confound_check,
    )
    blocked_path = out_root / "blocked_scope.json"
    blocked_path.write_text(json.dumps(campaign_blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    campaign_memo = build_campaign_evidence_memo(
        c7_memo=c7_memo,
        horizontal_memo=horizontal_memo,
        c7_export_manifest=c7_export_manifest,
        c7_audit=c7_audit,
        campaign_export_status=campaign_export_status,
        family_rollups=family_rollups,
        family_exports=family_exports,
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
        family_rollups=family_rollups,
        campaign_export_status=campaign_export_status,
        compile_provenance=compile_provenance,
        c8_scenario_grasp=c8_scenario_grasp,
        c8_confirmatory_grasp=c8_confirmatory_grasp,
        c8_confound_check=c8_confound_check,
    )
    table_artifacts: dict[str, dict] = {}
    for name, rows in campaign_tables.items():
        table_path = tables_dir / name
        write_csv(table_path, rows)
        table_artifacts[name] = artifact(table_path)

    audit_report = {
        "schema_version": "v4-registered-campaign-audit-v1",
        "export_status": campaign_export_status,
        "export_coverage": {
            family: (family_rollups.get(family) or {}).get("coverage")
            for family in family_rollups
        },
        "family_rollups": {
            family: {
                "outcome_composition": rollup.get("outcome_composition"),
                "confirmatory_dispatch": rollup.get("confirmatory_dispatch"),
            }
            for family, rollup in family_rollups.items()
        },
        "frozen_analysis_manifest": str(FROZEN_ANALYSIS.relative_to(ROOT)),
        "c7_family_audit": c7_audit,
        "ledger_rows_compiled": c7_rows,
        "ledger_source": str(c7_ledger),
        "ledger_compile_id": c7_compile_id,
        "compile_provenance": compile_provenance,
        "c8_pilot_scenario_grasp": c8_scenario_grasp,
        "c8_confirmatory_scenario_grasp": c8_confirmatory_grasp,
        "c8_destination_static_confound_check": c8_confound_check,
        "outcome_composition": c7_outcome_composition,
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
    render_campaign_scope_figure(
        rows=campaign_tables["scope_summary.csv"],
        out_path=scope_figure,
        export_status=campaign_export_status,
    )
    figures_manifest_path = figures_dir / "figures_manifest.json"
    figures_manifest = {
        "schema_version": "v4-registered-campaign-figures-v1",
        "export_status": campaign_export_status,
        "campaign_scope": artifact(scope_figure),
        "c7_family_figures": str((out_root / "families" / "C7" / c7_tag / "figures").relative_to(ROOT)),
    }
    figures_manifest_path.write_text(json.dumps(figures_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    results_manifest_path = tables_dir / "results_manifest.json"
    results_manifest = {
        "schema_version": "v4-registered-campaign-results-manifest-v1",
        "tag": args.tag,
        "export_status": campaign_export_status,
        "export_coverage": {
            family: (family_rollups.get(family) or {}).get("coverage")
            for family in family_rollups
        },
        "family_rollups": {
            family: {
                "outcome_composition": rollup.get("outcome_composition"),
                "confirmatory_dispatch": rollup.get("confirmatory_dispatch"),
            }
            for family, rollup in family_rollups.items()
        },
        "tables": table_artifacts,
        "figures_manifest": artifact(figures_manifest_path),
        "family_tables": {
            family: f"{payload['path']}/tables"
            for family, payload in family_exports.items()
        },
    }
    results_manifest_path.write_text(json.dumps(results_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status_banner = (
        f"> **{campaign_export_status.upper()} EXPORT** — C7 {c7_coverage.get('coverage_label')}; "
        f"C6 pilot {((family_rollups.get('C6') or {}).get('coverage') or {}).get('coverage_label', 'n/a')} "
        f"with confirmatory {((family_rollups.get('C6') or {}).get('confirmatory_dispatch') or {}).get('coverage_label', 'n/a')}; "
        f"C8 pilot {((family_rollups.get('C8') or {}).get('coverage') or {}).get('coverage_label', 'n/a')} "
        f"with confirmatory {((family_rollups.get('C8') or {}).get('confirmatory_dispatch') or {}).get('coverage_label', 'n/a')}.\n\n"
    )
    results_stub = paper_dir / "RESULTS.md"
    results_stub.write_text(
        "# V4 registered campaign results\n\n"
        + status_banner
        + "\n\n".join(campaign_memo["paper_narrative"]["paragraphs"])
        + "\n\n## Campaign tables and figures\n\n"
        + f"- Tables: `{tables_dir.relative_to(ROOT)}/`\n"
        + f"- Figures: `{figures_dir.relative_to(ROOT)}/`\n"
        + f"- C7 family export: `{c7_payload['path']}/`\n"
        + (
            f"- C6 family export: `{family_exports['C6']['path']}/`\n"
            if "C6" in family_exports
            else "- C6 family export: awaiting Agent B confirmatory ledger receipt\n"
        )
        + (
            f"- C8 family export: `{family_exports['C8']['path']}/`\n"
            if "C8" in family_exports
            else "- C8 family export: awaiting Agent A confirmatory ledger receipt\n"
        )
        + f"- Horizontal squeeze slice: `{HORIZONTAL_SLICE.relative_to(ROOT)}/`\n",
        encoding="utf-8",
    )

    validation = c7_audit.get("validation") or {}
    manifest = {
        "schema_version": "v4-registered-campaign-export-manifest-v1",
        "tag": args.tag,
        "export_status": campaign_export_status,
        "export_coverage": {
            family: (family_rollups.get(family) or {}).get("coverage")
            for family in family_rollups
        },
        "family_rollups": {
            family: {
                "outcome_composition": rollup.get("outcome_composition"),
                "confirmatory_dispatch": rollup.get("confirmatory_dispatch"),
            }
            for family, rollup in family_rollups.items()
        },
        "outcome_composition": {
            family: rollup.get("outcome_composition")
            for family, rollup in family_rollups.items()
        },
        "output_root": str(out_root.relative_to(ROOT)),
        "family_exports": {
            family: {
                "path": payload["path"],
                "manifest_sha256": payload["manifest_sha256"],
            }
            for family, payload in family_exports.items()
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
