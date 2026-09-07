"""Shared helpers for registered V4 campaign and family partial exports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_accepted_ledger_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def summarize_outcome_composition(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("success"):
            counts["success"] += 1
            continue
        outcome = row.get("outcome") or {}
        label = outcome.get("failure_label") or row.get("failure_label") or "unknown"
        counts[str(label)] += 1
    ordered = (
        "success",
        "no_grasp",
        "transport_incomplete",
        "wrong_goal_region",
        "wrong_placement",
        "unknown",
    )
    payload = {key: counts.get(key, 0) for key in ordered if counts.get(key, 0)}
    for label, value in sorted(counts.items()):
        if label not in payload:
            payload[label] = value
    return payload


def build_coverage_metadata(
    *,
    accepted: int,
    planned: int,
    compile_id: str | None = None,
) -> dict[str, Any]:
    planned = max(int(planned), 1)
    accepted = int(accepted)
    fraction = accepted / planned
    return {
        "export_status": "complete" if accepted >= planned else "partial",
        "accepted_valid_unique": accepted,
        "planned_episodes": planned,
        "missing_episodes": max(planned - accepted, 0),
        "coverage_fraction": round(fraction, 6),
        "coverage_label": f"{accepted}/{planned} ({100.0 * fraction:.1f}%)",
        "ledger_compile_id": compile_id,
    }


def format_outcome_decomposition(composition: dict[str, int]) -> str:
    parts = [f"{count} {label}" for label, count in composition.items() if count]
    return ", ".join(parts) if parts else "no accepted outcomes"


def outcome_composition_rows(composition: dict[str, int]) -> list[dict[str, Any]]:
    return [{"failure_label": label, "count": count} for label, count in composition.items() if count]


def load_manifest_by_episode_id(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode_id = row.get("episode_id")
        if isinstance(episode_id, str):
            by_id[episode_id] = row
    return by_id


def summarize_scenario_outcome_breakdown(
    rows: list[dict[str, Any]],
    manifest_by_episode_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], int] = {}
    for row in rows:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str):
            continue
        manifest_row = manifest_by_episode_id.get(episode_id, {})
        factors = manifest_row.get("factors") or {}
        scenario = str(factors.get("scenario") or "unknown")
        if row.get("success"):
            label = "success"
        else:
            outcome = row.get("outcome") or {}
            label = str(outcome.get("failure_label") or row.get("failure_label") or "unknown")
        grouped[(scenario, label)] = grouped.get((scenario, label), 0) + 1
    return [
        {"scenario": scenario, "failure_label": label, "count": count}
        for (scenario, label), count in sorted(grouped.items())
    ]


def build_dispatch_coverage_metadata(
    *,
    accepted: int,
    planned: int,
    dispatched: int,
    compile_id: str | None = None,
    wave: str | None = None,
    export_phase: str = "confirmatory",
) -> dict[str, Any]:
    payload = build_coverage_metadata(
        accepted=accepted,
        planned=planned,
        compile_id=compile_id,
    )
    payload["export_phase"] = export_phase
    payload["dispatched_episodes"] = int(dispatched)
    payload["dispatch_status"] = "dispatched_pending_capacity" if dispatched and not accepted else "dispatched"
    if wave:
        payload["wave"] = wave
    if dispatched and accepted < planned:
        payload["export_status"] = "dispatched_partial"
        payload["coverage_label"] = (
            f"{accepted}/{planned} confirmatory accepted; {dispatched}/{planned} dispatched"
        )
    return payload
