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
    ordered = ("success", "no_grasp", "transport_incomplete", "wrong_placement", "unknown")
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
