#!/usr/bin/env python3
"""Audit C7 duplicate-lane evidence after wave-005 redundant dispatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def attempt_dirs(pvc: Path, lane: str) -> dict[str, str]:
    """Map attempt_id -> relative attempt directory under pvc root."""
    candidates: dict[str, list[str]] = {}
    for path in pvc.rglob(f"lane-{lane}/attempt-attempt*"):
        if not path.is_dir() or path.parent.name != f"lane-{lane}":
            continue
        attempt_id = path.name.removeprefix("attempt-")
        rel = str(path.relative_to(pvc))
        candidates.setdefault(attempt_id, []).append(rel)

    mapping: dict[str, str] = {}
    for attempt_id, paths in candidates.items():
        def score(rel: str) -> tuple[int, int]:
            complete_count = len(list((pvc / rel).rglob("COMPLETE.json")))
            is_runtime = 1 if rel.startswith(".lane-runtime/") else 0
            return (complete_count, -is_runtime)

        mapping[attempt_id] = max(paths, key=score)
    return mapping


def complete_episodes(pvc: Path, attempt_dir: str | None) -> list[str]:
    if attempt_dir is None:
        return []
    base = pvc / attempt_dir
    if not base.is_dir():
        return []
    episodes: set[str] = set()
    for complete in base.rglob("COMPLETE.json"):
        rel_parts = complete.relative_to(base).parts
        if "episodes" in rel_parts:
            episode_index = rel_parts.index("episodes") + 1
            if episode_index < len(rel_parts):
                episode_id = rel_parts[episode_index]
                if episode_id.startswith("online_correction_v4-"):
                    episodes.add(episode_id)
                    continue
        parent = complete.parent
        if parent.name.startswith("online_correction_v4-"):
            episodes.add(parent.name)
    return sorted(episodes)


def load_ledger(path: Path) -> tuple[set[str], dict[str, Any]]:
    episode_ids: set[str] = set()
    by_episode: dict[str, Any] = {}
    if not path.is_file():
        return episode_ids, by_episode
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode_id = row.get("episode_id")
        if isinstance(episode_id, str):
            episode_ids.add(episode_id)
            by_episode[episode_id] = row
    return episode_ids, by_episode


def build_audit(
    *,
    pvc_root: Path,
    ledger_path: Path,
    owner_start: int = 307,
    redundant_start: int = 471,
    lane_count: int = 25,
) -> dict[str, Any]:
    owners = {f"c7m{index:02d}": f"attempt{owner_start + index:04d}" for index in range(lane_count)}
    redundant = {f"c7m{index:02d}": f"attempt{redundant_start + index:04d}" for index in range(lane_count)}
    ledger_eps, ledger_by_ep = load_ledger(ledger_path)

    lanes: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for lane in sorted(owners):
        owner_att = owners[lane]
        red_att = redundant[lane]
        by_attempt = attempt_dirs(pvc_root, lane)
        owner_dir = by_attempt.get(owner_att)
        red_dir = by_attempt.get(red_att)
        owner_eps = complete_episodes(pvc_root, owner_dir)
        red_eps = complete_episodes(pvc_root, red_dir)
        overlap = sorted(set(owner_eps) & set(red_eps))
        red_in_ledger = [episode_id for episode_id in red_eps if episode_id in ledger_eps]
        lane_ambiguous: list[dict[str, Any]] = []
        for episode_id in overlap:
            lane_ambiguous.append(
                {"episode_id": episode_id, "reason": "both_attempts_complete", "lane_id": lane}
            )
        for episode_id in red_in_ledger:
            lane_ambiguous.append(
                {"episode_id": episode_id, "reason": "redundant_attempt_in_ledger", "lane_id": lane}
            )
        lanes.append(
            {
                "lane_id": lane,
                "owner_attempt_id": owner_att,
                "redundant_attempt_id": red_att,
                "owner_attempt_dir": owner_dir,
                "redundant_attempt_dir": red_dir,
                "owner_complete_count": len(owner_eps),
                "redundant_complete_count": len(red_eps),
                "overlap_episode_ids": overlap,
                "redundant_in_ledger": red_in_ledger,
                "ambiguous": lane_ambiguous,
            }
        )
        ambiguous.extend(lane_ambiguous)

    duplicate_ledger_ids = len(ledger_eps) != len(ledger_by_ep)
    return {
        "schema_version": "v4-c7-duplicate-lane-audit-v1",
        "pvc_root": str(pvc_root),
        "accepted_ledger_path": str(ledger_path),
        "lanes_affected": len(lanes),
        "lanes_with_overlap": sum(1 for row in lanes if row["overlap_episode_ids"]),
        "lanes_with_redundant_complete": sum(1 for row in lanes if row["redundant_complete_count"] > 0),
        "redundant_complete_total": sum(row["redundant_complete_count"] for row in lanes),
        "accepted_ledger_rows": len(ledger_eps),
        "accepted_ledger_unique_episode_ids": len(ledger_eps),
        "duplicate_episode_ids_in_ledger": duplicate_ledger_ids,
        "accepted_evidence_invalidated_count": len(ambiguous),
        "ambiguous_episodes": ambiguous,
        "lanes": lanes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pvc-root",
        type=Path,
        default=Path("/data/users/ali/vla_wam/raw/v4/c7-object-pair-main"),
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger_path = args.ledger_path or (
        args.pvc_root / "compiled_ledger_20260907g/accepted_ledger.jsonl"
    )
    payload = build_audit(pvc_root=args.pvc_root, ledger_path=ledger_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
