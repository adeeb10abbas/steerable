#!/usr/bin/env python3
"""Report C6 confirmatory grasp rate by scenario from PVC COMPLETE.json terminals."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENARIOS = ("original_sham", "destination_static", "move_stop")
TRANSPORT_INCOMPLETE = "transport_incomplete"
WRONG_GOAL_REGION = "wrong_goal_region"


def scan_terminals(attempts_root: Path) -> dict:
    by_scenario: dict[str, dict] = defaultdict(
        lambda: {
            "episodes": 0,
            "grasp_achieved_c6_rule": 0,
            "grasp_achieved_transport_incomplete_only": 0,
            "wrong_goal_region": 0,
            "no_grasp": 0,
            "outcomes": defaultdict(int),
        }
    )
    for complete in attempts_root.rglob("COMPLETE.json"):
        episode_json = complete.parent / "episode.json"
        if not episode_json.is_file():
            continue
        row = json.loads(episode_json.read_text(encoding="utf-8"))
        scenario = str(row.get("scenario") or "?")
        label = str(row.get("failure_label") or "?")
        bucket = by_scenario[scenario]
        bucket["episodes"] += 1
        bucket["outcomes"][label] += 1
        if label == "no_grasp":
            bucket["no_grasp"] += 1
        else:
            bucket["grasp_achieved_c6_rule"] += 1
        if label == TRANSPORT_INCOMPLETE:
            bucket["grasp_achieved_transport_incomplete_only"] += 1
        if label == WRONG_GOAL_REGION:
            bucket["wrong_goal_region"] += 1
    payload = {
        "terminals": sum(v["episodes"] for v in by_scenario.values()),
        "grasp_definition_notes": {
            "c6_rule": "failure_label != no_grasp (includes wrong_goal_region, transport_incomplete, etc.)",
            "transport_incomplete_only": "failure_label == transport_incomplete (C8-comparable cell; C8 has zero wrong_goal_region)",
            "wrong_goal_region_share": "wrong_goal_region / episodes per scenario",
        },
        "by_scenario": {},
    }
    for scenario in SCENARIOS:
        bucket = by_scenario.get(
            scenario,
            {
                "episodes": 0,
                "grasp_achieved_c6_rule": 0,
                "grasp_achieved_transport_incomplete_only": 0,
                "wrong_goal_region": 0,
                "no_grasp": 0,
                "outcomes": {},
            },
        )
        eps = int(bucket["episodes"])
        grasp_c6 = int(bucket["grasp_achieved_c6_rule"])
        grasp_ti = int(bucket["grasp_achieved_transport_incomplete_only"])
        wgr = int(bucket["wrong_goal_region"])
        payload["by_scenario"][scenario] = {
            "episodes": eps,
            "grasp_achieved_c6_rule": grasp_c6,
            "grasp_rate_c6_rule_pct": round(100 * grasp_c6 / max(eps, 1), 1),
            "grasp_achieved_transport_incomplete_only": grasp_ti,
            "grasp_rate_transport_incomplete_only_pct": round(100 * grasp_ti / max(eps, 1), 1),
            "wrong_goal_region_count": wgr,
            "wrong_goal_region_share_pct": round(100 * wgr / max(eps, 1), 1),
            "no_grasp": int(bucket["no_grasp"]),
            "outcomes": dict(bucket["outcomes"]),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = scan_terminals(args.attempts_root.resolve())
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
