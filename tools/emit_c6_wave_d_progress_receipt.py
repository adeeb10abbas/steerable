#!/usr/bin/env python3
"""Emit C6 wave-D progress receipt with dual grasp conventions from PVC scan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.report_c6_confirmatory_grasp_by_scenario import SCENARIOS, scan_terminals


def scan_pvc_via_exec(
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    attempts_root: str,
) -> dict:
    script = f"""
import json, glob, os
from collections import defaultdict
root = {json.dumps(attempts_root)}
by = defaultdict(lambda: {{'episodes':0,'grasp_achieved_c6_rule':0,'grasp_achieved_transport_incomplete_only':0,'wrong_goal_region':0,'no_grasp':0,'outcomes':defaultdict(int)}})
for complete in glob.glob(root + '/**/COMPLETE.json', recursive=True):
    ep = os.path.join(os.path.dirname(complete), 'episode.json')
    if not os.path.isfile(ep):
        continue
    row = json.load(open(ep))
    scenario = str(row.get('scenario') or '?')
    label = str(row.get('failure_label') or '?')
    b = by[scenario]
    b['episodes'] += 1
    b['outcomes'][label] += 1
    if label == 'no_grasp':
        b['no_grasp'] += 1
    else:
        b['grasp_achieved_c6_rule'] += 1
    if label == 'transport_incomplete':
        b['grasp_achieved_transport_incomplete_only'] += 1
    if label == 'wrong_goal_region':
        b['wrong_goal_region'] += 1
out = {{'terminals': sum(v['episodes'] for v in by.values()), 'by_scenario': {{}}}}
for sc, b in by.items():
    eps = b['episodes']
    out['by_scenario'][sc] = {{
        'episodes': eps,
        'grasp_achieved_c6_rule': b['grasp_achieved_c6_rule'],
        'grasp_rate_c6_rule_pct': round(100*b['grasp_achieved_c6_rule']/max(eps,1),1),
        'grasp_achieved_transport_incomplete_only': b['grasp_achieved_transport_incomplete_only'],
        'grasp_rate_transport_incomplete_only_pct': round(100*b['grasp_achieved_transport_incomplete_only']/max(eps,1),1),
        'wrong_goal_region_count': b['wrong_goal_region'],
        'wrong_goal_region_share_pct': round(100*b['wrong_goal_region']/max(eps,1),1),
        'no_grasp': b['no_grasp'],
        'outcomes': dict(b['outcomes']),
    }}
print(json.dumps(out))
"""
    proc = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            namespace,
            "--context",
            kube_context,
            publisher_pod,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supersedes", type=Path, default=None)
    parser.add_argument("--wave-label", default="D")
    args = parser.parse_args(argv)

    if args.attempts_root is not None:
        grasp = scan_terminals(args.attempts_root.resolve())
    else:
        grasp = scan_pvc_via_exec(
            kube_context="prod-dcwi-warrenq1-vmkub007",
            namespace="211247-prod",
            publisher_pod="211247-ali-b200-1gpu",
            attempts_root="/data/users/ali/vla_wam/raw/v4/c6-containment-main",
        )

    terminals = int(grasp["terminals"])
    milestones = {"32": "passed", "128": "passed" if terminals >= 128 else "pending", "384": "passed" if terminals >= 384 else "pending"}

    by_scenario = {}
    ordering_rates = []
    for scenario in SCENARIOS:
        bucket = grasp.get("by_scenario", {}).get(scenario, {})
        by_scenario[scenario] = bucket
        ordering_rates.append((scenario, float(bucket.get("grasp_rate_c6_rule_pct") or 0)))

    ordering_rates.sort(key=lambda x: -x[1])
    ti_ordering = []
    for scenario in SCENARIOS:
        bucket = grasp.get("by_scenario", {}).get(scenario, {})
        ti_ordering.append((scenario, float(bucket.get("grasp_rate_transport_incomplete_only_pct") or 0)))
    ti_ordering.sort(key=lambda x: -x[1])
    ordering_note = (
        f"At n={terminals}, C6 rule ordering: {' > '.join(f'{s} ({r:.1f}%)' for s,r in ordering_rates)}; "
        f"TI-only ordering: {' > '.join(f'{s} ({r:.1f}%)' for s,r in ti_ordering)}. "
    )
    if terminals >= 384:
        ordering_note += (
            "FINAL at 384: destination_static ranks first under both conventions. "
            "Second and third ranks swapped at least three times as n grew (268, 330, 337); "
            "the final rank order at n=384 is not evidence of a stable rank between move_stop and original_sham. "
            "Only the claim that destination_static ranks first is supportable."
        )
    else:
        ordering_note += (
            "Second/third rank order has swapped across milestones — do not over-read rank stability. "
            "Also report transport_incomplete_only for C8-comparable cells."
        )

    receipt = {
        "schema_version": "v4-c6-wave-progress-receipt-v1",
        "compiled_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wave_id": "c6confirm20260908f",
        "wave_label": args.wave_label,
        "confirmatory_terminals": terminals,
        "milestones": milestones,
        "grasp_rate_by_scenario": by_scenario,
        "grasp_definition_notes": {
            "c6_rule": "failure_label != no_grasp (shared scorer with C8)",
            "transport_incomplete_only": "failure_label == transport_incomplete (C8-comparable label set)",
            "c8_wrong_goal_region_asymmetry": "C8 observed zero wrong_goal_region in 407 episodes; C6 has nonzero wgr — use ti_only for cross-platform magnitude comparison",
        },
        "ordering_note": ordering_note,
        "supersedes": (
            str(args.supersedes.resolve().relative_to(ROOT.resolve()))
            if args.supersedes is not None
            else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
