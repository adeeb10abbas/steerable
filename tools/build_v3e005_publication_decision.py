#!/usr/bin/env python3
"""Write the V3-E005 decision memo and manuscript-facing decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"


def pct(value: float) -> str:
    return f"{100.0 * value:+.1f} pp"


def cm(value: float) -> str:
    return f"{100.0 * value:+.1f} cm"


def interval(item: Mapping[str, Any], scale: float = 1.0) -> str:
    return f"[{scale * item['low']:+.3f}, {scale * item['high']:+.3f}]"


def build_text(report: Mapping[str, Any]) -> tuple[str, str]:
    complete = bool(report["coverage"]["complete"])
    gate = report["h4_gate"]
    boundary = (
        "All estimands are RoboTwin-only. The 27 seed replicates are nested within seven scenes; "
        "scene-clustered intervals, rather than seed-independent intervals, are used. No DROID "
        "success predicate, episode, or pooled estimate enters V3-E005. The symmetric object layout "
        "does not make the robot, embodiment, joint configuration, or cameras symmetric."
    )
    if not complete:
        memo = f"""# V3-E005 decision memo

Status: **partial_progress_no_publication_claims**.

Valid behavioral cells: **{report['valid_behavioral_episodes']}/108**. H4 is not evaluable until all 54 matched pairs are hash-bound. H1–H3 are withheld.

{boundary}
"""
        publication = f"""# V3-E005 publication decision

V3-E005 is incomplete. No publication claim is authorized, and H1–H3 remain withheld until the preregistered H4 positive control is evaluated on the complete cohort.

{boundary}
"""
        return memo, publication

    level_lines = []
    for level in ("0.00", "1.00"):
        item = gate["levels"][level]
        ci = item["scene_clustered_bootstrap_mean95"]
        level_lines.append(
            f"- s={level}: mean endpoint redirection {item['mean_m']:+.3f} m, "
            f"scene-clustered 95% CI {interval(ci)}, pass={str(item['pass']).lower()}."
        )
    if not gate["hard_gate_passed"]:
        result = "\n".join(level_lines)
        memo = f"""# V3-E005 decision memo

Status: **complete_h4_fail_h1_h3_withheld**.

## H4 — positive-control hard gate

{result}

H4 outcome: **fail**. The preregistered rule therefore withholds H1, H2, and H3. Their estimates are not printed, graphed, or interpreted. No reflected stage and no replacement checkpoint is authorized.

## Paper placement

Place this result in the supplement beside the FastWAM E004 positive-control failure. It is evidence that this RoboTwin checkpoint/layout did not support the causal language-steering interpretation required for the geometry hypotheses; it is not evidence that geometry has no effect.

{boundary}
"""
        publication = f"""# V3-E005 publication decision

**Recommendation: supplement; H1–H3 omitted.**

LingBot-VA failed the preregistered endpoint-redirection positive control at one or both layouts. Under the prospective decision rule, the geometry, equivalence, and failure-signature hypotheses are not interpreted. This checkpoint is not replaced post hoc.

Safe manuscript text:

> In the separate RoboTwin replication, the preregistered endpoint-redirection positive control failed at one or both object layouts. We therefore withhold the downstream geometry hypotheses and report the control failure in the supplement; no DROID/RoboTwin pooling is performed.

{boundary}
"""
        return memo, publication

    h1 = report["hypotheses"]["H1"]
    h2 = report["hypotheses"]["H2"]
    h3 = report["hypotheses"]["H3"]
    binary = h1["interaction_s1_minus_s0"]["binary"]
    depth = h1["interaction_s1_minus_s0"]["requested_depth_m"]
    bci = binary["scene_clustered_bootstrap_mean95"]
    dci = depth["scene_clustered_bootstrap_mean95"]
    h4_text = "\n".join(level_lines)
    failure_delta = h3["s1_minus_s0_failure_share"]
    memo = f"""# V3-E005 decision memo

Status: **complete_h4_pass_h1_h3_reported**.

## H4 — positive-control hard gate

{h4_text}

H4 outcome: **pass**. H1–H3 are reported only after this gate.

## H1 — geometry interaction

- Binary success-gap interaction (s1−s0): {pct(binary['mean'])}; scene-clustered 95% CI {interval(bci)}; exact within-seed layout-label permutation p={binary['exact_within_seed_layout_label_permutation']['exact_two_sided_p']:.6g}.
- Requested-depth interaction (s1−s0): {cm(depth['mean'])}; scene-clustered 95% CI {interval(dci, 100.0)} cm; exact within-seed layout-label permutation p={depth['exact_within_seed_layout_label_permutation']['exact_two_sided_p']:.6g}.

These are RoboTwin estimands under its frozen native outcome coordinate and predicate. They are not combined with DROID.

## H2 — equivalence boundary

No equivalence claim is authorized. Binary equivalence has a registered zero margin and is undefined; requested-depth equivalence was preregistered as underpowered. The s=1 estimates are retained descriptively: binary gap {h2['binary']['estimate']:+.3f}, depth gap {h2['requested_depth_m']['estimate']:+.3f} m.

## H3 — failure signature

Failure-only s1-minus-s0 shares: wrong-side {failure_delta['wrong_side'] if failure_delta['wrong_side'] is not None else 'NR'}, pick {failure_delta['pick_failed'] if failure_delta['pick_failed'] is not None else 'NR'}, transport {failure_delta['transport_failed'] if failure_delta['transport_failed'] is not None else 'NR'}. A no-failure cell is NR and is never converted to zero.

## Claim boundary

{boundary}
"""
    publication = f"""# V3-E005 publication decision

**Recommendation: report as an arena-separated RoboTwin replication.**

LingBot-VA passed the preregistered endpoint-redirection positive control at both layouts. The seed-matched binary-gap interaction was {pct(binary['mean'])} and the requested-depth interaction was {cm(depth['mean'])}; both use 20,000-resample scene-clustered intervals over seven scenes. No equivalence claim is permitted by the registered power boundary.

Safe manuscript text:

> In a separately registered RoboTwin replication, LingBot-VA passed the endpoint-redirection positive control at both object layouts. We then evaluated the scene-clustered layout interaction under RoboTwin's native success predicate and coordinate system. These estimates provide cross-arena replication evidence without pooling RoboTwin with DROID, and the preregistered design does not support an equivalence claim.

{boundary}
"""
    return memo, publication


def build(results_path: Path, memo_path: Path, decision_path: Path) -> dict[str, Any]:
    report = json.loads(
        Path(results_path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if report.get("amendment_id") != "V3-E005" or report.get("arena") != "robotwin":
        raise ValueError("publication builder accepts only V3-E005 RoboTwin results")
    memo, decision = build_text(report)
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text(memo, encoding="utf-8")
    decision_path.write_text(decision, encoding="utf-8")
    return {"status": report["status"], "h4_outcome": report["h4_gate"]["outcome"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=BASE / "results/results.json")
    parser.add_argument("--memo", type=Path, default=BASE / "DECISION_MEMO.md")
    parser.add_argument("--publication-decision", type=Path, default=BASE / "V3E005_PUBLICATION_DECISION.md")
    args = parser.parse_args()
    print(json.dumps(build(args.results, args.memo, args.publication_decision), indent=2))


if __name__ == "__main__":
    main()
