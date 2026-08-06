#!/usr/bin/env python3
"""Audit endpoint/margin coverage without touching raw behavioral evidence.

The historical result formats use several field names.  This audit normalizes
only algebraically equivalent endpoint quantities:

* DROID signed endpoint: raw robot-frame cube-minus-bowl y (robot LEFT > 0).
* RoboTwin signed endpoint: ``-native object-minus-target x`` so positive is
  robot LEFT, matching the shared v3 lateral convention.
* Requested-side margin: signed endpoint for LEFT, negated endpoint for RIGHT
  in DROID; the native RoboTwin scorer uses the opposite x sign convention.

No missing value is imputed from a binary success label.  A cohort fails the
audit unless every behavioral episode has either an explicit quantity or the
stored signed endpoint needed for the exact one-line transformation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "artifacts" / "vla_wam_shared_v1"
V2 = ROOT / "artifacts" / "vla_wam_shared_v2"
V3 = ROOT / "artifacts" / "vla_wam_shared_v3"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{label} is not a finite number: {value!r}")
    return float(value)


def relation_sign(relation: str) -> float:
    if relation == "left":
        return 1.0
    if relation == "right":
        return -1.0
    raise ValueError(f"invalid relation: {relation!r}")


def close(a: float, b: float, *, atol: float = 1e-8) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


def cohort(
    *,
    cohort_id: str,
    arena: str,
    source_paths: Iterable[Path],
    rows: list[dict[str, Any]],
    expected_count: int,
    extraction: str,
    offset_explicit: bool,
    margin_explicit: bool,
    margin_sign_fn: Callable[[str], float] | None = None,
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise ValueError(
            f"{cohort_id}: expected {expected_count} episodes, observed {len(rows)}"
        )
    for index, row in enumerate(rows):
        offset = finite(row.get("signed_endpoint_m"), f"{cohort_id}[{index}].offset")
        margin = finite(row.get("requested_side_margin_m"), f"{cohort_id}[{index}].margin")
        sign_fn = margin_sign_fn or relation_sign
        expected_margin = sign_fn(row["relation"]) * offset
        if not close(margin, expected_margin):
            raise ValueError(
                f"{cohort_id}[{index}] margin/offset inconsistency: "
                f"{margin} != {expected_margin}"
            )
    sources = []
    for path in source_paths:
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "cohort_id": cohort_id,
        "arena": arena,
        "behavioral_episode_count": len(rows),
        "signed_endpoint_coverage": f"{len(rows)}/{len(rows)}",
        "requested_side_margin_coverage": f"{len(rows)}/{len(rows)}",
        "signed_endpoint_storage": "explicit" if offset_explicit else "exactly_derivable",
        "requested_side_margin_storage": "explicit" if margin_explicit else "exactly_derivable",
        "extraction": extraction,
        "rerun_required_for_these_two_measurements": False,
        "sources": sources,
    }


def json_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = load_json(path).get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"missing episodes array: {path}")
    return episodes


def droid_from_display(
    episodes: list[dict[str, Any]],
    *,
    display_key: str,
    explicit_margin_key: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        relation = str(episode["requested_relation"])
        offset = -finite(episode[display_key], display_key)
        margin = (
            finite(episode[explicit_margin_key], explicit_margin_key)
            if explicit_margin_key is not None
            else relation_sign(relation) * offset
        )
        rows.append(
            {
                "relation": relation,
                "signed_endpoint_m": offset,
                "requested_side_margin_m": margin,
            }
        )
    return rows


def robotwin_flat(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        relation = str(episode["requested_relation"])
        native_x = finite(episode["final_dx_m"], "final_dx_m")
        offset = -native_x
        margin = episode.get("command_alignment_margin_m")
        if margin is None:
            margin = relation_sign(relation) * offset
        rows.append(
            {
                "relation": relation,
                "signed_endpoint_m": offset,
                "requested_side_margin_m": finite(margin, "command_alignment_margin_m"),
            }
        )
    return rows


def exact_two_sided_sign_test(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    smaller = min(positive, negative)
    probability = 2.0 * sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, probability)


def build_report(recorded_at_utc: str) -> dict[str, Any]:
    cohorts: list[dict[str, Any]] = []

    # V1 DROID: both quantities are explicit in the frozen episode CSV.
    v1_path = V1 / "final_evidence" / "episodes.csv"
    with v1_path.open(newline="") as handle:
        v1_rows_raw = list(csv.DictReader(handle))
    v1_rows = [
        {
            "relation": row["direction"],
            "signed_endpoint_m": finite(
                float(row["final_cube_minus_bowl_y_m"]), "final_cube_minus_bowl_y_m"
            ),
            "requested_side_margin_m": finite(
                float(row["requested_signed_final_offset_m"]),
                "requested_signed_final_offset_m",
            ),
        }
        for row in v1_rows_raw
    ]
    cohorts.append(
        cohort(
            cohort_id="v1_droid_wording_grid",
            arena="droid_robolab",
            source_paths=[v1_path],
            rows=v1_rows,
            expected_count=160,
            extraction="explicit final_cube_minus_bowl_y_m and requested_signed_final_offset_m",
            offset_explicit=True,
            margin_explicit=True,
        )
    )

    v2_droid_specs = [
        (
            "v2_pi0_fast_direct_confirmation",
            V2 / "pilot" / "results" / "pi0_fast_direct_confirmation.json",
            "final_lateral_display_m",
            "requested_signed_final_offset_m",
            20,
        ),
        (
            "v2_pi05_current_stack_direct_gate",
            V2 / "pilot" / "expansion" / "pi05_current_stack_v2a010_direct_gate.json",
            "endpoint_lateral_display_m",
            None,
            6,
        ),
        (
            "v2_cosmos3_edge_policy_droid_direct_gate",
            V2 / "pilot" / "expansion" / "cosmos3_edge_droid_direct_gate.json",
            "final_lateral_display_m",
            "requested_signed_final_offset_m",
            6,
        ),
        (
            "v2_cosmos3_nano_policy_droid_direct_gate",
            V2 / "pilot" / "expansion" / "cosmos3_nano_policy_droid_direct_gate.json",
            "final_lateral_display_m",
            "requested_signed_final_offset_m",
            6,
        ),
        (
            "v2_dreamzero_direct_gate",
            V2 / "pilot" / "expansion" / "dreamzero_droid_direct_gate.json",
            "final_lateral_display_m",
            None,
            6,
        ),
        (
            "v2_cosmos3_nano_g1_guidance_ablation",
            V2 / "pilot" / "expansion" / "cosmos3_nano_v2a015_no_cfg_g1_result.json",
            "final_lateral_display_m",
            "requested_signed_final_margin_m",
            6,
        ),
        (
            "v2_dreamzero_s2_guidance_ablation",
            V2 / "pilot" / "expansion" / "dreamzero_v2a015_action_cfg_s2_result.json",
            "final_lateral_display_m",
            "requested_signed_final_margin_m",
            6,
        ),
    ]
    for cohort_id, path, display_key, margin_key, count in v2_droid_specs:
        rows = droid_from_display(
            json_episodes(path), display_key=display_key, explicit_margin_key=margin_key
        )
        cohorts.append(
            cohort(
                cohort_id=cohort_id,
                arena="droid_robolab",
                source_paths=[path],
                rows=rows,
                expected_count=count,
                extraction=(
                    f"signed endpoint = -{display_key}; requested margin "
                    + (f"explicit in {margin_key}" if margin_key else "derived from relation")
                ),
                offset_explicit=False,
                margin_explicit=margin_key is not None,
            )
        )

    groot_paths = [
        V2 / "pilot" / "expansion" / f"groot_n17_droid_seed{seed}_slice.json"
        for seed in (8300, 8301, 8302)
    ]
    groot_rows = []
    for path in groot_paths:
        for episode in json_episodes(path):
            relation = str(episode["condition"]).lower()
            groot_rows.append(
                {
                    "relation": relation,
                    "signed_endpoint_m": finite(
                        episode["endpoint_cube_minus_bowl_world_xyz"][1],
                        "endpoint_cube_minus_bowl_world_xyz[1]",
                    ),
                    "requested_side_margin_m": finite(
                        episode["requested_direction_endpoint_scalar"],
                        "requested_direction_endpoint_scalar",
                    ),
                }
            )
    cohorts.append(
        cohort(
            cohort_id="v2_groot_n17_direct_gate",
            arena="droid_robolab",
            source_paths=groot_paths,
            rows=groot_rows,
            expected_count=6,
            extraction="explicit cube-minus-bowl endpoint xyz and explicit requested-direction scalar",
            offset_explicit=True,
            margin_explicit=True,
        )
    )

    # V2 RoboTwin uses native object-minus-target x as its LEFT/RIGHT relation axis.
    v2_robotwin_specs = [
        ("v2_efficient_wam_rt_pilot", V2 / "pilot" / "results" / "efficient_wam_rt_direct_gate.json", 6),
        ("v2_fastwam_pilot", V2 / "pilot" / "results" / "fastwam_direct_gate.json", 6),
        ("v2_lingbot_va_pilot", V2 / "pilot" / "results" / "lingbot_va_direct_gate.json", 6),
        ("v2_lingbot_vla_4b_gate", V2 / "pilot" / "expansion" / "lingbot_vla_4b_direct_gate.json", 6),
        ("v2_efficient_wam_rt_pairs04_09", V2 / "pilot" / "directional_confirmation" / "efficient_wam_rt_pairs04_09_slice.json", 12),
        ("v2_fastwam_pairs03_09", V2 / "pilot" / "directional_confirmation" / "fastwam_pairs03_09_slice.json", 14),
        ("v2_lingbot_va_pairs03_09", V2 / "pilot" / "directional_confirmation" / "lingbot_va_pairs03_09_slice.json", 14),
    ]
    for cohort_id, path, count in v2_robotwin_specs:
        episodes = json_episodes(path)
        cohorts.append(
            cohort(
                cohort_id=cohort_id,
                arena="robotwin_place_a2b",
                source_paths=[path],
                rows=robotwin_flat(episodes),
                expected_count=count,
                extraction="explicit native object-minus-target x; requested margin explicit or exact relation-sign transform",
                offset_explicit=True,
                margin_explicit=all("command_alignment_margin_m" in row for row in episodes),
            )
        )

    light_path = V2 / "pilot" / "expansion" / "light_wam_robotwin_direct_gate.json"
    light_rows = []
    for episode in json_episodes(light_path):
        relation = str(episode["requested_relation"])
        native_x = finite(
            episode["final"]["object_minus_target_x"],
            "final.object_minus_target_x",
        )
        offset = -native_x
        light_rows.append(
            {
                "relation": relation,
                "signed_endpoint_m": offset,
                "requested_side_margin_m": relation_sign(relation) * offset,
            }
        )
    cohorts.append(
        cohort(
            cohort_id="v2_light_wam_gate",
            arena="robotwin_place_a2b",
            source_paths=[light_path],
            rows=light_rows,
            expected_count=6,
            extraction="standardized signed lateral = -native object-minus-target x; exact relation-sign margin",
            offset_explicit=False,
            margin_explicit=False,
        )
    )

    efficient_pair03_path = (
        V2 / "pilot" / "directional_confirmation" / "efficient_wam_rt_pair03_integration.json"
    )
    efficient_pair03 = load_json(efficient_pair03_path)
    pair03_rows = []
    for cell in efficient_pair03["cells"]:
        relation = str(cell["requested_relation"])
        native_x = finite(
            cell["final_object_minus_target_x_m"],
            "final_object_minus_target_x_m",
        )
        offset = -native_x
        pair03_rows.append(
            {
                "relation": relation,
                "signed_endpoint_m": offset,
                "requested_side_margin_m": relation_sign(relation) * offset,
            }
        )
    cohorts.append(
        cohort(
            cohort_id="v2_efficient_wam_rt_pair03",
            arena="robotwin_place_a2b",
            source_paths=[efficient_pair03_path],
            rows=pair03_rows,
            expected_count=2,
            extraction="standardized signed lateral = -native final_object_minus_target_x_m; exact relation-sign margin",
            offset_explicit=False,
            margin_explicit=False,
        )
    )

    # V3 Phase A summaries retain both normalized quantities per behavioral episode.
    v3_droid_ids = [
        "pi05_current_stack_droid",
        "groot_n17_droid",
        "cosmos3_edge_policy_droid",
        "cosmos3_nano_policy_droid",
        "dreamzero_droid_action_cfg",
    ]
    nano_rows: list[dict[str, Any]] = []
    for model_id in v3_droid_ids:
        path = V3 / "results" / f"{model_id}_phase_a_summary.json"
        data = load_json(path)
        rows = []
        for episode in data["cells"]:
            measurements = episode.get("measurements", episode)
            row = {
                "relation": episode["relation"],
                "signed_endpoint_m": measurements["signed_final_lateral_offset_m"],
                "requested_side_margin_m": measurements["final_requested_signed_margin_m"],
                "seed": episode["seed"],
            }
            rows.append(row)
        if model_id == "cosmos3_nano_policy_droid":
            nano_rows = rows
        cohorts.append(
            cohort(
                cohort_id=f"v3_{model_id}_phase_a",
                arena="droid_robolab",
                source_paths=[path],
                rows=rows,
                expected_count=54,
                extraction="explicit signed_final_lateral_offset_m and final_requested_signed_margin_m",
                offset_explicit=True,
                margin_explicit=True,
            )
        )

    v3_robotwin_ids = ["efficient_wam_rt_robotwin", "fastwam_robotwin", "lingbot_va_robotwin"]
    for model_id in v3_robotwin_ids:
        path = V3 / "results" / f"{model_id}_phase_a_summary.json"
        data = load_json(path)
        rows = [
            {
                "relation": episode["relation"],
                "signed_endpoint_m": episode["measurements"]["signed_final_lateral_offset_m"],
                "requested_side_margin_m": episode["measurements"]["final_requested_signed_margin_m"],
            }
            for episode in data["v3_primary_results"]["episodes"]
        ]
        cohorts.append(
            cohort(
                cohort_id=f"v3_{model_id}_phase_a",
                arena="robotwin_place_a2b",
                source_paths=[path],
                rows=rows,
                expected_count=126,
                extraction="explicit signed_final_lateral_offset_m and final_requested_signed_margin_m from the v3 robot-base scorer",
                offset_explicit=True,
                margin_explicit=True,
                margin_sign_fn=relation_sign,
            )
        )

    bridge_path = V3 / "results" / "pi0_fast_old_name_config_v3a002_summary.json"
    bridge = load_json(bridge_path)
    bridge_rows = []
    for pair in bridge["pairs"]:
        for relation in ("left", "right"):
            offset = finite(
                pair[f"{relation}_signed_final_lateral_offset_m"],
                f"{relation}_signed_final_lateral_offset_m",
            )
            bridge_rows.append(
                {
                    "relation": relation,
                    "signed_endpoint_m": offset,
                    "requested_side_margin_m": relation_sign(relation) * offset,
                }
            )
    cohorts.append(
        cohort(
            cohort_id="v3a002_pi0_fast_compatibility_bridge",
            arena="droid_robolab",
            source_paths=[bridge_path],
            rows=bridge_rows,
            expected_count=40,
            extraction="explicit signed endpoint in pair rows; exact relation-sign requested margin",
            offset_explicit=True,
            margin_explicit=False,
        )
    )

    nano_by_seed: dict[int, dict[str, float]] = {}
    for row in nano_rows:
        nano_by_seed.setdefault(int(row["seed"]), {})[row["relation"]] = finite(
            row["requested_side_margin_m"], "Nano requested-side margin"
        )
    nano_gaps = [values["right"] - values["left"] for values in nano_by_seed.values()]
    positives = sum(value > 0 for value in nano_gaps)
    negatives = sum(value < 0 for value in nano_gaps)

    total = sum(item["behavioral_episode_count"] for item in cohorts)
    droid_total = sum(
        item["behavioral_episode_count"] for item in cohorts if item["arena"] == "droid_robolab"
    )
    robotwin_total = total - droid_total
    return {
        "schema_version": "vla-wam-measurement-coverage-audit-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "recorded_at_utc": recorded_at_utc,
        "status": "complete_no_measurement_coverage_rerun_required",
        "field_aliases": {
            "requested_side_margin_m": "V3 final_requested_signed_margin_m; older exact aliases include requested_signed_final_offset_m, requested_signed_final_margin_m, requested_direction_endpoint_scalar, and command_alignment_margin_m.",
            "signed_final_lateral_offset_m": "Positive robot LEFT in each arena. DROID stores raw robot-frame movable-minus-reference y; older RoboTwin stores native object-minus-target x with the opposite sign, so standardized lateral is exactly -x. Arena coordinates are never pooled.",
        },
        "scope": {
            "unique_behavioral_episode_count": total,
            "droid_robolab_episode_count": droid_total,
            "robotwin_episode_count": robotwin_total,
            "nonbehavioral_interface_probes": "not applicable: Cosmos-Reason2 and Cosmos3 base probes have no robot episode endpoint",
            "withdrawn_or_unreleased_models": "not applicable: LaWAM has zero behavioral episodes",
        },
        "coverage": {
            "requested_side_margin_available": f"{total}/{total}",
            "signed_final_lateral_offset_available": f"{total}/{total}",
            "values_imputed_from_success_labels": 0,
            "measurement_coverage_rerun_required": False,
        },
        "nano_phase_a_margin_sensitivity_reproduction": {
            "matched_pair_count": len(nano_gaps),
            "left_mean_requested_side_margin_m": sum(v["left"] for v in nano_by_seed.values()) / len(nano_by_seed),
            "right_mean_requested_side_margin_m": sum(v["right"] for v in nano_by_seed.values()) / len(nano_by_seed),
            "right_minus_left_mean_margin_gap_m": sum(nano_gaps) / len(nano_gaps),
            "positive_zero_negative_pair_counts": [positives, len(nano_gaps) - positives - negatives, negatives],
            "exact_two_sided_sign_test_p_excluding_ties": exact_two_sided_sign_test(positives, negatives),
        },
        "groot_phase_a_reconciliation": {
            "matched_pairs": 27,
            "behavioral_episodes": 54,
            "status": "already_complete_do_not_rerun",
            "source": "artifacts/vla_wam_shared_v3/results/groot_n17_droid_phase_a_summary.json",
        },
        "analysis_rule_for_new_mirror_work": {
            "full_sample": "Use signed final lateral offset for every valid episode, including failures.",
            "margin": "Retain requested-side margin for every episode; report the all-episode paired margin estimand and identify any success-conditional subset explicitly rather than deleting failures.",
            "pairing": "LEFT and RIGHT share the exact reset and seed; mirror/control comparisons use a separately frozen contemporaneous allocation.",
        },
        "cohorts": cohorts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recorded-at-utc")
    args = parser.parse_args()
    recorded_at = args.recorded_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    report = build_report(recorded_at)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
