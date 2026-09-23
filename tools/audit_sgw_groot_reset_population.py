"""Audit named GR00T reset recovery; do not infer broader population coverage."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import subprocess

from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import _validated_position
from tools.audit_sgw_historical_lineage import require
from tools.prove_sgw_r005_reset_bounds import function, read_bound, sha


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "artifacts/workshops/spatial_grounding_v1/infrastructure/historical-groot-resets-20260923cc"
RECOVERY_BINDING = {
    "bytes": 167874, "sha256": "047c78b990f55e71b9d80e3b83b10744a0ece6a67172e6f05d9c379b3bd85699",
}
REVISION = "0b24153a2ba1fff9173fc47937ad291d24959e72"
SOURCE_PATH = "experiments/v3/groot_droid/robolab_bridge.py"


def compile_audit() -> dict:
    recovered = read_bound(EXPORT / "resets.json", RECOVERY_BINDING)
    manifest = read_bound(ROOT / recovered["final_manifest"]["path"], recovered["final_manifest"])
    queue_path = ROOT / "artifacts/vla_wam_shared_v3/phase_a_cells.jsonl"
    queue_raw = queue_path.read_bytes()
    require(sha(queue_raw) == recovered["queue"]["sha256"]
            and len(queue_raw) == recovered["queue"]["bytes"], "frozen queue binding differs")
    queue = [json.loads(line) for line in queue_raw.splitlines()]
    cells = {row["cell_id"]: row for row in queue
             if row["model_id"] == "groot_n17_droid_vla" and row["status"] == "authorized_new"}
    records = recovered["records"]
    require(len(records) == len(cells) == manifest["episode_count"] == 54
            and len({row["registered_cell_id"] for row in records}) == 54
            and {row["registered_cell_id"] for row in records} == set(cells),
            "final-cell reset population differs")
    pairs = {row["seed"]: row for row in manifest["pairs"]}
    require(set(pairs) == set(range(8303, 8330)) and manifest["pair_count"] == 27,
            "final pair population differs")
    require(recovered["pair_manifests"] == [pair["pair_evidence_manifest"] for pair in manifest["pairs"]],
            "pair manifest bindings differ")
    points, distances = [], []
    for row in records:
        cell = cells[row["registered_cell_id"]]
        require(row["seed"] == cell["environment_seed"] and row["relation"] == cell["relation"],
                "reset cell identity differs")
        final = pairs[row["seed"]]["cells"][row["relation"]]
        for key, original in (
            ("state_trace", "state_score_trace"), ("state_stream", "state_stream_partial"),
            ("warmup_trace", "warmup_reset"),
        ):
            require(row[key] == final[original], "reset source binding differs")
        require(row["sample_count"] == row["actions_executed"] + 1
                and 0 < row["actions_executed"] <= 450, "captured action/state accounting differs")
        require(row["capture_contract"]["coordinates"]
                == "robot-base frame from Isaac root pose wxyz inverse rotation",
                "declared raw coordinate contract differs")
        for phase in ("episode_initial", "warmup_reset"):
            sample = row[phase]
            require(type(sample["action_step"]) is int and sample["action_step"] == 0,
                    "retained sample is not a pre-action reset")
            cube, bowl = (_validated_position(sample[key]) for key in ("object_xyz", "reference_xyz"))
            require(cube is not None and bowl is not None, "invalid retained root vectors")
            points.append((cube, bowl))
            distances.append(math.dist(cube, bowl))
        profile_sha = row["environment_config_projection_sha256"]
        profile = recovered["config_projections"][profile_sha]
        require(sha(json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
                == profile_sha, "native config projection digest differs")
    for pair in pairs.values():
        matched = [row for row in records if row["seed"] == pair["seed"]]
        for phase in ("episode_initial", "warmup_reset"):
            for key in ("object_xyz", "reference_xyz"):
                require(matched[0][phase][key] == matched[1][phase][key], "paired reset vectors differ")
    require(recovered["recorded_source_study_commit"] == manifest["source_study_commit"] == REVISION,
            "recorded producer revision differs")
    original = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", f"{REVISION}:{SOURCE_PATH}"])
    patch = json.loads((EXPORT / "runtime/manifest.json").read_bytes())
    require(sha(original) == patch["frozen_source"]["sha256"], "frozen producer source differs")
    runtime = (EXPORT / "runtime/robolab_bridge_runtime.py").read_bytes()
    require(sha(runtime) == patch["runtime_source"]["sha256"], "retained runtime patch digest differs")
    for binding in recovered["retained_runtime_sources"]:
        path = (EXPORT / binding["export_path"]).resolve()
        require(path.is_relative_to(EXPORT.resolve()), "retained runtime path escapes export")
        raw = path.read_bytes()
        require(len(raw) == binding["bytes"] and sha(raw) == binding["sha256"],
                "retained runtime source byte/hash mismatch")
        require(binding["historical_hash_anchor_in_final_manifest"] is False,
                "recovery unexpectedly claims historical runtime attestation")
    unchanged = {}
    for name in ("_sample", "_quat_inverse_rotate_wxyz"):
        old = function(ast.parse(original), name)
        new = function(ast.parse(runtime), name)
        require(ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False),
                "runtime root getter or metric transform differs from recorded source")
        unchanged[name] = {
            "frozen_lines": [old.lineno, old.end_lineno],
            "retained_runtime_lines": [new.lineno, new.end_lineno],
        }
    return {
        "schema_version": "sgw-01-groot-final-cell-reset-population-audit-v1",
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "recovery_sha256": RECOVERY_BINDING["sha256"],
        "final_manifest": recovered["final_manifest"],
        "frozen_queue_sha256": sha(queue_raw), "source_commit": REVISION,
        "frozen_bridge_sha256": sha(original), "retained_runtime_bridge_sha256": sha(runtime),
        "unchanged_root_capture_helpers": unchanged,
        "final_manifest_episode_count": len(records),
        "retained_behavioral_initial_resets": len(records),
        "retained_preinference_warmup_resets": len(records),
        "retained_snapshot_count": len(points),
        "distinct_declared_cube_bowl_root_pairs": len(set(points)),
        "declared_frame": "robot-base frame from Isaac root pose wxyz inverse rotation",
        "declared_frame_root_separation_range_m": [min(distances), max(distances)],
        "complete_declared_final_manifest_cell_selection": True,
        "historical_runtime_patch_independently_hash_anchored": False,
        "historical_population_coverage_complete": False,
        "root_frame_cross_comparison_qualified": False,
        "proved_historical_nonmatches_added": 0,
        "release_permitted": False, "new_model_requests": 0, "new_behavioral_episodes": 0,
        "claim_boundary": (
            "All 54 final-manifest cells and 54 named warmups are recovered and reconciled with "
            "their bound full streams. Their identical numerical root pairs are not 108 independent "
            "layouts. Other attempts, preflights and constructor transients remain outside this "
            "selection. The retained runtime patch's getter/rotation ASTs match the recorded Git "
            "source; its contemporaneous patch manifest is freshly retained but no independent "
            "historical hash anchor has been established. No source-qualified cross-frame exclusion "
            "or global historical population coverage follows from this recovery."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_audit()
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
