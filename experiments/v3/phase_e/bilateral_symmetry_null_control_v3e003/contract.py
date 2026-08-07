#!/usr/bin/env python3
"""Hash-bound registration contract for V3-E003.

This module deliberately does not alter any Phase-B source.  E003 reuses the
immutable B001 seed/order and prompt bindings while introducing a new,
single-bowl object layout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-E003"
PHASE = "E_publication_critical_controls"
MODEL_ID = "pi05_current_stack_droid"
ARENA = "droid_robolab"
SEEDS = tuple(range(9400, 9427))
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
SUCCESS_PREDICATE_ID = "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"
OPENPI_COMMIT = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
CHECKPOINT_MANIFEST_SHA256 = "f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca"
ACTION_CHUNK_STEPS = 15
ACTION_DIM = 8
ACTION_CAP = 450

B001_CELLS = Path("artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl")
B001_CELLS_SHA256 = "018b8b6ae76ac46f2f89eef83c4b16d7a4ff3d1ff15d91527b96fb56b5432c5a"
B001_MANIFEST = Path("artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_manifest.json")
B001_MANIFEST_SHA256 = "5c82268739feb41281435a51dcd848b575218cd9fbe5839d9ad130d1a7888830"
B001_AMENDMENT = Path("artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/post_result_nano_mirror_v3b001_amendment.json")
B001_AMENDMENT_SHA256 = "9d88c29733fa3b24a154977bc25d04d2d77df5be59e3213f0c3a6cfbe3edc6a0"

EQUIVALENCE_MARGIN = {
    "binary_gap_abs": 4 / 27,
    "requested_side_depth_contrast_abs_m": 0.05,
    "registered_before_inference": True,
}

# Candidate coordinates are robot-base coordinates.  The single bowl and
# target cube lie on y=0; the existing banana is represented as a mirrored
# pair at its B001 lateral magnitude.  The pair is generated in the new task
# scene from the same YCB banana payload; no bowl is duplicated.
SYMMETRIC_POSITIONS = {
    "bowl": [0.44258353114128113, 0.0, 0.07732785493135452],
    "rubiks_cube": [0.303364634513855, 0.0, 0.08113233000040054],
    "banana_left": [0.538878858089447, -0.07555567473173141, 0.0684281587600708],
    "banana_right": [0.538878858089447, 0.07555567473173141, 0.0684281587600708],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def prompt_sha256(relation: str) -> str:
    return hashlib.sha256(PROMPTS[relation].encode()).hexdigest()


def load_b001_rows(repo_root: Path) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    p = root / B001_CELLS
    if sha256_file(p) != B001_CELLS_SHA256:
        raise RuntimeError("immutable V3-B001 seed source changed")
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if len(rows) != 108:
        raise RuntimeError("V3-B001 source queue is not the 108-cell release")
    return rows


def expected_queue(repo_root: Path) -> list[dict[str, Any]]:
    rows = load_b001_rows(repo_root)
    # B001 randomization is retained only as a seed/order binding.  E003 has
    # two cells per matched seed, in the first/second direction positions of
    # the B001 block, and does not inherit either B001 movable-object layout.
    out = []
    for seed in SEEDS:
        source = [r for r in rows if r["environment_seed"] == seed]
        if len(source) != 4:
            raise RuntimeError(f"B001 seed {seed} is not a four-cell block")
        order = sorted(source, key=lambda r: int(r["execution_order_index_within_seed"]))
        for idx, relation in enumerate(RELATIONS, 1):
            source_row = order[idx - 1]
            out.append({
                "schema_version": "vla-wam-shared-v3e003-cell-v1",
                "study_id": STUDY_ID,
                "amendment_id": AMENDMENT_ID,
                "phase": PHASE,
                "arena": ARENA,
                "model_id": MODEL_ID,
                "cell_id": f"v3e003:pi05:seed{seed}:symmetric:{relation}",
                "matched_block_id": f"v3e003:pi05:seed{seed}",
                "layout": "symmetric_object_layout",
                "relation": relation,
                "prompt": PROMPTS[relation],
                "prompt_sha256": prompt_sha256(relation),
                "environment_seed": seed,
                "sampling_seed": seed,
                "execution_order_index_within_seed": idx,
                "source_v3b001_cell_id": source_row["cell_id"],
                "source_v3b001_queue_sha256": B001_CELLS_SHA256,
                "source_v3b001_randomization_key_sha256": source_row["randomization_key_sha256"],
                "fixture_id": "v3e003_pi05_symmetric_object_layout",
                "fixture_sha256": "REGISTERED_BY_SYMMETRY_GATE",
                "success_predicate_id": SUCCESS_PREDICATE_ID,
                "runtime_identity_requirement": {
                    "openpi_commit": OPENPI_COMMIT,
                    "robolab_commit": ROBOLAB_COMMIT,
                    "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
                    "open_loop_horizon": ACTION_CHUNK_STEPS,
                    "action_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
                    "action_cap": ACTION_CAP,
                },
                "required_raw_outputs": [
                    "viewport_video", "executed_action_trace", "raw_behavioral_episode_jsonl"
                ],
                "missing_measurement_policy": "NR is unavailable and never converted to zero",
                "technical_invalidity_policy": "Separate infrastructure ledger; no denominator entry",
                "valid_failure_policy": "Retain every valid behavioral failure in full-sample analyses",
            })
    return out
