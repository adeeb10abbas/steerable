#!/usr/bin/env python3
"""Build the immutable, pre-inference registration for V3-E001."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001"
SEEDS = list(range(9400, 9427))
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
MODELS = {
    "pi05_current_stack_droid": {"action_shape": [15, 8], "interface": "actions_only"},
    "cosmos3_nano_policy_droid": {"action_shape": [32, 8], "interface": "actions_plus_decoded_future"},
    "dreamzero_droid_action_cfg": {"action_shape": [1, 8], "interface": "actions_plus_native_future"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bind(path_text: str) -> dict[str, object]:
    path = ROOT / path_text
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path_text, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    parent_paths = [
        "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_manifest.json",
        "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl",
        "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/post_result_nano_mirror_v3b001_amendment.json",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_manifest.json",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_cells.jsonl",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/post_result_pi05_mirror_v3b002_amendment.json",
        "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/dreamzero_mirror_v3b003_manifest.json",
        "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/dreamzero_mirror_v3b003_cells.jsonl",
        "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/post_result_dreamzero_mirror_v3b003_amendment.json",
    ]
    registration = {
        "schema_version": "vla-wam-shared-v3e001-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E001",
        "status": "registered_before_inference",
        "question": "At an identical observation, does changing only the directional prompt produce an action effect larger than same-prompt policy-sampling variation?",
        "models": MODELS,
        "layouts": ["control", "position_mirrored"],
        "prompts": PROMPTS,
        "environment_seed_range": [9400, 9426],
        "policy_sampling_seeds": SEEDS,
        "design": {
            "requests_per_model_layout": 27 * 2,
            "exact_repeat_requests_per_model_layout": 2,
            "total_model_requests": 336,
            "behavioral_episode_count": 0,
            "preflight_per_model_layout": ["left_seed9400", "left_seed9400_exact_repeat", "right_seed9400"],
            "same_nonlanguage_observation_required": True,
            "same_sampling_seed_for_matched_prompt_pair": True,
            "no_action_execution": True,
        },
        "measurements": {
            "exact_repeat": ["bit_identity", "rms"],
            "same_prompt_noise": ["median", "mean", "p05", "p95", "maximum", "common_prefix_action_rms"],
            "prompt_effect": ["raw_rms", "per_dimension_rms", "exceeds_noise_p95_fraction"],
            "semantic_fk": "only_if_verified_native_action_frame_mapping_is_available",
            "layout_interaction": ["mean", "median", "paired_bootstrap_95_ci", "exact_two_sided_sign_test", "all_seed_effects"],
        },
        "invalidity_policy": "Infrastructure-invalid requests are retained separately and excluded from behavioral denominators; there are no behavioral episodes in E001.",
        "parent_bindings": [bind(path) for path in parent_paths],
    }
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUT / "registration.json"), "sha256": sha256(OUT / "registration.json"), "status": registration["status"]}, indent=2))


if __name__ == "__main__":
    main()
