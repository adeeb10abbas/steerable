#!/usr/bin/env python3
"""Fail-closed validation for the unreleased prospective Tier-B registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/registry.json"
EXPECTED = {
    "pi05_stochastic_eligibility_v3d001.json": (2585, "899a52c79355919210d56fa8f31d944f8a373e1e184650ee8974d62acfd6c788"),
    "nano_base_rotation_v3b006.json": (3398, "4e157931a1f2cbaa6f51b5e93d49caa0420787621452e791e7a4569f4be14fd6"),
    "fastwam_robotwin_mirror_v3b007.json": (3585, "84d14a5c6a02c5f6655384d2ed1ef6e3cdaab05341136d81a3b0e727268ecc8e"),
    "nano_start_side_v3b008.json": (3076, "8cd7c3bda7db0c3b9097e72c54d74fe0b81fd8a3d1909b6ec4aea00748c854c2"),
    "nano_role_swap_v3b009.json": (3102, "b6128c0ace0982980f1e650186324644cd89b896c2c1bb01c807adba584c1108"),
    "checkpoint_provenance.schema.json": (4374, "dc058dd11e09e70d5437d79db385910e6d5151642fc46829eb036b78e7fc8ddc"),
}
LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    registry = load(REGISTRY)
    require(registry.get("behavioral_cells_authorized_by_this_registry") == 0, "registry released behavior")
    require(registry.get("model_requests_authorized_by_this_registry") == 0, "registry released model requests")
    rows = registry.get("registrations")
    require(isinstance(rows, list) and len(rows) == 6, "registry must bind six prospective records")
    bound = {Path(row["path"]).name: row for row in rows}
    require(set(bound) == set(EXPECTED), "registry file set changed")
    base = REGISTRY.parent
    for name, (size, sha256) in EXPECTED.items():
        path = base / name
        require(path.is_file(), f"missing registration: {name}")
        require(path.stat().st_size == size and digest(path) == sha256, f"registration bytes changed: {name}")
        require(bound[name].get("bytes") == size and bound[name].get("sha256") == sha256, f"registry binding changed: {name}")

    pi05 = load(base / "pi05_stochastic_eligibility_v3d001.json")
    require(pi05["eligibility_probe"]["behavioral_episode_count"] == 0, "eligibility probe became behavioral")
    require(pi05["eligibility_probe"]["shared_candidate_sampling_seed_indices"] == list(range(8)), "probe seeds changed")
    require(pi05["eligibility_probe"]["exact_prompts"] == {"left": LEFT, "right": RIGHT}, "pi0.5 prompts changed")

    expected_counts = {
        "nano_base_rotation_v3b006.json": (27, 162),
        "fastwam_robotwin_mirror_v3b007.json": (27, 108),
        "nano_start_side_v3b008.json": (27, 162),
        "nano_role_swap_v3b009.json": (27, 108),
    }
    for name, (seed_count, ceiling) in expected_counts.items():
        value = load(base / name)
        require(value.get("status", "").find("not_released") >= 0, f"{name} became released")
        require(len(value["design"]["matched_seeds"]) == seed_count, f"{name} seed count changed")
        require(len(set(value["design"]["matched_seeds"])) == seed_count, f"{name} duplicate seed")
        require(value["design"]["behavioral_episode_ceiling_after_release"] == ceiling, f"{name} ceiling changed")
        require(value["model_blind_gate"]["model_requests"] == 0, f"{name} gate used model")
        require(value["release_boundary"]["behavioral_release"] is False, f"{name} released behavior")

    fastwam = load(base / "fastwam_robotwin_mirror_v3b007.json")
    require(fastwam["prospective_checkpoint_selection"]["known_direct_gate"] == {
        "left": "1/7",
        "right": "1/7",
        "scope": "preserved V2 r0 coverage layer; descriptive checkpoint-selection evidence only",
    }, "FastWAM prospective selection disclosure changed")
    require(fastwam["analysis_plan"]["arena_boundary"] == "Never pool these RoboTwin outcomes with DROID.", "arena boundary changed")

    provenance = load(base / "checkpoint_provenance.schema.json")
    require(set(provenance["required"]) >= {
        "checkpoint_identity", "training_episode_multiset", "preprocessing", "caption_exposure", "evidence"
    }, "provenance schema lost a required disclosure")
    print("Prospective Tier-B registry validation passed: 6 records, 0 released cells")


if __name__ == "__main__":
    main()
