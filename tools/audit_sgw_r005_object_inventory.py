"""Replay complete native-config recovery without inferring missing geometry."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path, PurePosixPath

from tools.prove_sgw_r005_reset_bounds import (
    INFRA, INVENTORY, LEDGER, REVISION, read_bound, require, sha, source_blob,
)


EXPORT = INFRA / "historical-r005-inventory-20260923cb"
CONFIG_INVENTORY = {
    "bytes": 47142, "sha256": "232096d3819cf7d77a25f12bd0984759c71efefc365d4c6fc5f74fea57f1deb9",
}
RIGID_CLASS = "isaaclab.assets.rigid_object.rigid_object:RigidObject"
ACTORS = ("banana", "banana_right", "bowl", "rubiks_cube", "table")


def reconstruct_config(base: bytes, row: dict) -> dict:
    delta = row["config_byte_delta"]
    start, end = delta["base_start_inclusive"], delta["base_end_exclusive"]
    require(type(start) is int and type(end) is int and 0 <= start <= end <= len(base),
            "invalid config byte-delta bounds")
    replacement = base64.b64decode(delta["replacement_base64"], validate=True)
    raw = base[:start] + replacement + base[end:]
    binding = row["source_binding"]
    require(len(raw) == binding["bytes"] and sha(raw) == binding["sha256"],
            "reconstructed config byte/hash mismatch")
    value = json.loads(raw)
    require(isinstance(value, dict), "native config must be a mapping")
    return value


def compile_audit() -> dict:
    inventory = read_bound(EXPORT / "inventory.json", CONFIG_INVENTORY)
    previous = read_bound(INVENTORY, {
        "bytes": 137174, "sha256": "1bfdfba81f1b177a080ccc73ad99366a8c3bc924f3343e3a74b28b443e4539de",
    })
    ledger = read_bound(LEDGER, {
        "bytes": 3382, "sha256": "3efd158f2502d6495006e32cdb9c715e0d8f23286242045148185839051c349c",
    })
    require(inventory["failure_report"] == previous["failure_report"] == ledger["raw_bindings"]["failure_report"],
            "config inventory belongs to another historical failure")
    require(inventory["construction_source"] == previous["construction_source"]
            and inventory["construction_source"]["study_commit"] == REVISION
            and inventory["input_bindings"] == previous["input_bindings"],
            "config source/runtime bindings differ")
    checkout = ledger["source"]["study_checkout"]
    runtime_raw, runtime_binding = source_blob(inventory["input_bindings"]["runtime_contract"], checkout)
    runtime = json.loads(runtime_raw)
    sources = inventory["pinned_robolab_source"]
    require(sources["revision"] == runtime["robolab_commit"], "pinned RoboLab revision differs")
    serializer = runtime["components"]["raw_writer"]["contract"]["native_runtime"]
    require(all(sources["files"]["robolab/core/environments/runtime.py"][key] == serializer[key]
                for key in ("bytes", "sha256")), "native serializer binding differs")
    layout = runtime["components"]["layout"]["contract"]
    source_bindings = {
        key: source_blob(binding, checkout)[1]
        for key, binding in {
            "producer": inventory["construction_source"],
            "fixture": layout["fixture"], "left_task": layout["left_task"],
            "paired_scene_asset": inventory["input_bindings"]["paired_scene_asset"],
        }.items()
    }
    base_binding = inventory["base_config"]
    require(base_binding["export_path"] == "base_env_cfg.json", "unexpected base config path")
    base = (EXPORT / "base_env_cfg.json").read_bytes()
    require(len(base) == base_binding["bytes"] and sha(base) == base_binding["sha256"],
            "base config byte/hash mismatch")
    records, lifecycles = inventory["records"], previous["environment_lifecycle"]
    require(len(records) == len(lifecycles) == 20
            and [row["environment_ordinal"] for row in records] == list(range(1, 21)),
            "native config lifecycle population differs")
    require(len({row["source_binding"]["path"] for row in records}) == 20,
            "native config source paths are not unique")
    configs = []
    for row, lifecycle in zip(records, lifecycles, strict=True):
        require(all(row[key] == lifecycle[key]
                    for key in ("environment_ordinal", "label", "candidate_rank", "stage", "role")),
                "native config lifecycle identity differs")
        require(lifecycle["created"] is True and lifecycle["fresh_reset_completed_in_this_environment"] is True,
                "native lifecycle lacks successful construction/reset")
        require(row["source_binding"]["path"] == str(PurePosixPath(lifecycle["native_output_dir"]) / "env_cfg.json"),
                "native config path differs from its lifecycle")
        config = reconstruct_config(base, row)
        require(config["recorders"]["dataset_export_dir_path"] == lifecycle["native_output_dir"],
                "serialized config output directory differs")
        scene = config["scene"]
        rigid = {name: value for name, value in scene.items()
                 if isinstance(value, dict) and value.get("class_type") == RIGID_CLASS}
        require(tuple(sorted(rigid)) == ACTORS and config["contact_object_list"] == list(ACTORS),
                "registered rigid/contact object inventory differs")
        require(config["_task_name"] == "V3E004DroidLeftTask"
                and scene["scene"]["spawn"]["usd_path"] == inventory["input_bindings"]["paired_scene_asset"]["path"],
                "native task or scene asset differs")
        profile = inventory["semantic_profiles"][row["semantic_profile_sha256"]]
        require(sha(json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
                == row["semantic_profile_sha256"], "semantic profile digest differs")
        require(profile["configured_rigid_objects"] == rigid
                and profile["scene_entry_names"] == sorted(scene)
                and profile["terminations"] == config["terminations"],
                "semantic projection differs from complete config")
        activation = lifecycle["construction_horizon_activation"]
        require(activation["only_mutated_field"] == "env.cfg.episode_length_s"
                and activation["termination_config_byte_equal"] is True
                and activation["original_episode_length_s"] == config["episode_length_s"],
                "registered post-construction horizon transition differs")
        for phase in ("before", "after"):
            require(activation[f"termination_contract_{phase}"]["termination_config"] == config["terminations"],
                    "native scored-reference contract differs")
        require(config["terminations"]["success"]["params"]["reference_object"] == "bowl",
                "scored reference differs")
        del config["recorders"]["dataset_export_dir_path"]
        configs.append(config)
    require(all(config == configs[0] for config in configs), "configs differ beyond their output directory")
    return {
        "schema_version": "sgw-01-r005-native-object-inventory-audit-v1",
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "inventory_sha256": CONFIG_INVENTORY["sha256"],
        "failure_report": inventory["failure_report"],
        "source_commit": REVISION, "source_bindings": source_bindings,
        "runtime_contract": runtime_binding,
        "pinned_robolab_source": sources,
        "hash_verified_native_configs": len(configs),
        "unique_complete_configs": len({row["source_binding"]["sha256"] for row in records}),
        "source_config_bytes_verified": sum(row["source_binding"]["bytes"] for row in records),
        "complete_configs_differ_only_in": "recorders.dataset_export_dir_path",
        "native_scene_entry_count": len(configs[0]["scene"]),
        "registered_rigid_objects": list(ACTORS),
        "native_success_reference": "bowl",
        "native_config_has_registered_plate": "plate" in configs[0]["scene"],
        "config_phase": "After environment construction; before the recorded construction-only horizon extension.",
        "serialization_contract": (
            "The hash-bound native runtime serializes env_cfg.to_dict() after successful gym.make/"
            "RobolabEnv construction and check_scene_valid. All twenty configs match their lifecycle "
            "and output directory. Every lifecycle retains the same before/after termination config."
        ),
        "inventory_completeness_boundary": (
            "These are complete serialized configs, not a composed USD-stage inventory. The pinned "
            "importer uses an explicit objects_of_interest allowlist and also spawns the complete "
            "USD scene. Registered rigid-object absence alone does not prove absence throughout "
            "transitive payloads. No independent historical imported-importer byte attestation "
            "or complete historical transitive asset closure is added."
        ),
        "materialization_bound_boundary": (
            "At the exact producer revision, _finalize_unchanged_gates returns the state even when "
            "state['passed'] is false; normal materialization completion does not imply those gates "
            "passed. Completed-reset bounds cannot be extended to missing later constructed states."
        ),
        "dist_no_plate_exclusion_status": "unresolved_not_excluded",
        "observed_pose_rows_added": 0,
        "missing_materialization_states_recovered": False,
        "historical_population_coverage_complete": False,
        "release_permitted": False, "model_requests": 0, "behavioral_episodes": 0,
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
