"""Retain all hash-bound native configs for the twenty R005 attempt01 lifecycles."""

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tarfile


ROOT = Path("/data/users/ali/vla_wam/raw/v3e006_r005/state_repair/98f0234-a40r06-attempt01/raw")
FAILURE = {
    "path": str(ROOT / "state_construction_failure.json"),
    "bytes": 25614537,
    "sha256": "e858a4431bd0451a1e380d66c8c5d05c5486718f5b23388f874d24a7d271ae8e",
}
ROBOLAB_ROOT = "/data/users/ali/vla_wam/external/RoboLab-11142d4"
ROBOLAB_REVISION = "0aef241fb088ca21bb4ebd24448940ed56620d17"


def read_bound(binding, maximum):
    if binding["bytes"] > maximum:
        raise ValueError("registered input exceeds extraction limit")
    with Path(binding["path"]).open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"input binding differs: {binding['path']}")
    return raw


failure = json.loads(read_bound(FAILURE, 32 * 1024 * 1024))
lifecycles = failure["environment_lifecycle"]
if len(lifecycles) != 20 or [row["environment_ordinal"] for row in lifecycles] != list(range(1, 21)):
    raise ValueError("historical lifecycle population differs")
expected = {
    key: value for key, value in failure["available_raw_artifacts"].items()
    if key.startswith("native/") and key.endswith("/env_cfg.json")
}
paths = {(Path(row["native_output_dir"]) / "env_cfg.json").relative_to(ROOT).as_posix()
         for row in lifecycles}
if paths != set(expected) or len(paths) != 20:
    raise ValueError("native configs do not account for every recorded lifecycle")
files = {}
profiles = {}
records = []
base_raw = None
for lifecycle in lifecycles:
    path = Path(lifecycle["native_output_dir"]) / "env_cfg.json"
    binding = expected[path.relative_to(ROOT).as_posix()]
    if str(path) != binding["path"]:
        raise ValueError("native config path binding differs")
    raw = read_bound(binding, 128 * 1024)
    config = json.loads(raw)
    scene = config["scene"]
    if base_raw is None:
        base_raw = raw
        files["base_env_cfg.json"] = raw
    start = 0
    while start < min(len(raw), len(base_raw)) and raw[start] == base_raw[start]:
        start += 1
    suffix = 0
    while suffix < min(len(raw), len(base_raw)) - start and raw[-suffix - 1] == base_raw[-suffix - 1]:
        suffix += 1
    end = len(base_raw) - suffix
    replacement = raw[start:len(raw) - suffix]
    if base_raw[:start] + replacement + base_raw[end:] != raw:
        raise ValueError("config byte-delta does not reproduce its source")
    profile = {
        "task_name": config["_task_name"],
        "instruction": config["instruction"],
        "contact_object_list": config["contact_object_list"],
        "scene_entry_names": sorted(scene),
        "scene_classes": {
            name: value.get("class_type") for name, value in scene.items() if isinstance(value, dict)
        },
        "configured_rigid_objects": {
            name: value for name, value in scene.items()
            if isinstance(value, dict)
            and value.get("class_type") == "isaaclab.assets.rigid_object.rigid_object:RigidObject"
        },
        "scene_spawn_assets": {
            name: value for name, value in scene.items()
            if isinstance(value, dict) and isinstance(value.get("spawn"), dict)
        },
        "terminations": config["terminations"],
    }
    profile_sha = hashlib.sha256(json.dumps(
        profile, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    profiles[profile_sha] = profile
    records.append({
        "environment_ordinal": lifecycle["environment_ordinal"],
        "label": lifecycle["label"], "candidate_rank": lifecycle["candidate_rank"],
        "stage": lifecycle["stage"], "role": lifecycle["role"],
        "source_binding": binding,
        "config_byte_delta": {
            "base_start_inclusive": start, "base_end_exclusive": end,
            "replacement_base64": base64.b64encode(replacement).decode("ascii"),
        },
        "semantic_profile_sha256": profile_sha,
    })
source_summaries = {}
for relative, names in {
    "robolab/core/environments/runtime.py": ("create_env",),
    "robolab/core/scenes/utils.py": ("_scrape_scene_cached", "import_scene"),
}.items():
    raw = subprocess.check_output([
        "git", "-C", ROBOLAB_ROOT, "show", f"{ROBOLAB_REVISION}:{relative}",
    ])
    tree = ast.parse(raw)
    functions = {}
    for name in names:
        matches = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1:
            raise ValueError(f"pinned source function absent or ambiguous: {relative}:{name}")
        node = matches[0]
        functions[name] = {"first_line": node.lineno, "last_line": node.end_lineno}
    source_summaries[relative] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "functions": functions,
    }
if source_summaries["robolab/core/environments/runtime.py"]["sha256"] != (
    "ffa43f43b3e3d1f3c87a8b1da34addbfc9279209fe17209a0eb187b0904e145b"
):
    raise ValueError("pinned runtime differs from the historical runtime-contract binding")
inventory = {
    "schema_version": "sgw-01-r005-all-native-config-inventory-v1",
    "failure_report": FAILURE,
    "construction_source": failure["construction_source"],
    "input_bindings": failure["input_bindings"],
    "records": records,
    "base_config": {
        "export_path": "base_env_cfg.json",
        "bytes": len(base_raw), "sha256": hashlib.sha256(base_raw).hexdigest(),
    },
    "semantic_profiles": profiles,
    "pinned_robolab_source": {
        "checkout": ROBOLAB_ROOT, "revision": ROBOLAB_REVISION, "files": source_summaries,
        "boundary": "Exact Git blobs, not an independent historical imported-module byte attestation.",
    },
    "unique_config_files": len({row["sha256"] for row in expected.values()}),
    "config_bytes_verified_including_identical_files": sum(row["bytes"] for row in expected.values()),
    "release_permitted": False, "historical_population_coverage_complete": False,
    "model_requests": 0, "behavioral_episodes": 0,
    "claim_boundary": (
        "Complete config selection for these twenty recorded lifecycles only. Configured "
        "object names, spawn assets and initial poses are not measured later states. "
        "No missing materialization pose, transitive asset integrity, historical exclusion "
        "or global population coverage is inferred by extraction."
    ),
}
files["inventory.json"] = (json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
    for name, raw in sorted(files.items()):
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
