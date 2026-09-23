"""Recover only the 432 named final V3-D001 cells and their reset snapshots."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

COMPACT = Path("/data/users/ali/vla_wam/raw/v3d/v3d001_behavior/compiled_result_attempt01/pi05_v3d001_episodes.jsonl")
RUNTIME = Path("/data/users/ali/vla_wam/raw/v3/pi05_current_stack/release/runtime_identity_b200gpu0_rtxexpansion_attempt03.json")
SOURCE = Path("/data/users/ali/vla_wam/src/steerable-v3d001-4c7ad8b")
BRIDGE = "experiments/v3/pi05_stochastic_v3d001/robolab_bridge.py"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound(binding):
    raw = Path(binding["path"]).read_bytes()
    if len(raw) != binding["bytes"] or digest(raw) != binding["sha256"]:
        raise ValueError(f"original artifact differs: {binding['path']}")
    return raw


compact_binding = {"path": str(COMPACT), "bytes": 1719784, "sha256": "2586bdc4f963a610ea26f5fbe609f9a8c133d85e9f83b58ac6dfe3dd4c798976"}
runtime_binding = {"path": str(RUNTIME), "bytes": 4692, "sha256": "e73fe7a0cc22db09fa8fdc0babf80dd8ad3280d0502285c6ad1c4d822c7fa532"}
rows = [json.loads(line) for line in bound(compact_binding).splitlines()]
if len(rows) != 432 or len({r["registered_cell_id"] for r in rows}) != 432:
    raise ValueError("final-cell selection differs")
runtime_raw = bound(runtime_binding)
entries = []
for row in sorted(rows, key=lambda r: r["registered_cell_id"]):
    raw_rows = [json.loads(line) for line in bound(row["raw_episode_jsonl"]).splitlines()]
    if len(raw_rows) != 1:
        raise ValueError("original cell is not a single raw episode")
    raw = raw_rows[0]
    capture = json.loads(bound(row["state_capture"]))
    attestations = row["pre_action_reset_attestations"]
    if (raw["registered_cell_id"] != row["registered_cell_id"] or raw["behavioral_result_valid"] is not True
            or raw["environment_seed"] != row["environment_seed"]
            or raw["requested_relation"] != row["requested_relation"]
            or raw["initial_state_sha256"] != row["initial_state_sha256"]
            or raw["runtime_identity"]["sha256"] != runtime_binding["sha256"]
            or raw["source_artifacts"]["state_capture"] != row["state_capture"]
            or raw["source_artifacts"]["pre_action_reset_attestations"] != attestations
            or capture["pre_action_reset_attestations"] != attestations
            or raw["steps"] != capture["samples"]):
        raise ValueError("raw/capture/reset provenance differs")
    if len(attestations) != 2 or capture["pre_action_reset_count"] != 2:
        raise ValueError("expected two retained pre-action reset snapshots")
    resets = [{"binding": b, "value": json.loads(bound(b))} for b in attestations]
    entries.append({
        "registered_cell_id": row["registered_cell_id"],
        "attempt_id": raw["attempt_id"],
        "environment_seed": row["environment_seed"],
        "requested_relation": row["requested_relation"],
        "sampling_index": row["shared_policy_sampling_seed_index"],
        "initial_state_sha256": row["initial_state_sha256"],
        "raw_initial_state_sha256": raw["initial_state_sha256"],
        "raw_behavioral_result_valid": raw["behavioral_result_valid"],
        "measurement_frame": raw["measurement_frame"],
        "runtime_identity": raw["runtime_identity"],
        "raw_episode": row["raw_episode_jsonl"],
        "state_capture": row["state_capture"],
        "raw_and_capture_samples_equal": True,
        "initial_sample": raw["steps"][0],
        "reset_attestations": resets,
    })
head = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
bridge = (SOURCE / BRIDGE).read_bytes()
if head != "6bd377f4e72ed2b285dd17654360527c1ee7ecd1" or digest(bridge) != "18887e4d80d849f78406a7b8305e12ce9bc6201d64010a4addb7af21d2863c73":
    raise ValueError("retained present-day source observation changed")
if bridge != subprocess.check_output(["git", "-C", str(SOURCE), "show", f"{head}:{BRIDGE}"]):
    raise ValueError("retained current bridge differs from its current Git object")
value = {
    "schema_version": "sgw-01-pi05-stochastic-final-reset-export-v1",
    "compact_episodes": compact_binding, "runtime_identity": runtime_binding,
    "entries": entries,
    "retained_current_source": {
        "root": str(SOURCE), "git_head": head, "bridge_path": BRIDGE,
        "bridge_bytes": len(bridge), "bridge_sha256": digest(bridge),
        "matches_current_git_object": True,
        "independent_historical_source_attestation": False,
    },
    "new_model_requests": 0, "new_behavioral_episodes": 0, "release_permitted": False,
}
files = {
    "resets.json": (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    "runtime_identity.json": runtime_raw,
}
probe = Path("/data/users/ali/vla_wam/raw/v3d/v3d001/eligibility_probe_attempt03")
for name, size, expected in (
    ("evidence_manifest.json", 8413, "6f192d8c179bd36bc8b4246b8013dbec5ec07ea81ec0ad9a2521776b5ba5cb98"),
    ("eligibility_report.json", 4661, "1108ff2c28c269f9dad307e80d363f93bb4ec7a9482894b74460da4281168b66"),
):
    files[name] = bound({"path": str(probe / name), "bytes": size, "sha256": expected})
policy_binding = json.loads(files["eligibility_report.json"])["runtime_attestation"]
policy_raw = Path(policy_binding["path"]).read_bytes()
if digest(policy_raw) != policy_binding["sha256"]:
    raise ValueError("policy runtime attestation changed")
files["policy_runtime_attestation.json"] = policy_raw
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    for name, raw in sorted(files.items()):
        member = tarfile.TarInfo(name)
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
