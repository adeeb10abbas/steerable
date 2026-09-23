"""Read only the final Phase-A pair catalog and its explicitly bound reset streams."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

CATALOG_PATH = "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_evidence_hash_manifest.json"
CATALOG_SHA = "dca633cfa74ff93926cb1d010d5a329908a02d81181f23cca4969c5d5f1f3cfd"
catalog_raw = subprocess.check_output([
    "git", "-C", "/data/users/ali/sgw-01/source/b429ddd", "show",
    f"b429ddd8c3e7be59f63759c1820e352029e48b8c:{CATALOG_PATH}",
])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bound(binding):
    raw = Path(binding["path"]).read_bytes()
    if sha(raw) != binding["sha256"] or ("bytes" in binding and len(raw) != binding["bytes"]):
        raise ValueError(f"original bytes differ: {binding['path']}")
    return raw


if sha(catalog_raw) != CATALOG_SHA:
    raise ValueError("final pair catalog differs")
catalog = json.loads(catalog_raw)
runtime_raw = bound(catalog["runtime_identity"])
runtime = json.loads(runtime_raw)
bridge_binding = runtime["runtime_instrumentation"]
bridge = bound(bridge_binding)
files = {"runtime_bridge.py": bridge, "runtime_identity.json": runtime_raw}
pairs = []
for pair in catalog["pairs"]:
    seed = pair["seed"]
    manifest_raw = bound(pair["pair_evidence_manifest"])
    manifest = json.loads(manifest_raw)
    name = f"pair-manifests/{seed}.json"
    files[name] = manifest_raw
    bindings = {row["path"]: row for row in manifest["files"]}
    if len(bindings) != len(manifest["files"]) or manifest["pair_id"] != pair["pair_id"]:
        raise ValueError("pair manifest identity or membership differs")
    if bindings[bridge_binding["path"]]["sha256"] != bridge_binding["sha256"]:
        raise ValueError("pair does not bind the runtime instrumentation")
    base = Path(pair["pair_evidence_manifest"]["path"]).parent
    preflight = json.loads(bound(bindings[str(base / "preflight.json")]))
    if preflight["runtime_identity"] != runtime or preflight["runtime_identity_sha256"] != sha(runtime_raw):
        raise ValueError("preflight runtime differs")
    log_binding = bindings[str(base / "guard_stdout.log")]
    log = bound(log_binding).decode()
    launches = [json.loads(line) for line in log.splitlines()
                if line.startswith('{"command":') and '"event": "worker_started"' in line]
    if len(launches) != 1:
        raise ValueError("historical launch command is ambiguous")
    command = launches[0]["command"]
    python_index = command.index("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python")
    if command[python_index + 1] != bridge_binding["path"]:
        raise ValueError("launch did not select the bound runtime producer")
    capture_dir = Path(command[command.index("--state-capture-dir") + 1])
    rows = [json.loads(line) for line in bound(pair["behavioral_jsonl"]).splitlines()]
    if len(rows) != 2 or {row["requested_relation"] for row in rows} != {"left", "right"}:
        raise ValueError("final pair does not contain exactly both directions")
    entries = []
    for row in rows:
        relation = row["requested_relation"]
        if (row["behavioral_result_valid"] is not True or row["environment_seed"] != seed
                or row["runtime_identity"]["sha256"] != sha(runtime_raw)):
            raise ValueError("raw episode identity or validity differs")
        capture_binding = bindings[str(capture_dir / f"seed{seed}_{relation}.json")]
        capture = json.loads(bound(capture_binding))
        partial_binding = bindings[capture["capture_contract"]["partial_state_stream"]]
        partial = [json.loads(line) for line in bound(partial_binding).splitlines()]
        setup_binding = bindings[str(capture_dir / f"seed{seed}_{relation}_setup_reset0.partial.jsonl")]
        setup = [json.loads(line) for line in bound(setup_binding).splitlines()]
        if (row["steps"] != capture["samples"] or row["steps"] != partial or len(setup) != 1
                or capture["registered_cell_id"] != row["registered_cell_id"]
                or row["steps"][0]["action_step"] != 0 or setup[0]["action_step"] != 0):
            raise ValueError("raw/capture/partial/reset streams differ")
        entries.append({
            "registered_cell_id": row["registered_cell_id"],
            "environment_seed": seed, "relation": relation,
            "measurement_frame": row["measurement_frame"], "runtime_identity": row["runtime_identity"],
            "initial_state_sha256": row["initial_state_sha256"],
            "initial_sample": row["steps"][0], "setup_sample": setup[0],
            "capture": capture_binding, "canonical_partial_stream": partial_binding,
            "setup_partial_stream": setup_binding,
            "sample_count": len(partial), "raw_capture_and_partial_streams_equal": True,
            "raw_behavioral_result_valid": True,
        })
    pairs.append({
        "pair_id": pair["pair_id"], "seed": seed,
        "manifest_binding": pair["pair_evidence_manifest"], "manifest_export_path": name,
        "raw_pair": pair["behavioral_jsonl"], "preflight": bindings[str(base / "preflight.json")],
        "guard_log": log_binding, "launch_event": launches[0], "entries": entries,
    })
if len(pairs) != 27 or {row["seed"] for row in pairs} != set(range(8303, 8330)):
    raise ValueError("final pair selection differs")
value = {
    "schema_version": "sgw-01-pi05-phase-a-reset-export-v1",
    "catalog_sha256": CATALOG_SHA, "runtime_identity": catalog["runtime_identity"],
    "runtime_instrumentation": bridge_binding, "pairs": pairs,
    "new_model_requests": 0, "new_behavioral_episodes": 0, "release_permitted": False,
}
files["resets.json"] = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    for name, raw in sorted(files.items()):
        member = tarfile.TarInfo(name)
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
