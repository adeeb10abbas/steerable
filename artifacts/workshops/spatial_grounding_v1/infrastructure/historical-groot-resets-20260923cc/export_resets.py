"""Recover every declared GR00T Phase-A final-cell reset without running a model."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


SOURCE = Path("/data/users/ali/sgw-01/source/a995e014-bu")
MANIFEST = "artifacts/vla_wam_shared_v3/results/groot_n17_droid_phase_a_evidence_hash_manifest.json"
MANIFEST_SHA = "c7ebddea0413090b18cec6e2044251e929aeef77d24fa0102fdd4f9ac63c09bd"
RAW = Path("/data/users/ali/vla_wam/raw/v3/groot_n17_droid")
PATCH = RAW / "runtime/instrumentation_attempt02"


def read_bound(binding, limit=4 * 1024 * 1024):
    path = Path(binding["path"])
    if binding["bytes"] > limit:
        raise ValueError(f"source exceeds finite extraction limit: {path}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"source byte/hash mismatch: {path}")
    return raw


raw = (SOURCE / MANIFEST).read_bytes()
if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
    raise ValueError("final cohort manifest differs")
manifest = json.loads(raw)
if manifest["pair_count"] != 27 or manifest["episode_count"] != 54:
    raise ValueError("declared cohort size differs")
if [pair["seed"] for pair in manifest["pairs"]] != list(range(8303, 8330)):
    raise ValueError("final pair population differs")
queue_raw = read_bound({**manifest["queue"], "path": str(SOURCE / "artifacts/vla_wam_shared_v3/phase_a_cells.jsonl")})
queue = [json.loads(line) for line in queue_raw.splitlines()]
registered = {row["cell_id"] for row in queue
              if row["model_id"] == "groot_n17_droid_vla" and row["status"] == "authorized_new"}
records, pair_bindings, configs = [], [], {}
for pair in manifest["pairs"]:
    pair_manifest = json.loads(read_bound(pair["pair_evidence_manifest"]))
    if pair_manifest["environment_seed"] != pair["seed"] or pair_manifest["matched_pair_valid"] is not True:
        raise ValueError("pair identity/validity differs")
    pair_bindings.append(pair["pair_evidence_manifest"])
    for relation in ("left", "right"):
        cell = pair["cells"][relation]
        entry = pair_manifest["relations"][relation]
        for outer, inner in (
            ("state_score_trace", "capture"), ("state_stream_partial", "state_stream"),
            ("warmup_reset", "warmup_reset"),
        ):
            if cell[outer] != entry[inner]:
                raise ValueError("final manifest and pair source binding differ")
        capture = json.loads(read_bound(cell["state_score_trace"]))
        stream = [json.loads(line) for line in read_bound(cell["state_stream_partial"]).splitlines()]
        warmup = [json.loads(line) for line in read_bound(cell["warmup_reset"]).splitlines()]
        expected_id = f"v3:droid:groot_n17_droid_vla:seed{pair['seed']}:{relation}"
        if cell["registered_cell_id"] != expected_id or capture["registered_cell_id"] != expected_id:
            raise ValueError("registered cell identity differs")
        if (capture["environment_seed"] != pair["seed"] or capture["requested_relation"] != relation
                or capture["behavioral_result_valid_candidate"] is not True):
            raise ValueError("capture identity/validity differs")
        samples = capture["samples"]
        if len(samples) != capture["actions_executed"] + 1 or len(stream) != len(samples):
            raise ValueError("N+1 raw state accounting differs")
        for index, (sample, streamed) in enumerate(zip(samples, stream, strict=True)):
            if sample["action_step"] != index or any(
                streamed[key] != sample[key] for key in ("action_step", "object_xyz", "reference_xyz", "grippers_open")
            ):
                raise ValueError("full raw state stream disagrees with capture")
        if len(warmup) != 1 or warmup[0]["action_step"] != 0:
            raise ValueError("warmup is not one zero-action reset")
        expected_warmup = Path(cell["warmup_reset"]["path"])
        actual_warmups = set(expected_warmup.parent.glob(f"seed{pair['seed']}_{relation}_warmup_reset*.jsonl"))
        if actual_warmups != {expected_warmup}:
            raise ValueError("additional or missing warmup sidecars require separate accounting")
        config_raw = read_bound(entry["environment"])
        config = json.loads(config_raw)
        projection = {
            "scene_entry_names": sorted(config["scene"]),
            "rigid_object_names": sorted(name for name, value in config["scene"].items()
                                         if isinstance(value, dict) and value.get("class_type")
                                         == "isaaclab.assets.rigid_object.rigid_object:RigidObject"),
            "robot_initial_state": config["scene"]["robot"]["init_state"],
            "robot_spawn": config["scene"]["robot"]["spawn"],
            "scene_spawn": config["scene"]["scene"]["spawn"],
            "terminations": config["terminations"],
        }
        projection_sha = hashlib.sha256(json.dumps(
            projection, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        configs[projection_sha] = projection
        records.append({
            "registered_cell_id": expected_id, "seed": pair["seed"], "relation": relation,
            "state_trace": cell["state_score_trace"], "state_stream": cell["state_stream_partial"],
            "warmup_trace": cell["warmup_reset"], "environment_config": entry["environment"],
            "environment_config_projection_sha256": projection_sha,
            "capture_contract": capture["capture_contract"],
            "sample_count": len(samples), "actions_executed": capture["actions_executed"],
            "episode_initial": samples[0], "warmup_reset": warmup[0],
        })
if {row["registered_cell_id"] for row in records} != registered or len(registered) != 54:
    raise ValueError("recovered cell population differs from frozen authorized-new queue")

files = {}
source_records = []
for name in (
    "manifest.json", "robolab_bridge_runtime.py", "launcher_manifest.json", "launch_groot_v3_pair.sh",
    "validator_manifest.json", "validate_compile_groot_v3_pair.py", "queue_manifest.json",
    "run_groot_v3_queue_8304_8329.sh",
):
    path = PATCH / name
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("runtime source exceeds finite extraction limit")
    files[f"runtime/{name}"] = raw
    source_records.append({
        "path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "export_path": f"runtime/{name}",
        "historical_hash_anchor_in_final_manifest": False,
    })
result = {
    "schema_version": "sgw-01-groot-phase-a-reset-recovery-v1",
    "final_manifest": {"path": MANIFEST, "bytes": (SOURCE / MANIFEST).stat().st_size, "sha256": MANIFEST_SHA},
    "recorded_source_study_commit": manifest["source_study_commit"],
    "queue": manifest["queue"], "pair_manifests": pair_bindings,
    "records": records, "config_projections": configs, "retained_runtime_sources": source_records,
    "complete_declared_final_manifest_cell_selection": True,
    "captured_initial_resets": len(records), "captured_warmup_resets": len(records),
    "historical_population_coverage_complete": False,
    "root_frame_cross_comparison_qualified": False, "release_permitted": False,
    "new_model_requests": 0, "new_behavioral_episodes": 0,
    "claim_boundary": (
        "Every final-manifest authorized-new GR00T Phase-A cell and its named warmup sidecar, "
        "not preflights, infrastructure-invalid attempts, constructor transients or other cohorts. "
        "Roots are preserved in the raw capture's declared robot-base frame, not relabeled world. "
        "Retained runtime patch files are freshly hash-recorded recovery evidence; this exporter "
        "does not infer an independent historical hash anchor for them."
    ),
}
files["resets.json"] = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
    for name, raw in sorted(files.items()):
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
