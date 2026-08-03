#!/usr/bin/env python3
"""Compile the frozen V2-A007 DreamZero six-cell DROID gate and media.

This compiler accepts only a complete six-cell raw collection manifest. It
validates the simulator, exact executed actions, official returned chunks,
measurement-only latent/decoded future retention, and the A007 source/runtime
contracts before emitting compact evidence. It never synthesizes a missing
cell or treats a missing future as zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


np: Any = None


REPO_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = REPO_ROOT / "artifacts/vla_wam_shared_v2/pilot/post_result_dreamzero_amendment.json"
DEFAULT_RESULT = REPO_ROOT / "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json"
DEFAULT_MEDIA_DIR = REPO_ROOT / "artifacts/vla_wam_shared_v2/media/dreamzero_droid"
GALLERY_RENDERER = REPO_ROOT / "tools/render_vla_wam_video_first_gallery.py"

MODEL_ID = "dreamzero_droid"
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
SEEDS = (8300, 8301, 8302)
RETURNED_HORIZON = 24
EXECUTION_HORIZON = 8
OFFICIAL_NOISE_SEED = 1140
PICKUP_LIFT_M = 0.03
MOTION_M = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, repository_relative: bool = False) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing evidence file: {path}")
    display = str(path.relative_to(REPO_ROOT)) if repository_relative else str(path)
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Missing JSON evidence: {path}")
    return json.loads(path.read_text())


def resolved(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def rows_from_ledger(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = load_json(path)
    if isinstance(value, list):
        return value
    for key in ("events", "attempts", "interventions", "rows"):
        if isinstance(value.get(key), list):
            return value[key]
    raise RuntimeError(f"Cannot identify ledger rows: {path}")


def rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    value /= np.linalg.norm(value, axis=1, keepdims=True)
    w, x, y, z = value.T
    matrix = np.empty((len(value), 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def robot_frame_delta(cube: np.ndarray, bowl: np.ndarray, robot: np.ndarray) -> np.ndarray:
    world_delta = cube[:, :3] - bowl[:, :3]
    return np.einsum("tij,ti->tj", rotation_wxyz(robot[:, 3:7]), world_delta)


def relation_mask(delta: np.ndarray, relation: str) -> np.ndarray:
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    sign = 1.0 if relation == "left" else -1.0
    cosine = np.divide(
        sign * delta[:, 1], horizontal,
        out=np.zeros_like(horizontal), where=horizontal > 1e-8,
    )
    return cosine >= math.cos(math.radians(45.0))


def first_consecutive(mask: np.ndarray, count: int = 3) -> int | None:
    if len(mask) < count:
        return None
    hits = np.convolve(mask.astype(np.int8), np.ones(count, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(hits == count)
    return int(indices[0]) if len(indices) else None


def initial_fingerprint(group: Any) -> str:
    import h5py

    arrays: dict[str, np.ndarray] = {}

    def collect(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset) and name.startswith(("articulation/", "rigid_object/")):
            arrays[name] = np.asarray(item)

    group.visititems(collect)
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def validate_hashed_entry(entry: dict[str, Any], base: Path, label: str) -> Path:
    path = resolved(entry["path"], base)
    if not path.is_file() or path.stat().st_size != entry.get("bytes", path.stat().st_size):
        raise RuntimeError(f"Missing or byte-mismatched {label}: {path}")
    if sha256(path) != entry["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {label}: {path}")
    return path


def validate_checkpoint_payloads(
    collection: dict[str, Any], base: Path, selected: dict[str, Any]
) -> tuple[Path, Path]:
    """Validate the committed nested manifest against an explicit PVC root.

    The compact manifest belongs in Git while its 64.8 GB payload remains on
    the PVC. Relative payload paths therefore resolve against the explicitly
    declared checkpoint root, never against the manifest's Git directory.
    """
    checkpoint_manifest = resolved(collection["checkpoint_payload_manifest"], base)
    checkpoint = load_json(checkpoint_manifest)
    payload = checkpoint.get("checkpoint")
    if (
        checkpoint.get("schema_version")
        != "vla-wam-shared-v2-dreamzero-official-source-checkpoint-manifest-v1"
        or checkpoint.get("status") != "verified"
        or not isinstance(payload, dict)
        or payload.get("repository") != selected["checkpoint"]
        or payload.get("revision") != selected["checkpoint_revision"]
        or payload.get("payload_file_count")
        != selected["checkpoint_observed_file_count"]
        or payload.get("payload_bytes")
        != selected["checkpoint_observed_payload_bytes"]
    ):
        raise RuntimeError("Committed DreamZero checkpoint manifest contract mismatch")
    checkpoint_files = payload.get("files")
    if (
        not isinstance(checkpoint_files, list)
        or len(checkpoint_files) != selected["checkpoint_observed_file_count"]
        or sum(int(record["bytes"]) for record in checkpoint_files)
        != selected["checkpoint_observed_payload_bytes"]
    ):
        raise RuntimeError("Checkpoint manifest payload inventory differs from A007")

    root_value = collection.get("checkpoint_payload_root")
    if not isinstance(root_value, str) or not root_value:
        raise RuntimeError("Collection must declare checkpoint_payload_root")
    checkpoint_root = resolved(root_value, base)
    if not checkpoint_root.is_dir():
        raise RuntimeError(f"Checkpoint payload root is not a directory: {checkpoint_root}")
    for index, record in enumerate(checkpoint_files):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"Checkpoint payload path escapes explicit root: {record['path']}"
            )
        candidate = (checkpoint_root / relative).resolve()
        try:
            candidate.relative_to(checkpoint_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Checkpoint payload path escapes explicit root: {record['path']}"
            ) from exc
        validate_hashed_entry(
            {**record, "path": str(candidate)}, checkpoint_root,
            f"checkpoint payload {index}",
        )
    return checkpoint_manifest, checkpoint_root


def load_simulator(cell: dict[str, Any], base: Path) -> tuple[dict[str, Any], np.ndarray, str]:
    import h5py

    seed = int(cell["environment_seed"])
    relation = cell["requested_relation"]
    task_dir = resolved(cell["simulator_task_dir"], base)
    if task_dir.name != TASKS[relation]:
        raise RuntimeError(f"Task/relation mismatch: {task_dir}")
    hdf5_path = task_dir / "run_0.hdf5"
    log_path = task_dir / "log_0_env0.json"
    env_path = task_dir / "env_cfg.json"
    videos = sorted(task_dir.glob("*_viewport.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"Expected one RTX viewport video in {task_dir}, found {videos}")
    log, env = load_json(log_path), load_json(env_path)
    if env.get("instruction") != PROMPTS[relation] or int(env.get("seed", -1)) != seed:
        raise RuntimeError(f"Static prompt/seed mismatch: {seed}/{relation}")
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data/demo_0"]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        cube = np.asarray(demo["states/rigid_object/rubiks_cube/root_pose"], dtype=np.float64)
        bowl = np.asarray(demo["states/rigid_object/bowl/root_pose"], dtype=np.float64)
        robot = np.asarray(demo["states/articulation/robot/root_pose"], dtype=np.float64)
        fingerprint = initial_fingerprint(demo["initial_state"])
    steps = int(log["final_step"])
    if not (len(actions) == len(cube) == len(bowl) == len(robot) == steps):
        raise RuntimeError(f"Simulator trajectory length mismatch: {seed}/{relation}")
    delta = robot_frame_delta(cube, bowl, robot)
    requested = relation_mask(delta, relation)
    opposite = relation_mask(delta, "right" if relation == "left" else "left")
    lift = cube[:, 2] - cube[0, 2]
    movement = np.linalg.norm(cube[:, :3] - cube[0, :3], axis=1)
    pickup = first_consecutive(lift >= PICKUP_LIFT_M)
    interaction = first_consecutive(movement >= MOTION_M)
    entered_indices = np.flatnonzero(requested)
    entered = int(entered_indices[0]) if len(entered_indices) else None
    success = bool(log["success"])
    failure_stage = (
        "success" if success else
        "no_object_interaction" if interaction is None else
        "object_moved_no_verified_pickup" if pickup is None else
        "picked_never_entered_requested_region" if entered is None else
        "entered_requested_region_not_released"
    )
    return ({
        "requested_success": success,
        "actions_executed": steps,
        "failure_stage": failure_stage,
        "initial_lateral_display_m": float(-delta[0, 1]),
        "final_lateral_display_m": float(-delta[-1, 1]),
        "verified_pickup_proxy": pickup is not None,
        "first_verified_pickup_proxy_step": pickup,
        "ever_entered_requested_region": bool(np.any(requested)),
        "first_requested_region_step": entered,
        "final_requested_relation": bool(requested[-1]),
        "ever_entered_opposite_region": bool(np.any(opposite)),
        "max_object_lift_m": float(np.max(lift)),
        "physical_initial_state_sha256": fingerprint,
        "raw_hdf5": file_record(hdf5_path),
        "raw_log": file_record(log_path),
        "raw_env_config": file_record(env_path),
        "simulator_video": file_record(videos[0]),
    }, actions, fingerprint)


def validate_actions_and_futures(
    cell: dict[str, Any], base: Path, simulator_actions: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    seed, relation = int(cell["environment_seed"]), cell["requested_relation"]
    trace_path = resolved(cell["action_trace_metadata"], base)
    trace = load_json(trace_path)
    if (
        trace.get("schema_version") != "vla-wam-shared-v2-dreamzero-action-trace-v1"
        or trace.get("prompt") != PROMPTS[relation]
        or trace.get("requested_relation") != relation
        or trace.get("sampling_seed_label") != seed
        or trace.get("effective_official_model_noise_seed") != OFFICIAL_NOISE_SEED
        or trace.get("open_loop_execution_horizon") != EXECUTION_HORIZON
        or trace.get("returned_action_horizon") != RETURNED_HORIZON
    ):
        raise RuntimeError(f"Action trace contract mismatch: {seed}/{relation}")
    executed_path = validate_hashed_entry(trace["executed_actions"], trace_path.parent, "executed actions")
    raw_path = validate_hashed_entry(trace["returned_raw_chunks"], trace_path.parent, "raw chunks")
    executable_path = validate_hashed_entry(
        trace["returned_executable_chunks"], trace_path.parent, "executable chunks"
    )
    executed = np.load(executed_path, allow_pickle=False)
    raw_chunks = np.load(raw_path, allow_pickle=False)
    executable = np.load(executable_path, allow_pickle=False)
    requests = int(trace["request_count"])
    if executed.shape != simulator_actions.shape or not np.array_equal(executed, simulator_actions):
        raise RuntimeError(f"Trace differs from simulator actions: {seed}/{relation}")
    if raw_chunks.shape != (requests, RETURNED_HORIZON, 8) or executable.shape != raw_chunks.shape:
        raise RuntimeError(f"Returned DreamZero chunk shape mismatch: {seed}/{relation}")
    expected_requests = math.ceil(len(executed) / EXECUTION_HORIZON)
    reconstructed = executable[:, :EXECUTION_HORIZON].reshape(-1, 8)[: len(executed)]
    if requests != expected_requests or not np.array_equal(reconstructed, executed):
        raise RuntimeError(f"Executed/open-loop chunk mismatch: {seed}/{relation}")

    future_path = resolved(cell["future_manifest"], base)
    future = load_json(future_path)
    if (
        future.get("schema_version") != "vla-wam-shared-v2-dreamzero-future-retention-v1"
        or future.get("official_repository_commit") != "ab790c198fbce33503358efbbd4187ce9a89adf3"
        or future.get("instrumentation_role") != "measurement_only"
        or future.get("request_count") != requests
        or len(future.get("requests", [])) != requests
    ):
        raise RuntimeError(f"Future-retention contract mismatch: {seed}/{relation}")
    future_requests = []
    for index, request in enumerate(future["requests"]):
        if request.get("request_index") != index or request.get("prompt") != PROMPTS[relation]:
            raise RuntimeError(f"Future prompt/index mismatch: {seed}/{relation}/{index}")
        official_action_path = validate_hashed_entry(
            request["official_action"], future_path.parent, "server official action"
        )
        latent_path = validate_hashed_entry(request["latent_video"], future_path.parent, "latent future")
        official_action = np.load(official_action_path, allow_pickle=False)
        if not np.array_equal(official_action, raw_chunks[index]):
            raise RuntimeError(f"Measurement instrumentation changed action: {seed}/{relation}/{index}")
        future_requests.append({
            "request_index": index,
            "official_action": file_record(official_action_path),
            "latent_video": file_record(latent_path),
            "latent_shape": request["latent_video"]["shape"],
        })
    decoded = []
    for record in future.get("official_reset_decode", []):
        path = validate_hashed_entry(record, future_path.parent, "official decoded future")
        decoded.append(file_record(path))
    return ({
        "action_trace_metadata": file_record(trace_path),
        "executed_action_trace": file_record(executed_path),
        "returned_raw_chunks": file_record(raw_path),
        "returned_executable_chunks": file_record(executable_path),
        "policy_request_count": requests,
        "future_interface": "joint_action_and_latent_video_prediction_with_official_decode_path",
        "future_manifest": file_record(future_path),
        "latent_future_request_count": len(future_requests),
        "latent_future_requests": future_requests,
        "official_decoded_future_count": len(decoded),
        "official_decoded_futures": decoded,
        "missing_future_evidence_scored_as_zero": False,
    }, executed)


def rms(left: np.ndarray, right: np.ndarray, limit: int = 8) -> float:
    count = min(len(left), len(right), limit)
    delta = left[:count].astype(np.float64) - right[:count].astype(np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def parse_ffmpeg_duration(stderr: str) -> float:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if match is None:
        raise RuntimeError("ffmpeg did not report an input duration")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Invalid ffmpeg input duration: {duration}")
    return duration


def ffprobe_duration(ffprobe: Path | None, ffmpeg: Path, video: Path) -> float:
    if ffprobe is None:
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-i", str(video)],
            check=False, capture_output=True, text=True,
        )
        return parse_ffmpeg_duration(result.stderr)
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Invalid video duration: {video}")
    return duration


def compose_pair(ffmpeg: Path, ffprobe: Path | None, left: Path, right: Path, output: Path) -> dict[str, Any]:
    left_duration = ffprobe_duration(ffprobe, ffmpeg, left)
    right_duration = ffprobe_duration(ffprobe, ffmpeg, right)
    duration = max(left_duration, right_duration)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.mp4")
    if output.exists() or temporary.exists():
        raise RuntimeError(f"Refusing to overwrite publication media: {output}")
    filter_graph = (
        f"[0:v]setpts=PTS-STARTPTS,scale=640:480:force_original_aspect_ratio=decrease,"
        f"pad=640:480:(ow-iw)/2:(oh-ih)/2:black,tpad=stop_mode=clone:stop_duration={duration:.6f}[l];"
        f"[1:v]setpts=PTS-STARTPTS,scale=640:480:force_original_aspect_ratio=decrease,"
        f"pad=640:480:(ow-iw)/2:(oh-ih)/2:black,tpad=stop_mode=clone:stop_duration={duration:.6f}[r];"
        "[l][r]hstack=inputs=2[v]"
    )
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(left), "-i", str(right),
        "-filter_complex", filter_graph, "-map", "[v]", "-an", "-t", f"{duration:.6f}",
        "-r", "15", "-c:v", "libx264", "-preset", "slow", "-crf", "27",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "1",
        "-map_metadata", "-1", str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(output)
    return {
        **file_record(output, repository_relative=True),
        "layout": "LEFT command on the left; RIGHT command on the right",
        "fps": 15,
        "temporal_alignment": "The shorter complete rollout holds its final frame until the longer complete rollout ends.",
        "ffmpeg_command": command[:-1] + [str(output)],
    }


def validate_collection(collection: dict[str, Any], base: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    amendment = load_json(AMENDMENT_PATH)
    selected = amendment["selected_model"]
    if (
        collection.get("schema_version") != "vla-wam-shared-v2-dreamzero-raw-collection-v1"
        or collection.get("status") != "complete"
        or collection.get("model_id") != MODEL_ID
        or collection.get("amendment_id") != "V2-A007"
        or collection.get("official_repository_commit") != selected["repository_commit"]
        or collection.get("checkpoint_revision") != selected["checkpoint_revision"]
    ):
        raise RuntimeError("Collection source/protocol contract mismatch")
    cells = collection.get("cells", [])
    expected = {(seed, relation) for seed in SEEDS for relation in PROMPTS}
    observed = {(int(cell["environment_seed"]), cell["requested_relation"]) for cell in cells}
    if len(cells) != 6 or observed != expected:
        raise RuntimeError(f"Expected exact six-cell A007 grid, got {sorted(observed)}")
    for cell in cells:
        seed, relation = int(cell["environment_seed"]), cell["requested_relation"]
        if (
            int(cell.get("sampling_seed", -1)) != seed
            or cell.get("prompt") != PROMPTS[relation]
            or cell.get("prompt_family") != "direct_command"
            or cell.get("prompt_controller") != "episode_static"
            or cell.get("oracle_actions") != 0
            or cell.get("dynamic_prompt_switches") != 0
            or cell.get("simulator_gpu_lane") != "raytrace-rtxpro6000-ali"
        ):
            raise RuntimeError(f"Cell protocol mismatch: {seed}/{relation}")

    server_contract_path = resolved(collection["server_contract"], base)
    server = load_json(server_contract_path)
    if (
        server.get("schema_version") != "vla-wam-shared-v2-dreamzero-server-contract-v1"
        or server.get("world_size") != 2
        or server.get("port") == 5000
        or len(server.get("visible_device_names", [])) != 2
        or any("B200" not in name for name in server.get("visible_device_names", []))
        or server.get("official_repository_commit") != selected["repository_commit"]
    ):
        raise RuntimeError("DreamZero study server contract mismatch")
    probe_path = resolved(collection["exact_repeat_probe"], base)
    probe = load_json(probe_path)
    if (
        probe.get("schema_version") != "vla-wam-shared-v2-dreamzero-exact-repeat-probe-v1"
        or probe.get("status") != "passed" or probe.get("passed") is not True
    ):
        raise RuntimeError("DreamZero exact-repeat/sensitivity gate did not pass")
    checkpoint_manifest, checkpoint_root = validate_checkpoint_payloads(
        collection, base, selected
    )
    metrics = probe.get("metrics", {})
    if not (
        metrics.get("left_exact_repeat_action_array_equal") is True
        and metrics.get("left_exact_repeat_latent_array_equal") is True
        and metrics.get("left_vs_right_action_rms", 0) > 0
        and metrics.get("left_vs_right_latent_rms", 0) > 0
    ):
        raise RuntimeError("DreamZero probe metrics do not satisfy the frozen release gate")
    probe_records = probe.get("records", {})
    if set(probe_records) != {"left_a", "left_b", "right"}:
        raise RuntimeError("DreamZero probe must retain exactly LEFT/repeat-LEFT/RIGHT")
    probe_decoded_futures = sum(
        int(record.get("official_decode_count", -1))
        for record in probe_records.values()
    )
    if probe_decoded_futures != 3:
        raise RuntimeError("DreamZero probe must retain one official decode per request")
    invalid_path = resolved(collection["invalid_attempt_ledger"], base)
    intervention_path = resolved(collection["runtime_intervention_ledger"], base)
    invalid, interventions = rows_from_ledger(invalid_path), rows_from_ledger(intervention_path)
    if len(invalid) != int(collection["invalid_attempt_count"]):
        raise RuntimeError("Invalid-attempt count does not match its separate ledger")
    if len(interventions) != int(collection["runtime_intervention_count"]):
        raise RuntimeError("Runtime-intervention count does not match its ledger")
    provenance = {
        "amendment": file_record(AMENDMENT_PATH, repository_relative=True),
        "server_contract": file_record(server_contract_path),
        "checkpoint_payload_manifest": file_record(checkpoint_manifest),
        "checkpoint_payload_root": str(checkpoint_root),
        "exact_repeat_probe": file_record(probe_path),
        "fixed_observation_probe_retention": {
            "request_count": len(probe_records),
            "latent_future_count": len(probe_records),
            "official_decoded_future_count": probe_decoded_futures,
        },
        "invalid_attempt_ledger": file_record(invalid_path),
        "runtime_intervention_ledger": file_record(intervention_path),
        "invalid_attempt_count": len(invalid),
        "runtime_intervention_count": len(interventions),
    }
    return cells, invalid, provenance


def compile_collection(args: argparse.Namespace) -> None:
    global np
    import numpy as numpy_module

    np = numpy_module
    collection_path = args.collection_manifest.resolve()
    collection = load_json(collection_path)
    cells, invalid_attempts, provenance = validate_collection(collection, collection_path.parent)
    episodes, action_arrays = [], {}
    for cell in sorted(cells, key=lambda row: (int(row["environment_seed"]), row["requested_relation"])):
        seed, relation = int(cell["environment_seed"]), cell["requested_relation"]
        simulation, simulator_actions, fingerprint = load_simulator(cell, collection_path.parent)
        future, executed = validate_actions_and_futures(
            cell, collection_path.parent, simulator_actions
        )
        episode = {
            "schema_version": "vla-wam-shared-v2-dreamzero-episode-v1",
            "model_id": MODEL_ID,
            "amendment_id": "V2-A007",
            "pair_id": f"droid_pair_seed_{seed}",
            "environment_seed": seed,
            "sampling_seed": seed,
            "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
            "requested_relation": relation,
            "prompt": PROMPTS[relation],
            "prompt_family": "direct_command",
            "prompt_controller": "episode_static",
            "oracle_actions": 0,
            "dynamic_prompt_switches": 0,
            "simulator_gpu_lane": "raytrace-rtxpro6000-ali",
            **simulation,
            **future,
        }
        episodes.append(episode)
        action_arrays[(seed, relation)] = executed

    pairs = []
    for seed in SEEDS:
        left = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "left")
        right = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "right")
        if left["physical_initial_state_sha256"] != right["physical_initial_state_sha256"]:
            raise RuntimeError(f"Matched initial state differs at seed {seed}")
        shift = right["final_lateral_display_m"] - left["final_lateral_display_m"]
        pairs.append({
            "pair_id": f"droid_pair_seed_{seed}",
            "environment_seed": seed,
            "sampling_seed": seed,
            "physical_initial_state_sha256": left["physical_initial_state_sha256"],
            "left_requested_success": left["requested_success"],
            "right_requested_success": right["requested_success"],
            "right_minus_left_endpoint_lateral_m": shift,
            "endpoint_ordering_aligned": bool(shift > 0),
            "executed_actions_distinct": bool(not np.array_equal(action_arrays[(seed, "left")], action_arrays[(seed, "right")])),
            "first_8_action_rms": rms(action_arrays[(seed, "left")], action_arrays[(seed, "right")]),
        })
    successes = {
        relation: sum(row["requested_success"] for row in episodes if row["requested_relation"] == relation)
        for relation in PROMPTS
    }
    competence = (
        "both_directions" if successes["left"] and successes["right"] else
        "left_only" if successes["left"] else "right_only" if successes["right"] else "zero_direction"
    )
    behavioral_latent_count = sum(
        int(row["latent_future_request_count"]) for row in episodes
    )
    behavioral_decode_count = sum(
        int(row["official_decoded_future_count"]) for row in episodes
    )
    probe_retention = provenance["fixed_observation_probe_retention"]
    result = {
        "schema_version": "vla-wam-shared-v2-dreamzero-droid-direct-gate-v1",
        "status": "complete",
        "model_id": MODEL_ID,
        "amendment_id": "V2-A007",
        "compiled_at_git_head": args.git_head,
        "collection_manifest": file_record(collection_path),
        "valid_episode_count": 6,
        "valid_failure_count": 6 - sum(successes.values()),
        "requested_success_count": sum(successes.values()),
        "success_by_relation": {
            relation: {"successes": successes[relation], "trials": 3} for relation in PROMPTS
        },
        "aligned_endpoint_pair_count": sum(row["endpoint_ordering_aligned"] for row in pairs),
        "distinct_executed_action_pair_count": sum(row["executed_actions_distinct"] for row in pairs),
        "competence_gate": competence,
        "wording_grid_eligible": competence == "both_directions",
        "future_interface": "joint_action_and_latent_video_prediction_with_official_decode_path",
        "missing_or_unexposed_future_evidence_scored_as_zero": False,
        "future_retention_audit": {
            "behavioral_episode_count": len(episodes),
            "behavioral_latent_future_count": behavioral_latent_count,
            "behavioral_official_decoded_future_count": behavioral_decode_count,
            "fixed_observation_probe_request_count": probe_retention["request_count"],
            "fixed_observation_probe_latent_future_count": probe_retention["latent_future_count"],
            "fixed_observation_probe_official_decoded_future_count": probe_retention["official_decoded_future_count"],
            "total_server_episode_count": len(episodes) + probe_retention["request_count"],
            "total_retained_latent_future_count": behavioral_latent_count + probe_retention["latent_future_count"],
            "total_official_reset_decode_count": behavioral_decode_count + probe_retention["official_decoded_future_count"],
        },
        "provenance": provenance,
        "pairs": pairs,
        "episodes": episodes,
        "invalid_attempts": invalid_attempts,
    }
    media_manifest_path = args.media_dir / "media_manifest.json"
    result_stage = args.result_output.with_name(args.result_output.name + ".staging")
    media_manifest_stage = media_manifest_path.with_name(media_manifest_path.name + ".staging")
    video_paths = {
        seed: args.media_dir / f"dreamzero_droid_seed{seed}_paired.mp4" for seed in SEEDS
    }
    video_stages = {
        seed: path.with_name(path.name + ".staging.mp4") for seed, path in video_paths.items()
    }
    targets = [args.result_output, result_stage, media_manifest_path, media_manifest_stage]
    targets.extend(video_paths.values())
    targets.extend(video_stages.values())
    existing = [path for path in targets if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite result/media or stale staging files: {existing}")
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.media_dir.mkdir(parents=True, exist_ok=True)
    result_stage.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    media_entries = []
    for seed in SEEDS:
        left = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "left")
        right = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "right")
        video_path = video_paths[seed]
        video = compose_pair(
            args.ffmpeg.resolve(), args.ffprobe.resolve() if args.ffprobe else None,
            Path(left["simulator_video"]["path"]), Path(right["simulator_video"]["path"]),
            video_stages[seed],
        )
        video["path"] = str(video_path.relative_to(REPO_ROOT))
        directions = [
            {
                "relation": relation.upper(),
                "prompt": PROMPTS[relation],
                "outcome": (
                    f"success after {episode['actions_executed']} actions" if episode["requested_success"]
                    else f"failure after {episode['actions_executed']} actions: {episode['failure_stage']}"
                ),
            }
            for relation, episode in (("left", left), ("right", right))
        ]
        media_entries.append({
            "id": f"dreamzero_droid_seed{seed}",
            "arena": "droid",
            "arena_label": "DROID / RoboLab",
            "model_label": "DreamZero DROID",
            "category": "WAM",
            "future_interface": "Joint actions and latent video prediction with official decoded-future path",
            "evidence_status": "Valid V2-A007 behavioral pair; committed RTX PRO 6000 publication video",
            "pair_label": f"seed {seed} matched pair",
            "seed": seed,
            "video": {key: video[key] for key in ("path", "bytes", "sha256")},
            "directions": directions,
            "selection_note": "Complete frozen pair; all three V2-A007 pairs are included, with no outcome-based selection.",
            "source_manifest": str(args.result_output.resolve().relative_to(REPO_ROOT)),
        })
    result_stage.replace(args.result_output)
    for seed in SEEDS:
        video_stages[seed].replace(video_paths[seed])
    ffmpeg_version = subprocess.run(
        [str(args.ffmpeg.resolve()), "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    media_manifest = {
        "schema_version": "vla-wam-shared-v2-dreamzero-droid-media-v1",
        "model_id": MODEL_ID,
        "arena": "droid",
        "source_result": file_record(args.result_output, repository_relative=True),
        "publication_policy": "All three exact LEFT/RIGHT V2-A007 pairs are included; no outcome-based selection.",
        "renderer": {
            "ffmpeg": file_record(args.ffmpeg.resolve()),
            "duration_probe": (
                {"backend": "ffprobe_json", **file_record(args.ffprobe.resolve())}
                if args.ffprobe else
                {"backend": "ffmpeg_input_stderr", **file_record(args.ffmpeg.resolve())}
            ),
            "ffmpeg_version": ffmpeg_version,
        },
        "gallery_entries": media_entries,
    }
    media_manifest_stage.write_text(json.dumps(media_manifest, indent=2, sort_keys=True) + "\n")
    media_manifest_stage.replace(media_manifest_path)
    if args.regenerate_gallery:
        subprocess.run([sys.executable, str(GALLERY_RENDERER)], check=True, cwd=REPO_ROOT)
    print(json.dumps({
        "status": "complete",
        "result": str(args.result_output),
        "media_manifest": str(media_manifest_path),
        "valid_episode_count": 6,
        "success_by_relation": successes,
        "competence_gate": competence,
        "publication_pair_count": 3,
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-pending", action="store_true")
    parser.add_argument("--collection-manifest", type=Path)
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--git-head")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--regenerate-gallery", action="store_true")
    args = parser.parse_args()
    if args.check_pending:
        canonical = DEFAULT_MEDIA_DIR / "media_manifest.json"
        if canonical.exists():
            raise SystemExit(f"DreamZero media is no longer pending: {canonical}")
        if args.regenerate_gallery:
            subprocess.run([sys.executable, str(GALLERY_RENDERER)], check=True, cwd=REPO_ROOT)
        print(json.dumps({"status": "pending", "behavioral_episode_count": 0, "media_manifest_exists": False}))
        return
    missing = [
        flag for flag, value in (
            ("--collection-manifest", args.collection_manifest),
            ("--git-head", args.git_head),
            ("--ffmpeg", args.ffmpeg),
        ) if value is None
    ]
    if missing:
        parser.error(f"complete compilation requires: {', '.join(missing)}")
    args.result_output = args.result_output.resolve()
    args.media_dir = args.media_dir.resolve()
    args.result_output.relative_to(REPO_ROOT)
    args.media_dir.relative_to(REPO_ROOT)
    compile_collection(args)


if __name__ == "__main__":
    main()
