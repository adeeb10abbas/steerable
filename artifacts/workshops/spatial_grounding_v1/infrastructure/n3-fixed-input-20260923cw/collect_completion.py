"""Verify and preserve the completed DK batch without loading any policy."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from experiments.workshops.spatial_grounding_v1.nano_fixed_input import ORDER, compare_actions


def record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(binding):
    path = Path(binding["path"])
    if record(path) != {key: binding[key] for key in ("path", "bytes", "sha256")}:
        raise ValueError(f"retained file differs: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pods", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.run_root / "evidence"
    result = json.loads((evidence / "result.json").read_bytes())
    plan = json.loads(args.plan.read_bytes())
    pods = json.loads(args.pods.read_bytes())["items"]
    jobs = json.loads(args.jobs.read_bytes())["items"]
    owner_path = args.run_root.parent / "request-owner/successor/owner.json"
    owner = json.loads(owner_path.read_bytes())
    names = {entry["job"] for entry in plan["entries"]}
    if len(names) != 8 or {job["metadata"]["name"] for job in jobs} != names:
        raise ValueError("batch must contain exactly the eight registered Jobs")
    if len(pods) != 8 or len({pod["metadata"]["uid"] for pod in pods}) != 8:
        raise ValueError("batch must contain exactly eight distinct Pods")
    job_uids = {job["metadata"]["uid"] for job in jobs}
    for job in jobs:
        if job["status"].get("active", 0) or not any(
            item["type"] in ("Complete", "Failed") and item["status"] == "True"
            for item in job["status"].get("conditions", [])
        ):
            raise ValueError("batch still has a nonterminal Job")
    for pod in pods:
        if pod["status"]["phase"] not in ("Failed", "Succeeded") or not any(
            ref["kind"] == "Job" and ref["uid"] in job_uids
            for ref in pod["metadata"].get("ownerReferences", [])
        ):
            raise ValueError("nonterminal or unowned batch Pod")
    winners = [pod for pod in pods if pod["metadata"]["uid"] == owner["pod_uid"]]
    if len(winners) != 1 or winners[0]["status"]["phase"] != "Succeeded":
        raise ValueError("exclusive successor is not the successful terminal Pod")
    if owner["recovery_output"] != str(args.run_root) or owner["maximum_model_requests"] != 6:
        raise ValueError("successor output or request allocation differs")
    if result["model_requests"] != 6 or result["behavioral_episodes"] != 0 or result["executed_actions"] != 0:
        raise ValueError("result exceeds the six nonbehavioral requests")
    if result["release_permitted"] is not False or (evidence / "failure.json").exists():
        raise ValueError("unexpected release or incomplete-run evidence")
    if json.loads((args.run_root / "process-outcome.json").read_bytes())["exit_code"] != 0:
        raise ValueError("native process did not finish successfully")
    trace = [json.loads(line) for line in (evidence / "trace.jsonl").read_text().splitlines()]
    if len(trace) != 6 or len(list(evidence.glob("request-*/intent.json"))) != 6:
        raise ValueError("request intent/trace count differs")
    if [request["prompt_id"] for request in result["requests"]] != ORDER:
        raise ValueError("registered prompt order differs")

    actions, futures = [], []
    for index, request in enumerate(result["requests"]):
        if request["index"] != index or request["trace"] != trace[index]:
            raise ValueError("per-request trace attribution differs")
        actions.append(np.load(checked(request["actions"]), allow_pickle=False))
        future = np.load(checked(request["future"]), allow_pickle=False)
        checked(request["latent"])
        if future.dtype != np.uint8 or list(future.shape) != [33, 528, 640, 3]:
            raise ValueError("unexpected native future encoding")
        futures.append(future)
    comparisons = compare_actions(actions)
    if comparisons != result["comparisons"]:
        raise ValueError("recomputed action comparisons differ")
    redecodes = result["offline_redecodes"]
    if [row["index"] for row in redecodes] != [3, 4, 5]:
        raise ValueError("offline redecode indices differ")
    for row in redecodes:
        equal = bool(np.array_equal(np.load(checked(row["output"]), allow_pickle=False), futures[row["index"]]))
        if row["new_model_requests"] != 0 or equal != row["native_decode_equal"]:
            raise ValueError("offline redecode receipt differs")
    for entry in plan["entries"]:
        if entry["output"] != str(args.run_root) and (Path(entry["output"]) / "evidence").exists():
            raise ValueError("a nonwinning candidate entered the model runner")

    import imageio_ffmpeg

    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.mkdir(parents=True, exist_ok=False)
    compact = args.output / "compact"
    compact.mkdir()
    for path in sorted(args.run_root.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".log") and "cache" not in path.parts:
            destination = compact / path.relative_to(args.run_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    shutil.copyfile(owner_path, compact / "successor-owner.json")
    raw_files = [record(path) for path in sorted(evidence.rglob("*")) if path.is_file()]
    media_root = args.output / "generated-future-videos"
    media_root.mkdir()
    media = []
    for index, future in enumerate(futures):
        path = media_root / f"request-{index:02d}-{ORDER[index]}-generated-future.mp4"
        subprocess.run([
            encoder, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", "640x528",
            "-framerate", "15", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ], input=future.tobytes(), check=True)
        decoded = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        metadata = next(decoded)
        frame_count = sum(1 for _ in decoded)
        if frame_count != 33 or metadata["size"] != (640, 528) or metadata["fps"] != 15:
            raise ValueError("encoded visualization lost frames or changed dimensions/cadence")
        media.append({
            **record(path), "request_index": index, "prompt_id": ORDER[index],
            "source": result["requests"][index]["future"], "frames": frame_count,
            "presentation_fps": 15, "physical_time_mapping_qualified": False,
            "scope": "Generated local prediction, not an executed robot rollout.",
        })
    summary = {
        "schema_version": "sgw-01-n3-fixed-input-completion-v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "six_requests_completed_independently_rechecked_not_behavioral_release",
        "registration_id": owner["registration_id"],
        "source_commit": owner["source_commit"],
        "compiler": record(Path(__file__)),
        "result": record(evidence / "result.json"),
        "batch_plan": record(args.plan),
        "kubernetes_pods": record(args.pods),
        "kubernetes_jobs": record(args.jobs),
        "successor_owner": record(owner_path),
        "winner_pod": owner["pod_name"],
        "winner_pod_uid": owner["pod_uid"],
        "winner_node": winners[0]["spec"]["nodeName"],
        "gpu": owner["idle"]["selected_gpu"],
        "candidate_pods": [{
            "name": pod["metadata"]["name"], "uid": pod["metadata"]["uid"],
            "node": pod["spec"]["nodeName"], "phase": pod["status"]["phase"],
            "reason": pod["status"].get("reason"),
            "message": pod["status"].get("message"),
            "containers": pod["status"].get("containerStatuses", []),
        } for pod in pods],
        "model_requests": 6,
        "additional_model_requests": 0,
        "behavioral_episodes": 0,
        "executed_actions": 0,
        "release_permitted": False,
        "comparisons": comparisons,
        "future_shapes": [list(future.shape) for future in futures],
        "future_repeat_equal": [bool(np.array_equal(futures[i], futures[i + 1])) for i in (0, 3)],
        "future_opposite_prompt_distinct": [not bool(np.array_equal(futures[i], futures[i + 2])) for i in (0, 3)],
        "offline_redecodes_equal": [row["native_decode_equal"] for row in redecodes],
        "raw_files": raw_files,
        "generated_future_videos": media,
        "existing_smolvla_training_modified": False,
        "scope": "Generated local predictions only. No physical success, live simulator qualification, or time-map release follows from this receipt.",
    }
    with (args.output / "completion.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: summary[key] for key in ("status", "model_requests", "comparisons")}))


if __name__ == "__main__":
    main()
