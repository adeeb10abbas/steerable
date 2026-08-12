#!/usr/bin/env python3
"""Collect one authorized, excluded C002 fixed-observation lane response."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
from openpi_client import websocket_client_policy
REPO_ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO_ROOT))
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, MODEL_ID, read_finite_json, registered_prompts, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002.runtime import validate_runtime_manifest

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True); p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--excluded-smoke-gate", type=Path, required=True); p.add_argument("--runtime-manifest", type=Path, required=True); p.add_argument("--runtime-manifest-sha256", required=True)
    p.add_argument("--fixture", type=Path, required=True); p.add_argument("--fixture-sha256", required=True); p.add_argument("--prompt-condition", default="canonical_left")
    p.add_argument("--sampling-seed", type=int, default=12_000_000); p.add_argument("--remote-host", required=True); p.add_argument("--remote-port", type=int, required=True)
    p.add_argument("--lane-pod-uid", required=True); p.add_argument("--lane-gpu-uuid", required=True); p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists(): raise ContractError(f"refusing to overwrite isolation response: {a.output_dir}")
    smoke = read_finite_json(a.excluded_smoke_gate)
    if not isinstance(smoke, dict) or smoke.get("schema_version") != "vla-wam-shared-v3c002-excluded-smoke-gate-v1" or smoke.get("status") != "passed_excluded_four_cell_smoke" or smoke.get("passed") is not True: raise ContractError("four-cell smoke gate has not passed")
    if smoke.get("excluded_from_behavioral_denominators") is not True or smoke.get("completed_cells") != 4: raise ContractError("smoke gate is not the excluded four-cell gate")
    runtime = validate_runtime_manifest(a.runtime_manifest, a.runtime_manifest_sha256, registration_path=a.registration, queue_path=a.queue, pod_uid=a.lane_pod_uid, gpu_uuid=a.lane_gpu_uuid)["runtime_identity"]
    if runtime.get("server_port") != a.remote_port: raise ContractError("runtime policy-server port changed")
    if not a.fixture.is_file() or sha256_file(a.fixture) != a.fixture_sha256: raise ContractError("fixed observation changed")
    prompt = registered_prompts().get(a.prompt_condition)
    if prompt is None: raise ContractError("isolation prompt condition is unregistered")
    with np.load(a.fixture, allow_pickle=False) as loaded: observation = {key: loaded[key] for key in loaded.files}
    response = websocket_client_policy.WebsocketClientPolicy(a.remote_host, a.remote_port).infer({**observation, "prompt": prompt["prompt"], "sampling_seed": a.sampling_seed})
    if response.get("v2a010_sampling_seed") != a.sampling_seed: raise ContractError("π0.5 server seed echo changed")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.shape != (15, 8) or not np.isfinite(actions).all(): raise ContractError("π0.5 isolation response is not finite [15,8]")
    a.output_dir.mkdir(parents=True, exist_ok=False); action_path = a.output_dir / "actions.npy"; np.save(action_path, actions, allow_pickle=False)
    value = {"schema_version": "vla-wam-shared-v3c002-fixed-observation-lane-v1", "status": "completed_excluded_fixed_observation_lane", "passed": True,
        "model_id": MODEL_ID, "model_request_count": 1, "behavioral_episode_count": 0, "excluded_from_behavioral_denominators": True,
        "fixture_path": str(a.fixture.resolve()), "fixture_sha256": a.fixture_sha256, "prompt_condition": a.prompt_condition, "prompt": prompt["prompt"], "prompt_utf8_hex": prompt["prompt_utf8_hex"], "prompt_sha256": prompt["prompt_sha256"], "sampling_seed": a.sampling_seed,
        "lane_id": runtime["lane_id"], "simulator_pod_uid": runtime["simulator_pod_uid"], "simulator_gpu_uuid": runtime["simulator_gpu_uuid"], "policy_server_pod_uid": runtime["policy_server_pod_uid"], "policy_server_gpu_uuid": runtime["policy_server_gpu_uuid"], "server_port": runtime["server_port"], "server_process_identity": runtime["server_process_identity"], "server_lock_identity": runtime["server_lock_identity"],
        "actions": {"path": str(action_path.resolve()), "bytes": action_path.stat().st_size, "sha256": sha256_file(action_path)}, "runtime_manifest": {"path": str(a.runtime_manifest.resolve()), "bytes": a.runtime_manifest.stat().st_size, "sha256": a.runtime_manifest_sha256}}
    output = a.output_dir / "lane_response.json"; output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(output)}, sort_keys=True))
if __name__ == "__main__": main()
