"""Render finite, non-preempting MAIN P admission candidates with two owners."""
import argparse
import copy
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--a40-nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    simulator_base = yaml.safe_load(Path("handoff/k8s/sgw01-ali-family-native-smoke-20260923bn.yaml").read_text())
    policy_base = yaml.safe_load(Path("handoff/k8s/sgw01-ali-n3-fixedinput-20260923di.yaml").read_text())
    nodes = [
        node["metadata"]["name"] for node in json.loads(args.a40_nodes.read_bytes())["items"]
        if not node["spec"].get("taints") and not node["spec"].get("unschedulable")
        and any(c["type"] == "Ready" and c["status"] == "True" for c in node["status"]["conditions"])
    ]
    source = "/data/users/ali/sgw-01/source/main-p-20260923dm"
    root = "/data/users/ali/sgw-01/main/n3-lat-p-20260923dm"
    script = r'''
set -euo pipefail
mkdir -p "$ROOT/admissions"
OUT="$ROOT/admissions/$ROLE-$INDEX"
mkdir "$OUT"
export OUT
exec > >(tee "$OUT/worker.log") 2>&1
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
export XDG_CACHE_HOME="$OUT/cache" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/torchinductor"
export WARP_CACHE_PATH="$OUT/cache/warp" MPLCONFIGDIR="$OUT/cache/matplotlib"
export SGW01_NANO_OUTPUT_DIR="$OUT/native-loader"
mkdir -p "$HOME/.cache" "$XDG_CACHE_HOME" "$WARP_CACHE_PATH" "$MPLCONFIGDIR"
export PYTHONPATH="$SOURCE:$NATIVE_SOURCE"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$("$PY" -m experiments.workshops.spatial_grounding_v1.gpu_idle_probe --expected-count 1 --output "$OUT/idle.json")
exec 9>"/data/users/ali/sgw-01/locks/gpu-$CUDA_VISIBLE_DEVICES.lock"
flock --nonblock 9
mkdir "$ROOT/claims/$ROLE"
"$PY" - <<'PY'
from pathlib import Path
import json,os,shutil,time
from experiments.workshops.spatial_grounding_v1.family_campaign_executor import _fsync_json
root=Path(os.environ["ROOT"])
free=shutil.disk_usage(root).free
if free < 4144643440640:
    raise RuntimeError("persistent recording reserve unavailable")
_fsync_json(root/"claims"/os.environ["ROLE"]/"owner.json",{
    "role":os.environ["ROLE"],"pod":os.environ["POD_NAME"],"pod_uid":os.environ["POD_UID"],
    "job_uid":os.environ["JOB_UID"],"source_commit":os.environ["SOURCE_COMMIT"],
    "started_at_unix":time.time(),"free_bytes":free,
    "idle":json.loads((Path(os.environ["OUT"])/"idle.json").read_text()),
    "scope":"six original MAIN N3 LAT P cells only","automatic_retries":0,
})
PY
cd "$NATIVE_SOURCE"
"$PY" -u -m experiments.workshops.spatial_grounding_v1.main_p_execution --role "$ROLE" --plan "$ROOT/execution-plan.json" --root "$ROOT"
'''
    items = []
    placements = {
        "simulator": sorted(nodes)[:12],
        "policy": ["dcwipphai0061.edc.nam.gm.com", "dcwipphai0063.edc.nam.gm.com"],
    }
    for role, targets in placements.items():
        for index, node in enumerate(targets):
            name = f"sgw01-ali-main-p-20260923dm-{role}-{index:02d}"
            base = simulator_base if role == "simulator" else policy_base
            job = {
                "apiVersion": "batch/v1", "kind": "Job",
                "metadata": {"name": name, "namespace": "211247-prod", "labels": {
                    "owner": "ali", "app.kubernetes.io/name": "sgw-01", "sgw-main-batch": "20260923dm",
                }},
                "spec": {"activeDeadlineSeconds": 14400, "backoffLimit": 0,
                         "template": copy.deepcopy(base["spec"]["template"])},
            }
            job["spec"]["template"]["metadata"]["labels"].update(job["metadata"]["labels"])
            spec = job["spec"]["template"]["spec"]
            for key in ("affinity", "nodeSelector", "activeDeadlineSeconds", "tolerations"):
                spec.pop(key, None)
            spec.update(nodeName=node, schedulerName=name, priority=0)
            container = spec["containers"][0]
            container["command"] = ["/bin/bash", "-ec"]
            container["args"] = [script]
            env = {entry["name"]: entry for entry in container["env"]}
            for key in ("JOB_COMPLETION_INDEX", "REGISTRATION", "OUT", "CLAIM", "AUX"):
                env.pop(key, None)
            native = ("/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241" if role == "simulator"
                      else "/data/users/ali/vla_wam/external/cosmos-framework-411d25b")
            python = ("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python" if role == "simulator"
                      else "/data/users/ali/vla_wam/envs/cosmos-nano-411d25b-v3-exact/bin/python")
            for key, value in {
                "ROOT": root, "SOURCE": source, "SOURCE_COMMIT": args.source_commit,
                "ROLE": role, "INDEX": str(index), "NATIVE_SOURCE": native, "PY": python,
                "USER": "ali", "LOGNAME": "ali", "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "SGW01_NANO_SOURCE_ROOT": "/data/users/ali/vla_wam/external/cosmos-framework-411d25b",
                "SGW01_NANO_CHECKPOINT_PATH": "/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid",
                "HF_HOME": "/data/users/ali/vla_wam/cache/huggingface-cosmos",
                "UV_CACHE_DIR": "/data/users/ali/vla_wam/cache/uv-cosmos",
            }.items():
                env[key] = {"name": key, "value": value}
            env["JOB_UID"] = {"name": "JOB_UID", "valueFrom": {"fieldRef": {
                "fieldPath": "metadata.labels['batch.kubernetes.io/controller-uid']",
            }}}
            container["env"] = list(env.values())
            items.append(job)
    with args.output.open("x") as stream:
        json.dump({"apiVersion": "v1", "kind": "List", "items": items}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"admission_candidates": len(items), "maximum_simulator_owners": 1,
                      "maximum_policy_owners": 1, "main_episodes": 6}))


if __name__ == "__main__":
    main()
