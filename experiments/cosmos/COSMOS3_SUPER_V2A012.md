# Cosmos3-Super base DROID feasibility — V2-A012

`V2-A012` freezes a conditional six-cell DROID gate for the public base model
[`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) at revision
`e0262be9d8f7586bc24c069a2aed2b665bdff266`. It does not authorize a model
request or behavioral rollout yet.

The base model is 64B and its exact weight index reports 129,230,007,264 bytes.
It exposes action and visual generation, but NVIDIA publishes DROID policy
checkpoints only for Cosmos3 Nano and Cosmos3 Edge. The official DROID action
documentation describes 10D end-effector pose plus gripper actions, while this
RoboLab study executes 8D `joint_pos`. Do not call Super a DROID policy, and do
not invent a conversion.

The authoritative artifacts are:

- `artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_super_droid_amendment.json`
- `artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_droid_v2a012_registry.json`
- `artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_v2a012_hf_snapshot.json`

## Topology gate

The preferred server is the ali-owned four-B200 pod defined in
`handoff/k8s/cosmos3-super-b200-4gpu-256gi-ali.yaml`. It is currently Pending
until a node has four free B200 GPUs. Query only that named pod; do not inspect
other users' workloads.

```bash
kubectl --context "$KCTX" -n "$KNS" get pod cosmos3-super-b200-4gpu-256gi-ali -o wide
```

If it remains pending, a strictly conditional, load-only fallback may use only
GPU 1 and GPU 2 of the existing ali-owned `lerobot-b200-4gpu-1-ali` pod, at
tensor parallelism two. GPU 0 is protected and GPU 3 belongs to the separate
Nano server. Check those exact devices and the ali PVC immediately before any
load; a failed load or OOM is infrastructure evidence and releases no request.

```bash
kubectl --context "$KCTX" -n "$KNS" exec lerobot-b200-4gpu-1-ali -- bash -lc '
  set -euo pipefail
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
  df -B1 /data/users/ali/vla_wam
'
```

No topology is sufficient by itself. The registry requires at least 200,000,000,000
free PVC bytes, all selected GPUs idle, and an exact model-load memory record.

## Asset and software freeze

The metadata builder contacts only the frozen Hugging Face revision; it does
not download model weights or load a model.

```bash
python3 tools/build_cosmos3_super_checkpoint_manifest.py \
  --output artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_v2a012_hf_snapshot.json
```

Only after its hash is unchanged, capacity and credentials are recorded, and a
dedicated storage path is confirmed, download the exact checkpoint to the ali
PVC and hash it before any server starts:

```bash
/data/users/ali/vla_wam/envs/hf-tools/bin/hf download \
  nvidia/Cosmos3-Super \
  --revision e0262be9d8f7586bc24c069a2aed2b665bdff266 \
  --local-dir /data/users/ali/vla_wam/checkpoints/cosmos3_super_base

python3 tools/finalize_cosmos3_super_registry.py \
  --registry artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_droid_v2a012_registry.json \
  --source-snapshot artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_v2a012_hf_snapshot.json \
  --checkpoint /data/users/ali/vla_wam/checkpoints/cosmos3_super_base \
  --output artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_droid_v2a012_registry.json
```

Restore separate source checkouts at NVIDIA Cosmos
`e494d734022ab0610061cdf57fa24c843e18767e` and vLLM-Omni
`900a7f0813d0482811b0e4dfd3cf7deabbe2429f`. Before launching, record those
Git heads, installed package versions, and either the immutable image digest or
the reproducible environment lock. The mutable `vllm/vllm-omni:cosmos3` tag is
not adequate provenance.

## Release decision

The first inference, when all earlier gates pass, is a no-environment,
fixed-observation diagnostic: `LEFT`, exact-repeat `LEFT`, and `RIGHT` with
sampling seed 8300 and byte-identical image/state inputs. It must preserve
returned actions and decoded futures. Repeated LEFT actions and futures must be
bit-identical; LEFT and RIGHT actions and futures must differ.

There are only two mutually exclusive execution branches:

| Returned contract | May execute? | Required reporting |
| --- | --- | --- |
| Official, normalized `[T,8]` `joint_pos` mapping | Yes, after the fixed gate | Native direct-interface result |
| Official DROID 10D end-effector-plus-gripper mapping | Only after a separately frozen deterministic CuRobo IK contract | Derived-control CuRobo IK intervention |
| Video-only, latent-only, undocumented, or silently transformed action output | No | Retain as interface evidence; no rollout |

The derived-control branch is user-authorized but is not native policy
execution. Before it can command RoboLab, freeze the exact 10D layout, units,
frame, normalization, absolute/delta semantics, robot URDF hash, joint order,
gripper mapping, CuRobo version/commit, collision configuration, solver seeds,
and tolerances. The controller receives only the model action plus current robot
state: never task progress, a success signal, oracle action, subtask coach, or
switched prompt. An IK, collision, limit, or stale-state rejection sends no
action; retain the raw action and rejection record in the controller ledger.

Only then may seeds 8300–8302 run static direct LEFT/RIGHT prompts. Every valid
cell needs viewport MP4, HDF5, exact executed actions, returned action chunk,
and decoded future. Publish **ACTUAL ROLLOUT** and **IMAGINED FUTURE** adjacent,
with the same seed, relation, prompt, model-action hash, and executed-action
hash. Never substitute one for the other, and never pool either Super branch
with Nano, Edge, Cosmos-Reason2, a VLA, or RoboTwin.
