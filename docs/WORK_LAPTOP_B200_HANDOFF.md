# Work-laptop → Kubernetes B200 continuation handoff

This is the no-chat restart point for the VLA/WAM language-steerability study.
It intentionally does **not** invent a cluster, context, namespace, pod, PVC,
GPU index, container image, driver, Vulkan ICD, or egress status.

## Authoritative state and reading order

After this handoff is pushed, clone its exact study branch on the work Mac.
Replace only the remote URL if the repository is mirrored internally.

```bash
set -euo pipefail
export STEERABLE_REMOTE="${STEERABLE_REMOTE:?pushed steerable Git remote URL}"
export STUDY_BRANCH="${STUDY_BRANCH:-codex/wam-language-steerability}"
export MAC_STUDY="${MAC_STUDY:-$PWD/steerable}"
git clone --branch "$STUDY_BRANCH" "$STEERABLE_REMOTE" "$MAC_STUDY"
cd "$MAC_STUDY"
git status --short
git rev-parse HEAD
python3 tools/validate_vla_wam_v2_protocol.py
```

Read, in order, before modifying code, downloading assets, or launching a
model:

1. `AGENTS.md`
2. `docs/VLA_WAM_CONTINUATION.md`
3. `artifacts/vla_wam_shared_v2/continuation_state.json`
4. `docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md`
5. `artifacts/vla_wam_shared_v2/pilot/action_trace_instrumentation_amendment.json`
6. `handoff/repo_bundles/MANIFEST.json` and `handoff/repo_bundles/README.md`

Then read the four external readmes at their pinned checkouts:
`Efficient-WAM/experiments/robotwin_language_gate/README.md`,
`FastWAM/experiments/robotwin_language_gate/README.md`,
`lerobot-lingbot/experiments/lingbot_language_gate/README.md`, and
`EfficientWAM-RoboTwin/README.md`.

Live-verified committed facts: use the check count in the current committed
validation report; every
currently runnable bounded gate is complete; all 42 prospective three-WAM
pairs03–09 episodes are valid committed evidence and **must not be rerun**.
The completed six-cell gates are GR00T N1.7, Cosmos3 Edge, LingBot-VLA 4B,
Light-WAM, and DreamZero. DreamZero's six execution videos, all nine official
imagination decodes, and three paired imagination views are committed. π0-FAST
V2-A008 failed its fixed-observation prompt-sensitivity release gate: the two
LEFT requests repeated bit-identically but LEFT and RIGHT had action RMS 0.0.
It has zero behavioral episodes and no runnable cell under the frozen protocol.
π0.5 V2-A010 is complete as a separate current-stack six-cell result (LEFT
1/3, RIGHT 3/3; all three endpoint pairs aligned and action pairs distinct).
Its selected seed-8300 actual-rollout pair is committed; it is not recovered
historical v1 footage. Cosmos3 Nano Policy DROID V2-A011 is also complete as a
separate six-cell current-stack result (LEFT 3/3, RIGHT 3/3; all endpoint and
action pairs aligned/distinct; 37 decoded futures retained). Its selected
seed-8300 actual rollout and model prediction are labelled separately. LaWAM was withdrawn before inference by
`V2-A009` and has zero remaining cells. The historical 60-cell π0-FAST wording
queue remains separately blocked on missing exact OpenPI/RoboLab revisions and
must not be merged with V2-A008. Cosmos3 Edge base V2-A013 and Cosmos3 Super
base V2-A012/V2-A014 have completed their exact three-request interface probes.
Both were deterministic on repeat LEFT and prompt-sensitive between LEFT and
RIGHT, but neither released behavior. Edge execution is blocked by the exact
CuRobo mapping audit; Super used the image-only action-and-video route and had
no robot state or controller. Their gallery clips are predictions, not
rollouts. Edge base is not the completed Edge-Policy-DROID checkpoint.

## Restrict Kubernetes discovery to ali-owned resources

Do not use `--all-namespaces`, inspect other users' pods, or pick a shared B200
workload. First list only local kubeconfig aliases which explicitly identify
`ali` (this does not query a cluster):

```bash
kubectl config get-contexts -o name | rg -i '(^|[-_.])ali([-_.]|$)'
```

Choose a known ali-owned alias, namespace, and pod. If namespace ownership
cannot be established from your normal access path, ask its owner; do not
enumerate namespaces to guess it.

```bash
export KCTX="${KCTX:?one ali-owned context alias}"
export KNS="${KNS:?one confirmed ali-owned namespace}"
export POD="${POD:?one confirmed ali-owned B200 pod}"
kubectl --context "$KCTX" auth can-i get pods -n "$KNS"
kubectl --context "$KCTX" get pod "$POD" -n "$KNS" -o wide
kubectl --context "$KCTX" get pod "$POD" -n "$KNS" -o jsonpath='{range .spec.volumes[*]}{.name}{"\t"}{.persistentVolumeClaim.claimName}{"\n"}{end}'
kubectl --context "$KCTX" get pods -n "$KNS" -o wide
```

Confirm the pod's writable PVC mount from its spec, then set its in-container
path. Raw output, checkpoints, environments, and external checkouts belong on
this PVC, not the container filesystem or ordinary Git.

```bash
export PVC_ROOT="${PVC_ROOT:?confirmed writable in-container PVC mount}"
export POD_SRC="$PVC_ROOT/vla_wam/src"
export POD_EXT="$PVC_ROOT/vla_wam/external"
export POD_RAW="$PVC_ROOT/vla_wam/raw"
kubectl --context "$KCTX" -n "$KNS" exec "$POD" -- bash -lc 'set -euo pipefail; mkdir -p "$1" "$2" "$3"; df -h "$1" "$2" "$3"; findmnt -T "$1"' bash "$POD_SRC" "$POD_EXT" "$POD_RAW"
```

## B200, Vulkan/SAPIEN, PVC, and egress gates

Run these before a model environment or cell. Pick `GPU_INDEX` only after
checking the pod-local process list and run one model/simulator process group
at a time.

```bash
export GPU_INDEX="${GPU_INDEX:?idle pod-local GPU index}"
kubectl --context "$KCTX" -n "$KNS" exec "$POD" -- bash -lc '
  set -euo pipefail
  nvidia-smi -L; nvidia-smi
  test -c /dev/nvidia0
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu --format=csv,noheader
  test -r /usr/share/vulkan/icd.d/nvidia_icd.json
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary
  curl --fail --silent --show-error --max-time 20 --head https://github.com/
  curl --fail --silent --show-error --max-time 20 --head https://huggingface.co/
'
```

For every model environment, prove the actual headless renderer, not merely an
import. Use its runner's `env -u DISPLAY`, actual readable NVIDIA ICD, and
model-specific `PYTHONPATH`; `sapien.Engine()`, `SapienRenderer()`, and scene
creation must succeed. Then run the model's `--help` or LingBot `--dry-run`
without loading a policy. A failed GPU, Vulkan, SAPIEN, CuRobo, PVC, auth, or
egress gate is a preserved technical event, not model evidence.

## Source and exact external revisions

With GitHub egress passing, clone the pushed study branch to the PVC; otherwise transfer
the already-cloned work-Mac checkout through the approved ali-owned pod path
without losing `.git` provenance.

```bash
kubectl --context "$KCTX" -n "$KNS" exec "$POD" -- bash -lc 'set -euo pipefail; git clone --branch "$1" "$2" "$3/steerable"' bash "$STUDY_BRANCH" "$STEERABLE_REMOTE" "$POD_SRC"
```

Inside the pod set `STUDY_ROOT=$POD_SRC/steerable`,
`BUNDLES=$STUDY_ROOT/handoff/repo_bundles`, and `EXT=$POD_EXT`. Follow
`handoff/repo_bundles/README.md` verbatim: it clones official repositories,
validates each incremental bundle, and checks out exactly:

```text
EfficientWAM-RoboTwin  0bd8e76fde3afcffa4b30a3e3e8f92a206aa66cc
Efficient-WAM          b0b6cfabcbd68d18888866e958c677ce640f0412
FastWAM                068d3fd70c89df3726c09893f47b75a624b20c02
lerobot-lingbot        d42efbc04e502057dab4b18bb14770cc48e85131
```

Never force a bundle if its official prerequisite is missing. The final three
model commits include the frozen prospective action-trace instrumentation:
pair03–09 `result.json` must contain action-trace `path`, `sha256`, `count`,
and `shape`.

Fetch and record resolved snapshot hashes/byte counts for these exact documented
asset IDs on the PVC: `jiajun0613/Efficient-WAM_RoboTwin`,
`Wan-AI/Wan2.2-TI2V-5B`, `google/umt5-xxl`, `yuanty/fastwam` (the
`robotwin_uncond_3cam_384.pt` and matching dataset-stats files),
`Wan-AI/Wan2.1-T2V-1.3B`, `lerobot/lingbot_va_robotwin`,
`robbyant/lingbot-va-posttrain-robotwin` restricted to `vae/**`,
`text_encoder/**`, `tokenizer/**`, and RoboTwin dataset
`TianxingChen/RoboTwin2.0` restricted to `background_texture.zip`,
`embodiments.zip`, `objects.zip`. Run FastWAM's checked-in ActionDiT
preprocessing command after its Wan asset is present. Use the exact installation
and download commands in the four readmes; keep their incompatible Python/Torch
environments separate.

## Current-stack status: completed gates and closed base probes

There is no runnable WAM confirmation cell left. Efficient-WAM-RT,
FastWAM, and LingBot-VA pairs03–09 are complete; preserve their raw PVC output
and committed compact slices without rerunning them.

| Priority | Queue | Cells | Status | Next gate |
| ---: | --- | ---: | --- | --- |
| 0 | π0-FAST V2-A008 current-stack three-wording replication | 0 | release gate failed; zero behavioral cells | do not run under the frozen protocol; compact probe records identical LEFT/RIGHT actions |
| 1 | π0.5 V2-A010 current-stack direct/media gate | 0 | complete: 6/6 valid; LEFT 1/3, RIGHT 3/3 | retain the PVC raw evidence and selected seed-8300 actual-rollout pair; do not rerun or call it historical v1 media |
| 2 | Cosmos3 Nano Policy DROID V2-A011 | 0 | complete: 6/6 valid; LEFT 3/3, RIGHT 3/3 | retain raw PVC evidence and the bounded seed-8300 actual-versus-prediction pair; do not rerun |
| 3 | Cosmos3 Super base V2-A012/V2-A014 | 0 released | complete: 3/3 image-only interface requests; repeat deterministic, RIGHT distinct | retain prediction media; no behavior, controller, or rerun |
| 4 | Cosmos3 Edge base V2-A013 | 0 released | complete: 3/3 interface requests; repeat deterministic, RIGHT distinct; mapping blocked | retain prediction media and CuRobo rejection; no behavior or rerun |

The unavailable historical π0-FAST queue remains blocked on OpenPI
`9e46d3aea26417bfb564227734b95d010aa827e5` and RoboLab
`11142d4319e44401e0464866bb5fedf7ec8a8927`. Recovering those objects would
restore the historical queue. Their absence did not prevent the separate
V2-A008 release probe, but its prompt-sensitivity failure now blocks V2-A008
behavioral inference under the frozen protocol. LaWAM must not be resumed.
The two base-model probes are complete and closed with zero behavioral cells.
Their selected generated videos may appear only as interface predictions with
actual rollout explicitly unavailable.

## Compile, validate, sync, and stop

After an active model's release gates pass, follow its frozen resume section in
`VLA_WAM_CONTINUATION.md`. Keep model-specific ledgers separate and
never combine DROID and RoboTwin success rates.

Then run:

```bash
cd "$STUDY_ROOT"
python3 tools/validate_vla_wam_v2_protocol.py --write-report artifacts/vla_wam_shared_v2/protocol_validation.json
python3 tools/select_vla_wam_v2_media.py
python3 tools/render_vla_wam_v2_reader_figures.py
python3 tools/render_vla_wam_v2_robotwin_videos.py
python3 tools/render_vla_wam_v2_droid_videos.py
python3 tools/build_vla_wam_v2_pilot_grid.py
git diff --check
```

The compiler must reject missing prospective action trace metadata. Generated
future evidence is scored only when its released interface provides a decodable
future; decoded video, latent-only future, and action-only interfaces remain
separate. Commit and sync only compact evidence, hashes, manifests, figures,
and renderers; never commit checkpoints, environments, PVC raw collections,
unbounded MP4s, or action traces.

Before stopping, stop only this study's policy/server/simulator/thermal process
group, retain all raw PVC output, and update
`continuation_state.json` with completed cells, invalid attempts, raw PVC
provenance, active job/process IDs, and the exact next command. Update
`VLA_WAM_CONTINUATION.md` only for a disclosed post-result decision. Do not
rerun a completed cell or start an unlisted model/wording expansion.
