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

Live-verified committed facts: the protocol validator was valid at handoff;
π0-FAST DROID's confirmation is complete and its wording grid is deferred;
the three WAM direct gates triggered direct-only confirmation. Efficient-WAM
pair03 is complete locally and **must not be rerun**. Work-laptop B200 facts
(hardware, drivers, namespace/PVC, egress, credentials, and installed tools)
are unknown until the gates below pass.

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

## Authorized queue: exactly 40 remaining cells

Every row is a single process invocation that emits two static direct-command
cells (LEFT and RIGHT), with viewport video. Do not pass many tasks/seeds to a
runner—repeated CLI values form a Cartesian product. No oracle, subtask coach,
prompt switching, or progress-conditioned instruction is allowed.

| Pair | Task | Env seed | Sampling seed | Efficient | Fast | LingBot |
| --- | --- | ---: | ---: | --- | --- | --- |
| 03 | `place_a2b_right` | 4300003 | 8403 | **completed; do not rerun** | run | run |
| 04 | `place_a2b_left` | 4300004 | 8404 | run | run | run |
| 05 | `place_a2b_right` | 4300005 | 8405 | run | run | run |
| 06 | `place_a2b_left` | 4300006 | 8406 | run | run | run |
| 07 | `place_a2b_right` | 4300007 | 8407 | run | run | run |
| 08 | `place_a2b_left` | 4300008 | 8408 | run | run | run |
| 09 | `place_a2b_right` | 4300009 | 8409 | run | run | run |

This is Efficient pair04–09 (12 cells), Fast pair03–09 (14), LingBot
pair03–09 (14): 40 total. Historical pair00–02 action traces are unavailable,
not zero. Preserve every valid model failure and every partial/invalid raw
attempt on the PVC.

For each row, copy the corresponding pair03 command in
`docs/VLA_WAM_CONTINUATION.md`, replace pair identifier/task/environment seed/
sampling seed together, and write outputs under a new immutable `$POD_RAW`
run directory. Keep the repeated `--requested-relation left/right` flags and
all frozen guidance, horizon, diffusion, and future-retention values. Always
wrap exactly one pair in `tools/native_process_group_thermal_guard.py --launch`
using the model's own runtime-intervention ledger and invalid-attempt ledger:

```text
runtime_interventions_efficient_wam_rt_robotwin.json   invalid_attempts_efficient_wam_rt_robotwin.json
runtime_interventions_fastwam_robotwin.json            invalid_attempts_fastwam_robotwin.json
runtime_interventions_lingbot_va_robotwin.json         invalid_attempts_lingbot_va_robotwin.json
```

The guard pauses at 87 C, resumes at 80 C, and holds at 90 C. A pause excludes
affected cells only from wall-latency aggregates; a hold or incomplete result
is `partial`/`technical_invalid`, not a behavioral failure. For LingBot inspect
the frozen wrapper's `--dry-run` output first, but never use its all-pairs
`--run` outside the guard.

## Compile, validate, sync, and stop

After each model's required pairs, copy only its compact model-specific ledgers
from the PVC into `artifacts/vla_wam_shared_v2/pilot/directional_confirmation/`
and run the three exact compiler commands in `VLA_WAM_CONTINUATION.md`, with
each `--input-root` pointed at that model's PVC raw root. Repeat the historical
pilot intervention ledger plus only that model's new ledger; never combine
model ledgers or pool DROID/RoboTwin success rates.

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
group, confirm ports 8000/5000 are free, retain all raw PVC output, and update
`continuation_state.json` with completed cells, invalid attempts, raw PVC
provenance, active job/process IDs, and the exact next command. Update
`VLA_WAM_CONTINUATION.md` only for a disclosed post-result decision. No WAM
wording sweep, π0-FAST wording grid, GR00T, or LingBot-VLA onboarding is
authorized until the three WAM confirmations are compiled.
