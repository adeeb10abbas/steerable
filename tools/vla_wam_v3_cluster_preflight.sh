#!/usr/bin/env bash
# Read-only, fail-closed Kubernetes preflight for a disclosed VLA/WAM v3 study.
# It never enumerates resources and never creates, scales, deletes, or launches work.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/vla_wam_v3_cluster_preflight.sh \
    --context KCTX --namespace KNS --pod POD --pvc-root PVC_ROOT \
    --study-root STUDY_ROOT --vulkan-icd ICD_PATH \
    --credential-gate-cmd 'MODEL-SPECIFIC-NON-SECRET-CHECK' \
    --sapien-gate-cmd 'MODEL-SPECIFIC-COMMAND' \
    [--study-branch codex/wam-language-steerability]

All target arguments are mandatory.  PVC_ROOT must be a writable path inside a
PVC mounted by the named pod.  SAPIEN_GATE_CMD must create an actual headless
SAPIEN engine, renderer, and scene and capture a frame in the intended model
environment; it is executed only after ownership, PVC, GPU, egress, and source
gates pass.
VULKAN_ICD must be a live-verified absolute path inside the named pod.
CREDENTIAL_GATE_CMD is run with output suppressed: it must return success only
when a local hash-pinned snapshot is complete or non-secret model auth passes.
EOF
}

KCTX=''
KNS=''
POD=''
PVC_ROOT=''
STUDY_ROOT=''
VULKAN_ICD=''
CREDENTIAL_GATE_CMD=''
SAPIEN_GATE_CMD=''
STUDY_BRANCH='codex/wam-language-steerability'

while (($#)); do
  case "$1" in
    --context) KCTX=${2:?missing value for --context}; shift 2 ;;
    --namespace) KNS=${2:?missing value for --namespace}; shift 2 ;;
    --pod) POD=${2:?missing value for --pod}; shift 2 ;;
    --pvc-root) PVC_ROOT=${2:?missing value for --pvc-root}; shift 2 ;;
    --study-root) STUDY_ROOT=${2:?missing value for --study-root}; shift 2 ;;
    --vulkan-icd) VULKAN_ICD=${2:?missing value for --vulkan-icd}; shift 2 ;;
    --credential-gate-cmd) CREDENTIAL_GATE_CMD=${2:?missing value for --credential-gate-cmd}; shift 2 ;;
    --sapien-gate-cmd) SAPIEN_GATE_CMD=${2:?missing value for --sapien-gate-cmd}; shift 2 ;;
    --study-branch) STUDY_BRANCH=${2:?missing value for --study-branch}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in KCTX KNS POD PVC_ROOT STUDY_ROOT VULKAN_ICD CREDENTIAL_GATE_CMD SAPIEN_GATE_CMD; do
  if [[ -z ${!required} ]]; then
    printf 'ERROR: %s is required.\n' "$required" >&2
    usage >&2
    exit 2
  fi
done

[[ "$VULKAN_ICD" == /* ]] || { printf 'ERROR: VULKAN_ICD must be an absolute in-pod path.\n' >&2; exit 2; }

say() { printf '\n== %s ==\n' "$1"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }
kget() { kubectl --context "$KCTX" -n "$KNS" "$@"; }
kexec() { kubectl --context "$KCTX" -n "$KNS" exec "$POD" -- "$@"; }

has_ali_owner_label() {
  local entry key value
  IFS=';' read -r -a entries <<< "$1"
  for entry in "${entries[@]}"; do
    [[ "$entry" == *=* ]] || continue
    key=${entry%%=*}
    value=${entry#*=}
    # Accept only an exact ali value under a tokenized owner/user label key.
    if [[ "$value" == 'ali' && "$key" =~ (^|[./_-])(owner|user)([./_-]|$) ]]; then
      return 0
    fi
  done
  return 1
}

say 'named-pod authorization and ownership'
kget auth can-i get pods || fail 'cannot read pods in the supplied namespace'
# The only cluster object fetched is the explicitly supplied pod name.
pod_json=$(kget get pod "$POD" -o json) || fail 'named pod was not readable'
pod_identity=$(python3 -c '
import json, sys
pod = json.load(sys.stdin)
name = pod.get("metadata", {}).get("name", "")
labels = pod.get("metadata", {}).get("labels", {})
print(name)
print(";".join(f"{key}={value}" for key, value in sorted(labels.items())))
' <<< "$pod_json") || fail 'named pod identity could not be parsed'
actual_pod_name=${pod_identity%%$'\n'*}
pod_labels=${pod_identity#*$'\n'}
[[ "$actual_pod_name" == "$POD" ]] || fail 'named pod response did not match the supplied pod name'
if [[ ! "$POD" =~ (^|[-_.])ali([-_.]|$) ]] && ! has_ali_owner_label "$pod_labels"; then
  fail 'named pod lacks an ali token and labels lack an exact ali owner/user value'
fi
printf 'PASS: named ali-owned pod %s is readable in %s.\n' "$POD" "$KNS"

say 'PVC mount and persistent path'
volume_rows=$(kget get pod "$POD" -o jsonpath='{range .spec.volumes[*]}{.name}{"\t"}{.persistentVolumeClaim.claimName}{"\n"}{end}')
mount_rows=$(kget get pod "$POD" -o jsonpath='{range .spec.containers[*].volumeMounts[*]}{.name}{"\t"}{.mountPath}{"\n"}{end}')
pvc_volume=''
while IFS=$'\t' read -r volume claim; do
  [[ -n ${volume:-} && -n ${claim:-} ]] || continue
  while IFS=$'\t' read -r mounted_volume mount_path; do
    [[ "$mounted_volume" == "$volume" ]] || continue
    if [[ "$PVC_ROOT" == "$mount_path" || "$PVC_ROOT" == "$mount_path"/* ]]; then
      pvc_volume=$volume
      printf 'PASS: PVC claim %s is mounted at %s (requested root: %s).\n' "$claim" "$mount_path" "$PVC_ROOT"
      break 2
    fi
  done <<< "$mount_rows"
done <<< "$volume_rows"
[[ -n "$pvc_volume" ]] || fail 'PVC_ROOT is not beneath a PVC-backed mount in the named pod spec'

kexec bash -lc '
  set -euo pipefail
  root=$1
  test -d "$root"
  test -w "$root"
  df -h "$root"
  findmnt -T "$root"
' bash "$PVC_ROOT" || fail 'PVC path is not readable, writable, and mounted in the named pod'

say 'study source, branch, clean tree, and frozen validators'
kexec bash -lc '
  set -euo pipefail
  root=$1 branch=$2
  test -d "$root/.git"
  actual=$(git -C "$root" branch --show-current)
  test "$actual" = "$branch"
  test -z "$(git -C "$root" status --short)"
  python3 "$root/tools/validate_vla_wam_v2_protocol.py"
  python3 "$root/tools/validate_vla_wam_v3_protocol.py"
  printf "PASS: study source is on the requested clean branch and both protocols validate.\n"
' bash "$STUDY_ROOT" "$STUDY_BRANCH" || fail 'study source branch, cleanliness, or protocol validator gate failed'

say 'explicit Vulkan ICD path and external egress'
kexec bash -lc '
  set -euo pipefail
  icd=$1
  test -r "$icd"
  curl --fail --silent --show-error --max-time 20 --head https://github.com/ >/dev/null
  curl --fail --silent --show-error --max-time 20 --head https://huggingface.co/ >/dev/null
  printf "PASS: explicit readable Vulkan ICD and GitHub/Hugging Face egress are available.\n"
' bash "$VULKAN_ICD" || fail 'explicit Vulkan ICD path or egress gate failed'

say 'model credential or local snapshot gate'
# Deliberately suppress hook output so this generic preflight never exposes a
# token. The per-model hook may instead attest complete hash-pinned local files.
kexec bash -lc '
  set -euo pipefail
  gate=$1
  bash -lc "$gate" >/dev/null 2>&1
  printf "PASS: credential or local snapshot gate passed.\n"
' bash "$CREDENTIAL_GATE_CMD" || fail 'credential or local snapshot gate failed'

say 'GPU availability and current owners'
kexec bash -lc '
  set -euo pipefail
  # Kubernetes may expose physical /dev/nvidiaN while CUDA renumbers the
  # allocated device to logical GPU 0.  Do not assume that physical N is zero.
  test -c /dev/nvidiactl
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
' || fail 'GPU query failed in the named pod'

say 'model-specific headless SAPIEN gate'
# Intentionally supplied by the model runbook, so this script never guesses an
# environment. It must render and capture a frame, rather than merely import SAPIEN.
kexec bash -lc '
  set -euo pipefail
  gate=$1 icd=$2
  test -r "$icd"
  env -u DISPLAY VK_ICD_FILENAMES="$icd" bash -lc "$gate"
' bash "$SAPIEN_GATE_CMD" "$VULKAN_ICD" || fail 'the supplied SAPIEN headless render-and-capture gate failed'

printf '\nPASS: v3 preflight completed. This did not reserve GPUs or authorize inference.\n'
