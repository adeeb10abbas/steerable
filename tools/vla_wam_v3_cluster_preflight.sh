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
    --sapien-gate-cmd 'MODEL-SPECIFIC-COMMAND' \
    [--study-branch codex/wam-language-steerability]

All target arguments are mandatory.  PVC_ROOT must be a writable path inside a
PVC mounted by the named pod.  SAPIEN_GATE_CMD must create an actual headless
SAPIEN engine, renderer, and scene in the intended model environment; it is
executed only after ownership, PVC, GPU, egress, and source gates pass.
VULKAN_ICD must be a live-verified absolute path inside the named pod.
EOF
}

KCTX=''
KNS=''
POD=''
PVC_ROOT=''
STUDY_ROOT=''
VULKAN_ICD=''
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
    --sapien-gate-cmd) SAPIEN_GATE_CMD=${2:?missing value for --sapien-gate-cmd}; shift 2 ;;
    --study-branch) STUDY_BRANCH=${2:?missing value for --study-branch}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in KCTX KNS POD PVC_ROOT STUDY_ROOT VULKAN_ICD SAPIEN_GATE_CMD; do
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
pod_labels=$(kget get pod "$POD" -o jsonpath='{range $k,$v := .metadata.labels}{$k}={$v}{";"}{end}') \
  || fail 'named pod was not readable'
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

say 'GPU availability and current owners'
kexec bash -lc '
  set -euo pipefail
  test -c /dev/nvidia0
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
' || fail 'GPU query failed in the named pod'

say 'Vulkan and external egress'
kexec bash -lc '
  set -euo pipefail
  icd=$1
  test -r "$icd"
  VK_ICD_FILENAMES="$icd" vulkaninfo --summary
  curl --fail --silent --show-error --max-time 20 --head https://github.com/ >/dev/null
  curl --fail --silent --show-error --max-time 20 --head https://huggingface.co/ >/dev/null
  printf "PASS: Vulkan ICD and GitHub/Hugging Face egress are reachable.\n"
' bash "$VULKAN_ICD" || fail 'Vulkan or egress gate failed'

say 'model-specific headless SAPIEN gate'
# Intentionally supplied by the model runbook, so this script never guesses an environment.
kexec bash -lc '
  set -euo pipefail
  gate=$1 icd=$2
  test -r "$icd"
  env -u DISPLAY VK_ICD_FILENAMES="$icd" bash -lc "$gate"
' bash "$SAPIEN_GATE_CMD" "$VULKAN_ICD" || fail 'the supplied SAPIEN headless-renderer gate failed'

printf '\nPASS: v3 preflight completed. This did not reserve GPUs or authorize inference.\n'
