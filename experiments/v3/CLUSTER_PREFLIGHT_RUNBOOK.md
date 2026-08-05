# V3 cluster preflight (read-only, fail-closed)

This runbook prepares a **future disclosed v3 expansion**.  It does not amend
the frozen v2 protocol, authorize a rerun, or reserve compute.  The v2 DROID
and RoboTwin results remain separate and all completed cells remain protected.

## Scope and safety

The preflight accepts one explicit context, namespace, pod, PVC path, and
study-root path. It fetches only that named pod. It does not list namespaces or
pods, and it never creates, scales, deletes, or launches Kubernetes work.

Before any `exec`, it requires either an `ali` **token** in the supplied pod
name (delimited by `-`, `_`, or `.`) or an exact `ali` value in a tokenized
`owner`/`user` pod label. Substring matches such as `malik` are rejected. A
path is accepted only when the named pod spec proves that it lies beneath a
PVC-backed mount. It prints no tokens, credentials, or environment values.

The cluster was unreachable when this runbook was written. The following is a
known target from the committed v2 handoff, **not live verification**:

```bash
export KCTX=prod-dcwi-warrenq1-vmkub007
export KNS=211247-prod
export POD=lerobot-b200-4gpu-1-ali
kubectl --context "$KCTX" -n "$KNS" get pod "$POD" -o wide
```

If the command does not establish that this is still an ali-owned pod, stop.
Do not search another namespace or select another user's workload.

## Required inputs after connectivity returns

1. Confirm the pod and its writable PVC mount using the command above and its
   pod spec. Set `PVC_ROOT` only to an observed in-container PVC path.
2. Confirm the checkout location on that mount and set `STUDY_ROOT` only to
   the observed Git checkout.
3. Confirm the actual NVIDIA Vulkan ICD path inside the named pod and set
   `VULKAN_ICD` to that absolute in-pod path. Live discovery on the known B200
   pod previously found `/etc/vulkan/icd.d/nvidia_icd.json`; this is a hint,
   **not a default or current verification**.
4. Supply a model-specific, no-policy-load SAPIEN command that uses its real
   Python environment and constructs `sapien.Engine()`, `SapienRenderer()`,
   and a scene **and renders/captures a frame**. An import-only test is not an
   execution gate. The generic preflight does not guess a Python environment.
5. Supply `CREDENTIAL_GATE_CMD`, a model-specific, non-secret check that exits
   successfully only when either its required local snapshot is complete and
   hash-pinned or its authentication is valid. Its output is suppressed by the
   preflight, so it must not depend on emitting a token or a credential value.
   The current B200 Hugging Face CLI reports that it is not logged in; this is
   not a blocker when a model's complete verified local snapshot satisfies its
   staging gate.

Example invocation (the two path values and the SAPIEN hook are deliberately
placeholders until they are verified):

```bash
tools/vla_wam_v3_cluster_preflight.sh \
  --context "$KCTX" --namespace "$KNS" --pod "$POD" \
  --pvc-root '<verified in-container PVC root>' \
  --study-root '<verified checkout under that PVC root>' \
  --vulkan-icd '<verified absolute in-container NVIDIA ICD path>' \
  --credential-gate-cmd '<verified non-secret auth or local hash-pinned snapshot check>' \
  --sapien-gate-cmd '<verified model-environment SAPIEN engine/renderer/scene render-and-capture check>'
```

The preflight verifies the branch
`codex/wam-language-steerability`, a clean worktree, **both frozen v2 and v3
validators before any GPU inspection**, PVC capacity and persistence, GPU
identity/free memory/current compute processes, the explicit readable NVIDIA
Vulkan ICD, a real SAPIEN render-and-capture gate, and GitHub/Hugging Face
egress. After egress and before GPU/render checks, a silent per-model
credential-or-local-snapshot gate must pass. `vulkaninfo` is not required: the
actual SAPIEN capture is the renderer-execution check. A failure is an
infrastructure record, not behavioral evidence.

## After a passing preflight

A passing preflight is still not permission to run cells. Before execution,
add a disclosed v3 amendment with:

- a prespecified seed list shared by each matched LEFT/RIGHT pair;
- raw JSONL episode records and a separate infrastructure-invalid stream;
- fixed success predicates and exact static prompts;
- failure categories and continuous trajectory fields;
- no pooling between DROID and RoboTwin.

Only then choose idle GPUs after reviewing the process list. Keep raw outputs,
checkpoints, and environments under the verified PVC rather than Git.
