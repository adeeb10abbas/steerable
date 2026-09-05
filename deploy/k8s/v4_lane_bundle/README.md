# V4 Kubernetes lane bundle (prospective)

This package renders one simulator lane and one policy server as **fresh immutable
Kubernetes `Job` objects** for the online-correction V4 campaign. Each container
starts the audited lane entrypoint as PID 1, runs synchronous fail-closed startup
preflight, then replaces itself with the exact experiment process. It does **not**
use `kubectl exec`, background servers, reusable Pods, or `sleep infinity`.

This bundle is **prospective infrastructure design**, not scientific authorization
and not a claim that any V4 episode has run or that A40/A100/B200/H100 strata are
qualified. A passing startup preflight proves only that the declared runtime can
execute CUDA, render a frame, import the pinned stack, encode/decode video, read
bound checkpoints/inputs, and write to the output parent.

## Inputs and rendering

Render from a complete immutable launch specification (`spec.example.json` schema
`vla-wam-v4-k8s-lane-render-spec-v1`). The specification must bind explicitly:

- `kube_context`, namespace, lane ID, attempt ID, and a launch/config hash in every
  object name and selector;
- content-addressed container image digest;
- **qualified GPU class per role** via `gpu_product` (nodeSelector) and
  `expected_gpu_name` (nvidia-smi display name)—never hardcoded in the renderer;
- PVC output parent (not an episode directory);
- absolute Python, ffmpeg, probe, checkpoint, and experiment executable paths;
- exact policy and simulator argv arrays;
- checkpoint and launch-config SHA-256 digests;
- every launch-critical file as `{path, bytes, sha256}`;
- policy port, private Service identity, readiness contract, preStop behavior,
  and simulator-side policy-wait parameters.

Unlike the historical V3 bundle, V4 does **not** substitute GPU labels. Configure
A40/A100/B200/H100 (or any authorized product string) in the spec; changing
hardware stratum requires a new spec and fresh qualification, not a label edit.

The rendered launch ConfigMap is `immutable: true`. Do not hand-edit rendered YAML
or transcribe argv/hashes into an interactive command. Re-render under a new
attempt/config identity when any input changes.

Before cluster creation, run local checks (dry-run safe—no cluster mutation):

```bash
python3 tools/render_v4_k8s_lane_bundle.py --help
python3 tools/validate_v4_k8s_lane_bundle.py --help
python3 tools/render_v4_k8s_lane_bundle.py \
  --spec /absolute/path/to/render-spec.json \
  --output-root /absolute/path/to/rendered-bundle
python3 tools/validate_v4_k8s_lane_bundle.py \
  --root /absolute/path/to/rendered-bundle \
  --spec /absolute/path/to/the-original-render-spec.json
kubectl create --dry-run=client -k /absolute/path/to/rendered-bundle -o yaml \
  > /absolute/path/to/rendered-bundle/client-decoded.yaml
```

The validator checks Job shape, configurable GPU class, cache isolation, ConfigMap
immutability, PID-1 entrypoint, private Service, readiness launch-config binding,
single-container policy port isolation, qualification/behavioral argv contracts,
unique lane/attempt/config labels, create-only dispatch, and evidence contracts. Client decoding is inspection only.

## Creating a lane

Use the bound cluster context and namespace from the rendered ConfigMap. **Create**
new immutable objects with `kubectl create`; **do not** use `kubectl apply`, because
an existing Job, Service, or ConfigMap must fail closed rather than be silently
reused or patched. The renderer refuses to overwrite an existing bundle directory;
dispatch must therefore target a fresh render output or a never-before-created object
set whose names encode the unique `v4-lane-id`, `v4-attempt-id`, and `v4-config-sha`.

```bash
KUBE_CONTEXT="$(kubectl create --dry-run=client -f rendered-bundle/configmap.yaml -o json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["kube.context"])')"
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod \
  create -k /absolute/path/to/rendered-bundle
```

Object names and the policy Service selector include lane, attempt, and
launch/config hash (`v4-lane-id`, `v4-attempt-id`, `v4-config-sha`). Each Job
selects exactly one logical `cuda:0` on its configured `nvidia.com/gpu.product`.
Never propagate a host GPU index into either Job.

Kubernetes does not sequence Jobs through a readiness probe. The policy Service
selects only the current lane/attempt/config identity; the simulator preflight
waits for that Service. For HTTP `/healthz` readiness (HTTP 200, body `OK`), raw
TCP connects are forbidden.

Inspect live state read-only:

```bash
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod get jobs,pods,service \
  -l "v4-config-sha=$CONFIG_HASH" -o wide
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod logs -f "job/$SIMULATOR_JOB"
```

## Output identity and evidence

The output parent must already exist and be writable on the PVC. Evidence paths
derive from immutable Downward-API identity under
`<output-parent>/.lane-runtime/<POD_UID>/lane-<LANE_ID>/attempt-<ATTEMPT_ID>/<role>/`.

Episode directories are write-once: the entrypoint never pre-creates them. Retries
require a new attempt ID.

## Job lifecycle

`restartPolicy: Never`, `backoffLimit: 0`, and no TTL preserve failed Jobs and
logs. Terminate the long-lived policy Job deliberately after the simulator
completes and evidence is verified; retain cluster JSON and logs before deletion.
