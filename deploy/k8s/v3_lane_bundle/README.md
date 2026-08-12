# V3 Kubernetes lane bundle

This package launches one simulator lane and one policy server as fresh
Kubernetes `Job` objects.  Each container starts the audited lane entrypoint as
PID 1, runs a synchronous fail-closed startup preflight, then replaces itself
with the exact experiment process.  It does not use `kubectl exec`, background
servers, reusable Pods, or `sleep infinity`.

The bundle is infrastructure, not scientific authorization.  A passing startup
preflight proves that the declared runtime can execute CUDA, render a frame,
import the pinned Python/CuRobo stack, encode/decode video, read the bound
checkpoint and inputs, and write to the output parent.  It does **not** prove
that an experiment's starting state, intervention, predicate, registration, or
causal interpretation is scientifically valid.  Experiment-specific gates
remain mandatory.

## Inputs and rendering

Render from a complete immutable launch specification.  The specification must
bind, rather than infer, all of the following:

- namespace, lane ID, attempt ID, and a launch/config hash used in every object
  name and selector;
- content-addressed simulator and policy images (A40 and A100 respectively);
- the existing PVC output parent (not an episode directory);
- absolute Python, ffmpeg, CUDA/Isaac-render-probe, checkpoint, and
  experiment executable paths;
- exact policy and simulator argv arrays;
- checkpoint and launch-config SHA-256 digests;
- every registration, queue, runner, adapter, checkpoint manifest, and other
  launch-critical file as `{path, bytes, sha256}`;
- the policy port and unique Service identity; and
- simulator-side policy-wait parameters.

The simulator's combined Vulkan contract is a successful Isaac AppLauncher RTX
camera frame under the bound NVIDIA Vulkan ICD; the live lane images do not
contain `vulkaninfo`. The rendered launch ConfigMap is immutable. Do not hand-edit rendered YAML
or transcribe argv/hashes into an interactive command.  Re-render under a new
attempt/config identity when an input changes.  The renderer command and exact
specification schema are documented by `tools/render_v3_k8s_lane_bundle.py`.

Before cluster creation, run both local checks:

```bash
.venv/bin/python tools/render_v3_k8s_lane_bundle.py --help
.venv/bin/python tools/validate_v3_k8s_lane_bundle.py \
  --root /absolute/path/to/rendered-bundle \
  --spec /absolute/path/to/the-original-render-spec.json
kubectl create --dry-run=client -k /absolute/path/to/rendered-bundle -o yaml \
  > /absolute/path/to/rendered-bundle/client-decoded.yaml
```

The validator checks the Job, GPU, cache, ConfigMap, PID-1, Service, readiness,
preflight, and evidence contracts.  Client decoding is inspection only; it does
not qualify a cluster node or authorize inference.

The Jobs mount runtime and runner files from the shared PVC. The checked-in
qualification spec expects `/data/users/ali/vla_wam/src/steerable`; sync the
intended pushed commit there and hold its bytes fixed for the whole attempt.
For real runs, render a new spec pointing to an immutable, commit-suffixed
detached checkout. Verify its commit and file hashes from both GPU classes.
Never update any checkout while a lane refers to it. The startup
`file_bindings` fail closed if the mounted bytes differ.

The checked-in example is qualification-only: after every infrastructure probe
the simulator executes `/usr/bin/true`; it produces no scientific behavior.
For a real experiment, create a separate immutable spec with the complete
registered runner argv and bindings. Never repurpose or hand-edit this example.

## Creating a lane

Use a cluster context and namespace explicitly.  Create new immutable objects;
do not use `kubectl apply`, because an existing Job, Service, or ConfigMap must
fail closed rather than be silently reused or patched.

```bash
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod \
  create -k /absolute/path/to/rendered-bundle
```

Object names and the policy Service selector must include the lane, attempt,
and launch/config hash.  The simulator is pinned to one logical `cuda:0` on an
`NVIDIA-A40`; the policy server is pinned to one logical `cuda:0` on an
`NVIDIA-A100-SXM4-80GB`.  Never propagate a host GPU index into either Job.

Kubernetes does not sequence Jobs through a readiness probe.  The policy
Service selects only the current lane/attempt/config identity, and the simulator
preflight waits for that Service. For the frozen π0.5 server, both the policy
probe and simulator send a valid HTTP/1.1 `GET /healthz` and require status 200
with body `OK`; raw TCP connects are forbidden because they create invalid
WebSocket-handshake noise. The hash-bound server exposes health only after its
checkpoint is loaded. The imported OpenPI health-server source itself is an
exact file binding, while the policy preflight separately binds the checkpoint
and launch config. Servers with an audited metadata endpoint may
instead use `metadata_jsonl` readiness.

Inspect live state without entering the containers:

```bash
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod get jobs,pods,service \
  -l "v3-config-sha=$CONFIG_HASH" -o wide
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod wait \
  --for=condition=ready pod -l "v3-lane-role=policy,v3-config-sha=$CONFIG_HASH" \
  --timeout=30m
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod logs \
  -f "job/$SIMULATOR_JOB"
```

The exact labels and object names are in the rendered manifests.  Do not select
a Pod by a historical name alone: names can be reused, while `POD_UID` is the
runtime identity retained in evidence.

## Output identity and evidence

The output parent must already exist and be writable on the PVC.  The
entrypoint derives evidence from immutable Downward-API identity:

```text
<output-parent>/.lane-runtime/<POD_UID>/lane-<LANE_ID>/attempt-<ATTEMPT_ID>/<role>/
  runtime-preflight.json
  runtime-startup.json
  runtime-preflight.passed
  rendered-frame.png
  ffmpeg-encode.mp4
  ffmpeg-decode.raw
  prestop-received.json                 # after termination, when delivered
  entrypoint-failure.json               # only for an early infrastructure failure
```

The reports retain the effective allowlisted environment, real GPU UUID and
driver, expected image digest, complete argv, launch-config hash, checkpoint
hash, exact file bindings, pod UID/name/namespace/IP, and all preflight results.
The actual runtime image ID must subsequently be reconciled from Pod status;
the Downward API cannot expose it inside the container.

The runner receives this path but the entrypoint never creates it:

```text
<output-parent>/<POD_UID>/lane-<LANE_ID>/attempt-<ATTEMPT_ID>/episodes
```

The experiment owns the first atomic, write-once creation.  A pre-existing
episode directory is an infrastructure error, not a resumable lane.  Attempt
and policy-port lock names are also created with `O_EXCL`; retries require a new
attempt ID and preserve the failed attempt.

## Logs, completion, and removal

`restartPolicy: Never`, `backoffLimit: 0`, and the absence of a TTL preserve
failed Jobs and their logs.  The finite simulator Job exits naturally.  A
policy-server Job is long-lived, so after the simulator has completed and its
raw/evidence files have been verified, terminate the policy Job deliberately.
Its `preStop` hook writes and fsyncs a PVC marker, sends SIGINT to policy PID 1,
then polls PID 1 boundedly (120 seconds by default, within the 300-second grace)
so the asyncio server can close before Kubernetes sends SIGTERM. The simulator
hook writes its marker but does not signal PID 1.

Before removing any cluster object, retain Kubernetes-side evidence outside the
objects themselves:

```bash
mkdir -p "$CLUSTER_EVIDENCE_DIR"
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod get \
  job "$POLICY_JOB" "$SIMULATOR_JOB" -o json \
  > "$CLUSTER_EVIDENCE_DIR/jobs.json"
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod get pods \
  -l "v3-config-sha=$CONFIG_HASH" -o json \
  > "$CLUSTER_EVIDENCE_DIR/pods.json"
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod logs \
  "job/$POLICY_JOB" > "$CLUSTER_EVIDENCE_DIR/policy.log"
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod logs \
  "job/$SIMULATOR_JOB" > "$CLUSTER_EVIDENCE_DIR/simulator.log"
sha256sum "$CLUSTER_EVIDENCE_DIR"/* > "$CLUSTER_EVIDENCE_DIR/SHA256SUMS"
```

Verify the PVC runtime evidence and experiment output. Then stop the one exact
long-lived policy Job so its `preStop` marker can be retained:

```bash
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod delete job "$POLICY_JOB"
```

After verifying that marker, remove only the remaining exact rendered object
names; never delete by a broad owner or study label:

```bash
kubectl --context "$KUBE_CONTEXT" --namespace 211247-prod delete \
  job "$SIMULATOR_JOB" \
  service "$POLICY_SERVICE" \
  configmap "$CONFIGMAP"
```

Deleting Kubernetes objects does not delete PVC evidence.  Report any failed
startup as infrastructure-invalid and exclude it from behavioral denominators;
retain valid behavioral failures in their registered denominators.
