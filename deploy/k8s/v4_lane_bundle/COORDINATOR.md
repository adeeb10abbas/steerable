# V4 campaign coordinator dispatch

The coordinator entrypoint plans dependency-aware dispatch across qualified lanes
without mutating an existing Kubernetes object set.

```bash
python3 tools/launch_online_correction_v4.py --help
python3 tools/launch_online_correction_v4.py --dry-run \
  --runtime-lock /persistent/v4/runtime_lock.json \
  --queue /path/to/queue.jsonl \
  --queue-manifest /path/to/queue_manifest.json \
  --launch-matrix /path/to/launch_matrix.json
```

Dry-run never calls `kubectl`. Rendering fresh immutable bundles additionally
requires explicit cluster binding fields and a never-before-used
`--render-output-root`:

```bash
python3 tools/launch_online_correction_v4.py --dry-run \
  --runtime-lock /persistent/v4/runtime_lock.json \
  --render-output-root /tmp/v4-lane-render \
  --kube-context ali-prod-gpu \
  --namespace 211247-prod \
  --pvc 211247-prod-pvc \
  --output-parent /data/users/ali/vla_wam/raw
```

Behavioral cluster creation is opt-in, create-only, and requires durable group
leases so repeated waves cannot dispatch overlapping groups:

```bash
python3 tools/launch_online_correction_v4.py --create \
  --runtime-lock /persistent/v4/runtime_lock.json \
  --render-output-root /tmp/v4-lane-render \
  --group-lease-root /persistent/v4/group_leases \
  --coordination-state /persistent/v4/coordination_state.json \
  --kube-context ali-prod-gpu \
  --namespace 211247-prod \
  --pvc 211247-prod-pvc \
  --output-parent /data/users/ali/vla_wam/raw
```

Infrastructure qualification uses an explicit mode that keeps `/usr/bin/true`
simulator argv and reports zero behavioral episodes:

```bash
python3 tools/launch_online_correction_v4.py --create --qualification-only \
  --runtime-lock /persistent/v4/runtime_lock.json \
  --render-output-root /tmp/v4-qual-render \
  --kube-context ali-prod-gpu \
  --namespace 211247-prod \
  --pvc 211247-prod-pvc \
  --output-parent /data/users/ali/vla_wam/raw
```

Resume accepts completed or partial group receipts under
`--group-receipts-dir`. Each `*.group_receipt.json` must list accepted episode
IDs; partial groups preserve completed cells and dispatch only missing ones.

The coordination-state file (`v4-coordination-state-v1`) carries lane quarantine
counts, per-episode infra retry exhaustion, reserved attempt IDs, and the next
attempt index. Behavioral renders bind an immutable lane dispatch manifest with
exact `group_ids` and `episode_ids`, plus the released
`tools/run_online_correction_v4.py` runner SHA-256 from the runtime lock.

The returned plan includes an exact teardown inventory
(`ConfigMap`, both `Job`s, and the policy `Service`) keyed by immutable lane,
attempt, and config labels. Retain those objects and logs before deletion.

The coordinator fails closed while `runtime_lock.json` or `launch_matrix.json`
remain `NOT_RELEASED`, while all qualified lanes are quarantined but dispatchable
work remains, or when a qualification-only spec would masquerade as behavioral
dispatch. It never infers qualification, never retries valid behavioral failures,
and never reads interim effect sizes for scheduling.
