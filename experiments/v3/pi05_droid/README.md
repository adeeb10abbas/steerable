# π0.5 current-stack DROID — powered v3 Phase A

This adapter adds only the prospectively authorized seeds `8303`–`8329` to
the exact V2-A010 current-stack identity:

- OpenPI `c23745b5ad24e98f66967ea795a07b2588ed6c79`
- RoboLab `0aef241fb088ca21bb4ebd24448940ed56620d17`
- config/checkpoint `pi05_droid_jointpos_polaris`, bound to the committed
  26-file, 12,434,530,510-byte SHA-256 manifest
- 15-action open-loop execution, per-request seed
  `environment_seed * 1000 + request_index`
- the exact neutral cube/bowl reset, static direct prompts, frozen detached
  release-inside-45-degree-cone success predicate, viewport video, and
  450-action failure cap

Seeds `8300`–`8302` are preserved V2-A010 evidence and are rejected by this
launcher. Every fresh launch is a complete matched LEFT/RIGHT pair; a valid
failure runs to action 450. The state proxy retains the initial state plus
every post-action cube/reference pose in the robot-base frame. RoboLab does
not expose a verified physical contact stream in this integration, so contact
is explicitly `instrumentation_unavailable`; grasp is never substituted.

Before a plan is emitted, provide two compact manifests outside Git:

1. `vla-wam-shared-v3-pi05-current-runtime-identity-v1`, containing the exact
   repository status hashes, checkpoint/environment hashes, renderer identity,
   committed queue hash, and `adapter_contract_sha256` values checked by
   `adapter.validate_runtime_identity`.
2. `vla-wam-shared-v3-pi05-current-release-gate-v1`, produced on that exact
   runtime. It must bind the runtime bytes and queue hash and pass neutral
   reset, raw video/action/JSONL writes, bit-identical LEFT repeat, and
   non-zero fixed-observation LEFT/RIGHT action RMS.

Fail-closed preflight (no model request):

```bash
python3 experiments/v3/pi05_droid/adapter.py preflight \
  --study-root "$STEERABLE_ROOT" --seed 8303 \
  --runtime-identity "$RAW_ROOT/pi05/runtime_identity.json" \
  --release-gate "$RAW_ROOT/pi05/release_gate.json" \
  --check-live-repositories
```

Emit, but do not execute, the full matched-pair command:

```bash
python3 experiments/v3/pi05_droid/adapter.py plan \
  --study-root "$STEERABLE_ROOT" --seed 8303 \
  --runtime-identity "$RAW_ROOT/pi05/runtime_identity.json" \
  --release-gate "$RAW_ROOT/pi05/release_gate.json" \
  --output-dir "$RAW_ROOT/pi05/phase_a/seed8303/simulator" \
  --action-trace-dir "$RAW_ROOT/pi05/phase_a/seed8303/actions" \
  --remote-host 127.0.0.1 --remote-port 8001
```

The bridge leaves append-only partial state streams on the PVC. Only captures
reconciled as success termination or full-cap failure can be compiled into
behavioral JSONL. Setup and partial attempts use the separate infrastructure
compiler and never receive a behavioral taxonomy.
