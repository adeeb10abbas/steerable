# GR00T N1.7 DROID — v3 Phase A

This adapter extends the frozen GR00T direct-command integration to the new
registered seeds `8303`–`8329`. It does not modify or rerun the valid v2 seeds
`8300`–`8302`.

The live path reuses these hash-pinned v2 components:

- `experiments/groot_droid/v2_robolab_client.py`
- `experiments/groot_droid/v2_seeded_server.py`
- both frozen `robolab_v2_tasks` files, including the neutral reset, exact
  prompts, 45-degree relation scorer, detached-release requirement, and
  success termination
- RoboLab's existing evaluator, one environment/run, viewport video, and
  30-second task timeout

There is no failure-based or progress-based early stopping. Only the frozen
success termination and task timeout can end a behavioral episode.

## Required live artifacts

Before planning or launching a pair, produce two small manifests outside Git:

1. A `vla-wam-shared-v3-groot-runtime-identity-v1` manifest with the exact
   checkpoint/repository/environment/renderer hashes required by
   `adapter.validate_runtime_identity`.
2. A `vla-wam-shared-v3-groot-release-gate-v1` manifest from the current
   runtime's model-blind fixture/write test and fixed-observation repeat plus
   LEFT/RIGHT sensitivity gate. All five release booleans must be true. The
   manifest must bind the committed Phase-A queue hash and runtime-manifest
   hash.

Fail-closed local preflight (no model request):

```bash
python3 experiments/v3/groot_droid/adapter.py preflight \
  --study-root "$STEERABLE_ROOT" \
  --seed 8303 \
  --runtime-identity "$RAW_ROOT/groot/runtime_identity.json" \
  --release-gate "$RAW_ROOT/groot/release_gate.json" \
  --check-live-repositories
```

Emit the exact matched-pair command without running it:

```bash
python3 experiments/v3/groot_droid/adapter.py plan \
  --study-root "$STEERABLE_ROOT" \
  --seed 8303 \
  --runtime-identity "$RAW_ROOT/groot/runtime_identity.json" \
  --release-gate "$RAW_ROOT/groot/release_gate.json" \
  --output-dir "$RAW_ROOT/groot/phase_a/seed8303/simulator" \
  --action-trace-dir "$RAW_ROOT/groot/phase_a/seed8303/actions" \
  --remote-host 127.0.0.1
```

The bridge must run from the restored RoboLab environment. It captures every
initial/post-action object state in the robot-base frame, executed actions via
the unchanged v2 client, and viewport video via RoboLab. Physical first-contact
timing is recorded as `instrumentation_unavailable` because the frozen runtime
does not expose a verified contact stream; grasp is deliberately not used as a
surrogate. A live integration should add the real contact stream before launch
if RoboLab exposes one without changing policy inputs or success semantics.

The bridge records the frozen scorer's success flag, true action
cap/right-censoring status, and the exact v2 legacy failure stage from the raw
state stream. The shared compiler independently recomputes that stage and
rejects any disagreement. Only success termination or a completed action-cap
failure can enter the behavioral denominator; technical and partial attempts
stay in the separate infrastructure stream and never receive a behavioral
taxonomy.
