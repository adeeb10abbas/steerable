# π0.5 V3-D001 nested stochastic rollouts

This package executes only the released `V3-D001` queue: 27 frozen DROID
environment seeds × 8 policy-sampling indices × 2 matched directions = 432
behavioral cells. The queue order is authoritative. Every LEFT/RIGHT block
uses one identical reset; the environment seed is never split across lanes.

For request index `i`, the policy server receives exactly
`policy_sampling_seed_base + i`. Prompts remain the exact static direct
commands, the action chunk remains 15×8 joint position, and failures run to
the frozen 450-action cap. Each episode retains one viewport video, every
executed action, every returned chunk, every state sample, and queryable
diagnostics. Infrastructure/partial attempts stay in their own stream and
never enter behavioral denominators.

On the released ali-owned simulator lane, run the first matched block as a
smoke test before removing `--limit-blocks 1`:

```bash
cd /data/users/ali/vla_wam/src/steerable-v3d001-4c7ad8b
python -m experiments.v3.pi05_stochastic_v3d001.queue run-queue \
  --repo-root "$PWD" \
  --release-manifest "$PWD/artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3d001/release_manifest.json" \
  --runtime-identity /data/users/ali/vla_wam/raw/v3/pi05_current_stack/release/runtime_identity_b200gpu0_rtxexpansion_attempt03.json \
  --phase-a-release-gate /data/users/ali/vla_wam/raw/v3/pi05_current_stack/release/rtxexpansion_attempt03/release_gate.json \
  --raw-root /data/users/ali/vla_wam/raw/v3 \
  --remote-host PI05_POLICY_HOST --remote-port 8001 \
  --lane-pod-uid ALI_SIMULATOR_POD_UID --lane-gpu-uuid ALI_SIMULATOR_GPU_UUID \
  --lane-index 0 --lane-count 1 --attempt-index 3 --limit-blocks 1
```

The bridge and compiler both revalidate every committed release/gate hash and
the exact Phase-A runtime before inference or denominator admission.

When the two additional exact ali RTX pods remain unschedulable, run
`sequential_supervisor.py` behind lane 0. It waits for all 144 lane-0 cells
and 72 pair manifests to validate before launching lanes 1 and 2 sequentially.
It fails closed if lane 0 exits early or either subsequent queue returns a
nonzero status; it never infers completion from a PID alone.

If an additional ali-owned lane is explicitly released while lane 0 is
running, restart the supervisor with `--external-lane-pid 1=PID` (or lane 2).
It will not duplicate that shard: it runs the remaining local shard, waits
for the external lane, and requires both its cells and pair manifests before
declaring completion.
