# π0.5 V3-B002 position-reflection ablation

This package runs the committed `V3-B002` registration only: seeds
`9400..9426`, four B001-ordered cells per seed, static direct commands, 15×8
π0.5 chunks, viewport video, and the frozen 450-action DROID predicate.

The launch sequence is fail-closed:

1. Run `model_blind_preflight.py` separately on every ali-owned RTX lane that
   will execute cells (six lane reports for the current topology). It makes
   zero model requests and checks all four exact B001 physical tasks, 60+15
   settling, neutral resets, RGB/viewport output, and fresh raw/action writers.
2. Run `fixed_observation_probe.py` against the assigned policy endpoint. It
   performs exactly three diagnostic requests: LEFT, identical LEFT repeat,
   and RIGHT.
3. Build the runtime identity and release gate with `runtime.py`, passing every
   lane report after the single `--model-blind-preflight` flag. The runtime
   binds the current study commit/source hashes, current policy pod UID/GPU2
   UUID, and every ali-owned simulator lane pod UID/GPU UUID.
4. Use `queue.py plan` before `run-queue`. `--lane-index I --lane-count N`
   assigns whole four-cell seed blocks; a seed is never split across lanes.

Every bridge process is launched through
`tools/native_process_group_thermal_guard.py`. Thermal events, model-specific
runtime interventions, invalid attempts, partial state streams, videos, action
arrays, and raw cell JSONL stay on the PVC. Infrastructure failures remain in
a separate append-only stream and are excluded from behavioral denominators.

After all 108 valid cells exist, invoke `compiler.py` with the 108 individual
cell JSONL paths. It emits 108 enriched episode rows, 54 separate pair rows,
H1/H2/H3 results, infrastructure rows, and post-close hash manifests.

Focused local validation:

```bash
python3 -m unittest \
  tests.test_pi05_v3b002_runtime \
  tests.test_pi05_v3b002_compiler
python3 tools/validate_vla_wam_v3_protocol.py
git diff --check
```
