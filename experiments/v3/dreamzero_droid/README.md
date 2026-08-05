# DreamZero s=2 action-guidance arm — powered v3 Phase A

## Exact identity (and what this is not)

The v3 registry uses model ID `dreamzero_droid_action_cfg`. That ID maps
exactly to the frozen V2-A015 arm `dreamzero_action_cfg_s2`. It does **not** map
to the original V2-A007 baseline, whose distinct model ID is
`dreamzero_droid` and whose conditional-action-equivalent scale is `s=1`.

Accordingly, this adapter fails closed unless all of the following are true:

- DreamZero source commit `ab790c198fbce33503358efbbd4187ce9a89adf3`
- checkpoint `GEAR-Dreams/DreamZero-DROID` revision
  `96ad344138c66e82536422432ad742f015784942`, plus its 25-file payload and
  tokenizer hash contracts
- V2-A015 overlay target hash
  `65dc9873aef37563dedf3787fd7b59e0a6d50e575775e38b70edd9e38489f9b8`
- action guidance `s=2`, released video guidance `5`, 16 runtime denoising
  steps with eight evaluated DiT steps under cache, `[24,8]` returned chunks,
  and eight-action open-loop execution

The s=2 intervention is derived CFG-style negative-branch action guidance
using DreamZero's fixed visual-quality negative prompt. It is not an official
DreamZero action-CFG feature and is never blended with or labeled as the s=1
baseline.

## Seed and evidence contract

Only new seeds `8303`–`8329` are launchable. Every launch contains the exact
LEFT and RIGHT prompts from the same neutral reset and environment seed.
DreamZero's released server hard-codes effective model-noise seed `1140`, so
the v3 `sampling_seed` is recorded honestly as a matched-pair/environment label
rather than misrepresented as a new effective policy-noise seed. This runtime
is therefore not eligible for Phase-D stochastic replication without a future
effective-seed release.

Every behavioral cell requires:

- full RTX viewport execution video;
- exact executed actions and returned action chunks;
- initial plus every post-action cube/reference pose in robot-base coordinates;
- all exposed latent video futures and returned actions;
- at least one hash-verified official full reset decode from the episode;
- a valid success termination or a complete 450-action failure.

Physical contact is `instrumentation_unavailable` unless a verified contact
stream is added without changing policy inputs or success semantics. Grasp is
not used as a contact surrogate.

## Fail-closed setup

Before planning a behavioral pair, provide outside Git:

1. `vla-wam-shared-v3-dreamzero-s2-runtime-identity-v1`, including the exact
   repository status, overlay, checkpoint, tokenizer, environment, renderer,
   queue, binding, and adapter hashes checked by
   `adapter.validate_runtime_identity`.
2. `vla-wam-shared-v3-dreamzero-s2-release-gate-v1`, bound to those runtime
   bytes and the v3 queue. It must pass the model-blind reset/write gates,
   bit-exact repeat LEFT actions and latent future, prompt-sensitive LEFT/RIGHT
   actions and latent future, complete future retention, and official reset
   decode. It also binds the isolated two-rank V2-A015 server contract and
   rejects protected port `5000`.

Preflight without a model request:

```bash
python3 experiments/v3/dreamzero_droid/adapter.py preflight \
  --study-root "$STEERABLE_ROOT" --seed 8303 \
  --runtime-identity "$RAW_ROOT/dreamzero_s2/runtime_identity.json" \
  --release-gate "$RAW_ROOT/dreamzero_s2/release_gate.json" \
  --check-live-repositories
```

Emit, but do not execute, the complete RTX matched-pair command:

```bash
python3 experiments/v3/dreamzero_droid/adapter.py plan \
  --study-root "$STEERABLE_ROOT" --seed 8303 \
  --runtime-identity "$RAW_ROOT/dreamzero_s2/runtime_identity.json" \
  --release-gate "$RAW_ROOT/dreamzero_s2/release_gate.json" \
  --output-dir "$RAW_ROOT/dreamzero_s2/phase_a/seed8303/simulator" \
  --action-trace-dir "$RAW_ROOT/dreamzero_s2/phase_a/seed8303/actions" \
  --remote-host "$ISOLATED_B200_SERVER" --remote-port 18022
```

Append-only partial state streams remain on the PVC after interruption.
Partial/setup attempts compile only to the infrastructure stream and never
receive a behavioral outcome.
