# V3-C001: four static phrasings

This package materializes the frozen Phase-C registry as 480 prospective DROID cells: three checkpoints, seeds 8500–8519, four prompt forms, and matched LEFT/RIGHT conditions. It does not launch inference.

`build_registration.py` verifies the committed source hashes, records the exact UTF-8 prompt bytes, and deterministically ranks all eight conditions within every model/seed block. A seed block is indivisible across execution lanes.

Before a model can run behavioral cells, its exact registered runtime must independently pass:

1. its Phase-A direct-command release proof;
2. all eight exact prompt-byte hashes;
3. exact fixed-observation repetition for every prompt form;
4. prompt-only LEFT/RIGHT sensitivity for every prompt form; and
5. a raw simulator-video, executed-action, state-trace, and per-episode JSONL write test.

`fixed_observation_gate.py` evaluates retained probe responses. GR00T is action-only; Edge and Nano must repeat and change both actions and their exposed decoded futures. `runner.py` accepts only a complete model-specific release manifest and emits a whole-seed execution plan. A model-specific live bridge must consume that plan and preserve its order; this package deliberately does not retrofit the seed-8303 Phase-A adapters.

`live_fixed_observation.py` is the zero-behavior live collector.  Its GR00T
backend consumes the exact Phase-A neutral-observation NPZ; its Cosmos backend
consumes the exact grounded image and source plan.  All returned arrays remain
on the PVC as hash-bound `.npy` files.  The compact JSONL contains artifact
references, never unbounded decoded-future pixels.  Evaluate that JSONL with
`fixed_observation_gate.py`, then use `build_release.py`; the latter still
requires an independent, zero-request raw video/action/state/JSONL write proof.
`raw_write_preflight.py` supplies that model-blind PVC/format proof and binds
it to the separately retained Phase-A simulator-release evidence.

The GR00T behavioral path starts with `groot_behavioral_bridge.py
--preflight-only`.  It resolves prompts by exact registered bytes rather than
searching for the words LEFT or RIGHT; this matters for the contrastive prompts,
which contain both words.  It validates one indivisible randomized eight-cell
seed block and the eight scorer-preserving task subclasses.  This slice still
issues zero model requests: live execution remains fail-closed until the
prompt-aware action/state/video writer is validated against that preflight.
`groot_task_registration_preflight.py` is the next zero-action gate: it starts
the real Isaac renderer, registers all eight task subclasses, performs only the
two frozen initialization resets, verifies the exact prompt exposed by every
environment, checks neutral/matched geometry, and closes without contacting the
policy server.

`groot_live_bridge.py` consumes both zero-request preflights and revalidates
the release, execution-plan, runtime-repository, checkpoint, task-source, and
fresh-output identities before Isaac starts.  It executes one indivisible
eight-cell seed block in its registered order.  The writer defers opening a
state stream until the first action so RoboLab's two initialization resets do
not overwrite or masquerade as evidence; it then retains the second reset,
every post-action state, exact-prompt action chunks/modalities, a decodable
viewport MP4, and one validated v3 behavioral JSONL row per cell.

Every release assertion must name its retained proof path and SHA-256. The runner recomputes those hashes, including the runtime-identity file, before it will produce a plan.

The wording block remains exploratory under the committed V3 analysis plan. Confirmatory wording claims require a later prospective power and inference amendment made before Phase-C behavioral inference.
