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

Every release assertion must name its retained proof path and SHA-256. The runner recomputes those hashes, including the runtime-identity file, before it will produce a plan.

The wording block remains exploratory under the committed V3 analysis plan. Confirmatory wording claims require a later prospective power and inference amendment made before Phase-C behavioral inference.
