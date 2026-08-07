# V3-B008 / V3-B009 isolated Nano lanes

These wrappers turn the already-released V3-B008 and V3-B009 queues into two
independent Nano server contracts. They do not release or launch behavior by
themselves.

- V3-B008 (target start-side): TCP `18018`, 9-request fixed-observation gate,
  162 released behavioral cells.
- V3-B009 (target/reference role swap): TCP `18019`, 6-request
  fixed-observation gate, 108 released behavioral cells.

Each lane requires a fresh server process. Do not share a server, request
counter, client, output directory, or simulator session between amendments.
Start neither lane until the active V3-B005 process has stopped cleanly and its
raw-integrity check has passed.

## Order of operations

1. Bind the exact committed release to the verified Phase-A or V3-B005 Nano
   runtime with `bind_runtime.py`.
2. Start the amendment-specific wrapper in `probe_only` mode on its pinned
   port. Use the same official Nano CLI values as V3-B005; only the port
   differs.
3. Capture one model-blind fixed observation for every registered arm, then
   run `fixed_observation_gate.py`. The order is LEFT, exact LEFT repeat,
   RIGHT within each arm. Any repeatability, action-sensitivity, or
   decoded-future-sensitivity failure closes the lane at zero behavior.
4. Build the behavioral release with `build_behavioral_release_gate.py`.
5. Stop the probe server. Start a fresh amendment-specific wrapper in
   `behavior_only` mode with the behavioral-release path. Attach exactly one
   behavior client to that server.

The dedicated wrappers are `serve_v3b008_nano.py` and
`serve_v3b009_nano.py`. Startup fails before model load if the manifest,
runtime identity, checkpoint path, revision, CLI, or port differs.

All model responses retain the exact sampling seed, registered cell,
release/runtime hashes, and (for behavior) reset fingerprint. Behavioral
failures remain in the denominator; partial and infrastructure-invalid
attempts remain separate.
