# V4 qualification record (prospective)

No live-cluster qualification run has been recorded for this V4 lane bundle.
This document exists so a future infrastructure gate can append dated evidence
without conflating it with historical V3 qualification or V4 scientific outcomes.

The checked-in `spec.example.json` is **qualification-only**: the simulator's final
argv is `/usr/bin/true`, so it produces no behavioral episode. Passing a future G1
infrastructure gate would demonstrate CUDA, rendering, imports, checkpoint digests,
writable-parent probes, ffmpeg encode/decode, policy readiness, and graceful
termination on the **configured GPU product strings** in that spec—not on every
authorized V4 hardware stratum by default.

Each new GPU class (for example B200 or H100), driver version, image digest, or
runtime layout requires its own spec, render, validator pass, and dated append to
this record before behavioral V4 work on that stratum.
