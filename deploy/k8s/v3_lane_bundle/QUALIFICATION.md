# Qualification record

The qualification-only bundle introduced at commit
`bf6c6ff130752a46217e3bbf25095a5385ecba1c` was exercised once on the live
cluster on 2026-08-12.  This record is intentionally separate from scientific
experiment evidence: the simulator's final argv was `/usr/bin/true`, no episode
directory was created, and the retained logs contain no accepted WebSocket or
policy-inference event.

The following gates passed from fresh Kubernetes Jobs:

- policy entrypoint/preflight ran as PID 1 on A100
  `GPU-f29b160e-c1fb-4766-1b20-ced8645f38de` with driver `580.95.05`;
- simulator entrypoint/preflight ran as PID 1 on A40
  `GPU-8fcaadc2-f836-2077-c879-c0ee94c6a205` with driver `580.95.05`;
- both CUDA kernels, exact imports, checkpoint/file digests, writable-parent,
  isolated-cache, and ffmpeg encode/decode probes passed;
- the simulator produced a 20,526-byte RTX frame with SHA-256
  `d46f0fc0cdfb877354f11adb2c6c55666ef874fd3cb4ad060cfe4a6f9ec46e36`;
- the policy became Ready after checkpoint loading, and the simulator Job
  completed with exit code 0; and
- all qualification ConfigMap, Service, Job, and Pod objects were removed.

The first live variant used raw TCP availability for readiness.  That passed,
but generated expected invalid-WebSocket-handshake log noise.  The live run
also showed that the frozen policy process did not disappear promptly after
Kubernetes began termination.  The `preStop` receipt and final empty-selector
receipt were retained, but a terminal Pod JSON was not captured before garbage
collection, so this record does not assert an exact terminal exit code.  This
is an infrastructure finding, not a scientific failure.

The current bundle prospectively corrects both findings.  Readiness uses the
frozen OpenPI server's hash-bound `GET /healthz` path (HTTP 200, body `OK`) and
never enters its WebSocket inference handler.  Policy teardown writes and
fsyncs its receipt, sends SIGINT to PID 1, and waits boundedly for PID 1 to
disappear so the asyncio server can close before Kubernetes' SIGTERM deadline.
Both corrections have negative tests;
the revised identity must receive a fresh qualification before being used for
behavioral work.

The complete first-run evidence was copied to the PVC at:

```text
/data/users/ali/vla_wam/raw/v3_lane_bundle_qualification/
  bf6c6ff130752a46217e3bbf25095a5385ecba1c/
  attempt01-tcp-readiness/
```

The finalized directory contains 48 files (358,923 bytes).  Its initial
inventory covers 42 payload files (332,679 bytes), including both runtime
reports, Pod UIDs/image IDs, rendered frame, logs, launch bundle, and
retired-idle-pod receipts.  Additive final events and empty-selector receipts
are the two entries covered by `FINAL_CLEANUP_SHA256SUMS`.
