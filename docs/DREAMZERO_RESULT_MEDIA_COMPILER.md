# DreamZero V2-A007 result and video integration

This is the fail-closed compiler boundary between raw DreamZero execution on
the ali PVC and compact committed evidence. It does not run the model and it
does not create missing episode evidence.

The compiler requires all six valid cells before it writes anything:

- seeds 8300, 8301, and 8302;
- one static direct-command LEFT and RIGHT cell per seed;
- exact matched DROID reset and RTX PRO 6000 viewport video;
- simulator HDF5 actions exactly equal to the instrumented executed trace;
- official 24×8 action chunks, eight-action execution, and gripper processing;
- server-side official actions exactly equal to client-received raw chunks;
- every exposed latent and official decoded future retained and hash-valid;
- passed exact-repeat and prompt-sensitivity probe;
- separate invalid-attempt and runtime-intervention ledgers.

Missing cells, mismatched resets, missing videos, changed prompts, use of port
5000, a non-two-B200 server, incomplete futures, or action differences cause a
hard error. Missing or unexposed future evidence is never scored as zero.

## Raw collection manifest

Create this manifest on the PVC after all six cells exist. Paths may be
absolute or relative to the manifest. This example is a schema illustration,
not episode evidence:

```json
{
  "schema_version": "vla-wam-shared-v2-dreamzero-raw-collection-v1",
  "status": "complete",
  "model_id": "dreamzero_droid",
  "amendment_id": "V2-A007",
  "official_repository_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
  "checkpoint_revision": "96ad344138c66e82536422432ad742f015784942",
  "server_contract": "/data/users/ali/vla_wam/raw/dreamzero_droid/v2_a007/server/server_contract.json",
  "checkpoint_payload_manifest": "/data/users/ali/vla_wam/checkpoints/DreamZero-DROID/payload_manifest.json",
  "checkpoint_payload_root": "/data/users/ali/vla_wam/checkpoints/DreamZero-DROID-96ad344",
  "exact_repeat_probe": "/data/users/ali/vla_wam/raw/dreamzero_droid/v2_a007/probe/exact_repeat_probe.json",
  "invalid_attempt_ledger": "/data/users/ali/vla_wam/raw/dreamzero_droid/v2_a007/invalid_attempts.json",
  "invalid_attempt_count": 0,
  "runtime_intervention_ledger": "/data/users/ali/vla_wam/raw/dreamzero_droid/v2_a007/runtime_interventions.json",
  "runtime_intervention_count": 0,
  "cells": [
    {
      "environment_seed": 8300,
      "sampling_seed": 8300,
      "requested_relation": "left",
      "prompt": "Put the Rubik's cube to the left of the bowl.",
      "prompt_family": "direct_command",
      "prompt_controller": "episode_static",
      "oracle_actions": 0,
      "dynamic_prompt_switches": 0,
      "simulator_gpu_lane": "raytrace-rtxpro6000-ali",
      "simulator_task_dir": "/data/users/ali/vla_wam/raw/dreamzero_droid/v2_a007/seed8300/RubiksCubeLeftOfBowlMatchedTask",
      "action_trace_metadata": "/data/users/ali/vla_wam/raw/dreamzero_droid/v2_a007/action_traces/seed8300_left_executed_actions.json",
      "future_manifest": "/data/users/ali/vla_wam/raw/dreamzero_droid/v2_a007/futures/episode_000/future_manifest.json"
    }
  ]
}
```

The `cells` array must contain the analogous six exact seed/relation entries.
The explicit future-manifest path prevents a compiler from guessing the mapping
between server episode indices and behavioral cells.

`checkpoint_payload_manifest` points to the compact committed official-source
and checkpoint manifest. Its checkpoint inventory is nested under
`checkpoint.files`. `checkpoint_payload_root` is mandatory and names the exact
PVC directory containing those relative payload paths; the compiler rejects
absolute paths, parent traversal, symlink escape, missing files, byte-count
mismatches, and SHA-256 mismatches.

## Pending-state check

This validates that no DreamZero behavioral media is being implied before the
canonical manifest exists and regenerates the gallery with its pending card:

```bash
python3 tools/compile_vla_wam_v2_dreamzero.py \
  --check-pending \
  --regenerate-gallery
```

## Exact six-cell ingestion command

Discover the available ffmpeg/ffprobe pair in the execution pod; the compiler
records the exact binary hashes and ffmpeg version in the media manifest:

```bash
python3 tools/compile_vla_wam_v2_dreamzero.py \
  --collection-manifest /data/users/ali/vla_wam/raw/dreamzero_droid/v2_a007/collection_manifest.json \
  --git-head "$(git rev-parse HEAD)" \
  --ffmpeg "$(command -v ffmpeg)" \
  --ffprobe "$(command -v ffprobe)" \
  --result-output artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json \
  --media-dir artifacts/vla_wam_shared_v2/media/dreamzero_droid \
  --regenerate-gallery
```

The command emits the compact six-cell result, three paired RTX publication
videos, the canonical
`artifacts/vla_wam_shared_v2/media/dreamzero_droid/media_manifest.json`, and
regenerated HTML/Markdown galleries. All three frozen pairs are published;
there is no outcome-based video selection.

The compiler refuses to overwrite an existing result, clip, or media manifest.
Preserve partial attempts outside the complete collection, repair only the
invalid cell, and rerun compilation after the six valid cells are present.
