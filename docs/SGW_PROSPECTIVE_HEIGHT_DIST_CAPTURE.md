# Prospective HEIGHT/DIST native capture

`prospective_family_scene.py` creates a new **prospective** USD overlay that
sublayers the supplied base scene without modifying it. Its supports and the
DIST plate are source-controlled design inputs, not measured geometry,
qualified fixtures, or frozen candidates. No command below creates candidates
or consumes the family’s 100-candidate cap.

The overlay builder binds the exact base scene bytes and the existing measured
LAT workspace receipt. For the current known baseline the receipt is
`artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json`,
whose receipt SHA-256 is
`3766dd5b9ed08f7fc60525077ba96df0d98b67e980ff453ee7d19559932b8239`;
the base asset manifest hash is
`3a9b8ceeff333aa3a060dad97707b5f2a50e7c4db1423c84350bf4d8a19b0806`.

The builder requires that receipt's **v2 measurement schema** and authors the
measured cube/bowl orientations as well as their translated actor roots.
Both families have a separate neutral cube support. DIST uses a circular plate,
with its support top at the disc's bottom, and separate landing supports clear
of the anchors. Both counterbalances are checked against the actual committed
workspace geometry; these are authoring checks, not physical qualification.

Parent must use the pinned RoboLab `0aef241` base USD path below on the assigned
lane only after reviewing the generated manifest:

```bash
.venv/bin/python -m experiments.workshops.spatial_grounding_v1.prospective_family_scene \
  --family HEIGHT --upper-side left \
  --base-scene /data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241/assets/scenes/rubiks_cube_banana_bowl.usda \
  --base-workspace-receipt artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json \
  --output-usda /data/users/ali/sgw-01/prospective/height-left.usda \
  --manifest-output /data/users/ali/sgw-01/prospective/height-left.manifest.json

.venv/bin/python -m experiments.workshops.spatial_grounding_v1.prospective_family_capture \
  --study-root "$PWD" --robolab-root /ABSOLUTE/PINNED/RoboLab \
  --assets-manifest /ABSOLUTE/assets-manifest.json \
  --renderer-receipt /ABSOLUTE/passed-renderer.json \
  --overlay-manifest /data/users/ali/sgw-01/prospective/height-left.manifest.json \
  --output /data/users/ali/sgw-01/prospective/height-left/capture.json \
  --headless --renderer realtime --rendering-type balanced --rendering_mode balanced \
  --device cuda:0 --num-envs 1
```

Run DIST analogously with `--family DIST --bowl-side left|right`. The capture
requires a 120-frame render-only warmup/video, zero controller actions, actual
root/center/offset/AABB records, lossless camera arrays, and contact sensor
inventory. Before candidate proposals, a parent-reviewed capture must prove
the named supports have usable cube-contact sensors, each final support is
reachable and released, the plate is visually distinct in actual views, and
the counterbalanced neutral starts are physically valid.

## Durable bounded sequencing

The four source/asset-bound designs are staged at
`/data/users/ali/sgw-01/infrastructure/source-stage-20260923aq/prospective`,
using exact source `ff2c361d51b3fe370cef2ec2b07347082a4ea5d4`.
`sgw01-ali-prospective-capture-20260923ar` is created **suspended**, with four
fixed indexes, no retries, and a 2,400-second Job deadline. Every capture keeps
its own render-only video, lossless frames, material inventory and raw receipt
on the PVC. These are not behavioral episodes or qualified candidate layouts.

`capture_sequencer.py` runs in a finite CPU-only Job independently of the
laptop. It checks exact Job UIDs and frozen specification hashes, and only
unsuspends ar after all 39 ak indexes complete and active/terminating
allocations are zero. Failure, changed identity, or a deadline blocks the
handoff. The next Job still performs a fresh idle-GPU check and exclusive
GPU-UUID lock. No other study GPU phase may be launched concurrently with
this two-Job chain.

The controller's dedicated ServiceAccount can only read ak and read/patch ar;
it cannot create Jobs, read secrets, list other workloads, or launch learned
inference. Its projected API token rotates and is not stored in Git or
receipts. Both GPU Jobs retain `automountServiceAccountToken: false`.
Completing this chain authorizes **evidence review only**. Candidate freezing,
scripted qualification, historical deduplication and model-runtime gates
remain separate.

## Prospective design-plan boundary

After—and only after—the two native baseline captures for each family have
been reviewed, `prospective_family_designs.py` may produce a fixed 1–100-slot
design plan. This is a **synthetic capture schema used by local tests until
the native baseline receipts exist**; it is not an observed capture, a
measured layout, a reachable fixture, or a released candidate.

Each plan binds both side-specific capture receipts, their baseline overlay
manifests, every captured USD dependency hash, the base scene/workspace/source
and asset identities, and a fixed seed. It applies only bounded XY rigid
translations to scored actor roots and the already-authored added supports;
the robot, table, cameras, retained root quaternions, and root-local scoring
offsets remain fixed. A geometric rejection still consumes its assigned slot:
there is no refill. Layout duplicates use the fixture reset rule, namely
maximum coordinate error at most 3 mm and orientation error at most 2 degrees,
regardless of side label.

An accepted design remains `prospective_design_requires_zero_model_capture`.
It must first author a no-overwrite candidate overlay and then collect a
**fresh zero-model capture of that exact overlay**. Candidate capture verifies
the candidate overlay and all mutable inherited baseline layers immediately
before AppLauncher. Only that new hash-bound capture can become input to later
measured-layout materialization and the separate six-trial qualification gate.
