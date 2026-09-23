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

## Engineering repair boundary (SGW-ENG-007)

The independent `bf` scene review found artifact-valid media but rejected all
four scenes for engineering repair. New source overlays extend every added
kinematic support from the measured table top to its existing authored top
plane: footprint, color, side, scoring centers, and object orientations are
unchanged. This is a prospective visual/collision repair, not a rewrite of
the `bf` evidence or a fixture release.

The inherited banana is moved consistently in both families and both sides
using the measured root-local center offset, retained quaternion, measured
root-to-bottom offset, and measured table top. Its target geometric center XY
is `(0.80, 0.39)` m; the source does not describe that authored pose as an
observation. Every replacement capture must measure the banana root, center,
AABB, and filtered banana/table and banana/support contact matrices. Candidate
geometry screens fail closed if the measured banana leaves the table or comes
within 20 mm XY AABB clearance of any translated added support.

The DIST anchor remains the frozen simplified disc at its existing dimensions
and center, but is rendered off-white to contrast with its support. It retains
a documented category caveat: actual policy-sized views must be reviewed later;
the color change does not assert semantic recognition or alter a prompt.

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

### Observed ar infrastructure outcome

The original four-index ar attempt completed at the Kubernetes level on
23 September at 04:01:32 UTC, but all four containers exited zero without
`capture.json`, original camera arrays or render-diagnostic media. It produced
**zero valid captures**. The original Job, scene inputs and raw output roots
are immutable infrastructure evidence; they must not be overwritten or
misclassified as physical or model failures.

The capture CLI now fsyncs an exception/traceback receipt before native
cleanup can terminate the process, and writes its successful capture receipt
before environment teardown. Launchers must run the native Python command
as a child, not shell `exec`, then call `verify_capture_artifacts()` in a fresh
CPU-only Python process before reporting success. This checks the receipt,
source layers, all three original views, every retained warmup array and
all 121 decoded video frames. An exit-zero process without those files fails.
The check is artifact integrity, not visual review or physical qualification.

The initial recovery is one bounded diagnostic of the unchanged height-left
baseline in a new output directory. Capture failure recording fixes a
diagnostic blind spot; it does not establish the native root cause before
new evidence exists. No candidate cap, pose or scientific threshold changes.

### Diagnosed contact-body repair (SGW-ENG-006)

Diagnostic bc reproduced height-left using the original bytes. Its durable
exception identifies a contact sensor on `height_reference_support` with no
contact-reporter rigid body. The native process still exited zero, but the
separate output validator correctly made the Job fail. The pinned importer
activates rigid-body contact reporting; its all-pairs sensor builder cannot
attach a body sensor to an ordinary static collider.

New baseline revisions therefore author every added support with
`PhysicsRigidBodyAPI` and `physics:kinematicEnabled = true`. They stay fixed
instead of becoming gravity-driven dynamic objects; DIST's plate remains
dynamic. This is a disclosed collision/contact implementation correction,
not proof of identical physical behavior. New source-stage receipts must
compare all dimensions, colors, poses, quaternions and counterbalances with
the original four manifests, allowing only the support body flags to differ.
Original aq/ar/bc inputs and outputs are never edited or reused as successful
evidence. No HEIGHT/DIST candidate has yet been frozen.

New captures additionally retain the actual filtered cube-support force
matrices, including zero-valued contacts. A sensor name or a nonzero force at
one reset is not six-trial reachability or fixture qualification.

## Prospective design-plan boundary

After—and only after—the two native baseline captures for each family have
been reviewed, `prospective_family_designs.py` may produce a fixed 1–100-slot
design plan. Local tests use **synthetic capture-schema fixtures**, not native
evidence. A prospective plan is not an observed capture, a measured layout,
a reachable fixture, or a released candidate.

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
Materialization rechecks those inherited bytes and the captured source/asset
identity; a once-valid capture cannot authorize mutated scene dependencies.
