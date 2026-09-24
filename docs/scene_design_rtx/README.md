# LAT and HEIGHT scene repair on the RTX workstation

This work is separate from the cluster's active study checkout. It changes
object/support placement and, at Ali's request, removes the office panorama
and wood texture. It does not run learned policies, edit frozen prompts or
scoring, or control Kubernetes workers.

Workstation root: `/home/ali/sgw-scene-design-20260923`.
Source branch: `scene-design-rtx-20260923`, based on `70a6bbaa`.

## Clean scenes

![LAT](previews/lat.png)
![HEIGHT with the higher platform on the right](previews/height.png)

These are native simulator captures. `clean-studio-v1` uses a uniform dome
light, matte tabletop and the existing floor with a neutral material. The
table, object, robot and camera transforms are unchanged by this appearance
revision. The initial LAT captures before/after the change have exactly equal
measured object geometry, robot reset configuration and all three camera poses.
The original room versions and every result are retained separately.
Newly generated designs default to this clean appearance. Treat it as a new
visual condition in the paper: every wording within a matched comparison
must use the same appearance. Keep any existing office-background model
results labeled separately.

Commit `cd3018b7` preserves the exact three authoring/runner/task source files
hashed in registration B and used by the already launched prototypes. Later
changes only make the clean style the generation default and retain a measured
support-side label for subsequent candidate selection. The running workstation
prototypes continue with their recorded source snapshot.

## Current evidence and running work

This is **scene development**, not a completed 29-layout collection.

| Input | Scene | Current result |
|---|---|---|
| `prototype-00.json` | Revised LAT, original appearance | Independently verified 5/6. Positive goal, reset 1 failed strict terminal stability. |
| `prototype-01.json` | Revised HEIGHT, upper platform left, original appearance | Independently verified 6/6. |
| `prototype-02.json` | Revised HEIGHT, upper platform right, original appearance | Authored, not launched; clean version used instead. |
| `prototype-03.json` | Clean version of 00 | Independently verified 5/6; the same positive-reset-1 stability failure. |
| `prototype-04.json` | Clean version of 01 | Running on GPU 1. |
| `prototype-05.json` | Clean version of 02 | Independently verified 6/6. |
| `prototype-06.json` | Clean LAT using exact historical SGW-ENG-008 geometry | Native aggregate 6/6; independent recheck in progress. Historical reuse is disclosed in registration C. |

For live status read
`/home/ali/sgw-scene-design-20260923/evidence/scene-design-status.json`.
The bounded collector verifies the raw records and decodes every saved video
after each complete run. It then archives arrays losslessly, decompresses and
checks every byte before removing the redundant loose arrays. JSON records,
preview PNGs and videos remain directly readable. A failed processing step is
reported explicitly and does not change a physical outcome. The collector
does not launch or retry experiments.

The numerical placement in LAT-00 was correct in the failed trial: signed
margin 130.27 mm, supported and released, and bowl drift about 0.03 mm. It
failed because terminal angular speed reached 0.235 rad/s, above the unchanged
0.2 rad/s criterion. Do not describe this as an instruction-understanding
failure; no learned model ran. Prototype 06 tests the exact previously
successful coordinates rather than rounding that example to a new geometry.

## Fixed qualification campaign

The new campaign is recorded under
`artifacts/workshops/spatial_grounding_v1/clean_campaign_20260924`.
Its immutable plan SHA-256 is
`3fdd3ab3fecad77aafd67d5fac834b2a3f7a7df729a1a3888e1d94b2014cabd1`.
It lists 100 candidate inputs per family **before collecting their outcomes**,
in a fixed order. It uses the exact historical LAT template translated on a
10 mm lattice, and the revised short-transfer HEIGHT geometry. Zero
translation is excluded so the reserved historical pilot is not reused.

The two finite workers use GPU 0 for LAT and GPU 1 for HEIGHT. LAT waits for
verified prototype 06; HEIGHT waits for verified 04 and 05. Each candidate
gets one fresh simulator process, two goals, three resets and 450 actions per
trial. Every physical rejection is retained. A family stops once its required
29 scenes are available, or at 100 attempted candidates, 24 hours, less than
40 GiB free, an occupied GPU, or an infrastructure/evidence failure. Stopping
short of 29 is explicit; it cannot produce a completed assignment file.

The campaign uses the same environment seed, 20260923. HEIGHT requires a
right-side pilot, two development scenes per side and twelve confirmation
scenes per side. Assignments are selected in the frozen candidate order.
Additional accepted scenes on one side cannot substitute for the other side.

Live files on the workstation:

- `evidence/SGW-CLEAN-20260924/LAT/status.json`
- `evidence/SGW-CLEAN-20260924/HEIGHT/status.json`
- `logs/clean-campaign-LAT.log` and `logs/clean-campaign-HEIGHT.log`

When a family finishes, its `assignments.json` maps P01/D01-D04/C01-C24 to
the retained scene directories. This qualifies fixtures; it does not launch
models, release a model experiment, modify active workers, or merge the new
visual condition with old model results. At the measured prototype speed,
building both full sets takes several hours even without rejections.

To resume a stopped worker after addressing its recorded cause, use the
same plan and output root. Never change a rejected candidate's input. The
worker skips verified completed outputs, refuses partial attempts and uses a
per-family lock to prevent duplicate launches:

```bash
PY=/home/ali/sgw-scene-design-20260923/venv/bin/python
$PY -m experiments.workshops.spatial_grounding_v1.scene_design_batch run \
  --plan artifacts/workshops/spatial_grounding_v1/clean_campaign_20260924/plan.json \
  --expected-sha256 3fdd3ab3fecad77aafd67d5fac834b2a3f7a7df729a1a3888e1d94b2014cabd1 \
  --family LAT --gpu 0 --task-root /home/ali/sgw-scene-design-20260923
```

Use `HEIGHT --gpu 1` for the other family. Leave current workers alone while
they are running. Source hashes are checked against the recorded plan.

## Commands for an agent

Do not duplicate the already running/queued prototypes. Their outputs are
under `evidence/prototype-NN-a`; logs are under `logs/prototype-NN-a.log`.

For a **new, prospectively recorded** design, from the isolated checkout:

```bash
bash docs/scene_design_rtx/run-scene.sh /absolute/path/design.json 0 /absolute/path/new-evidence
```

One process occupies one GPU. Use GPU 0 or 1 only after the current process
on that GPU finishes. The launcher uses the existing isolated environment;
there is no need to reinstall Isaac Sim or download the scene assets again.
Use a fresh output path: existing evidence is never overwritten. Keep at least
40 GiB free before starting another pair of runs. Do not start a 100-scene
batch until its measured archive size fits the remaining storage budget.

After a run finishes:

```bash
PY=/home/ali/sgw-scene-design-20260923/venv/bin/python
$PY -m experiments.workshops.spatial_grounding_v1.scene_design_verify /absolute/path/new-evidence
$PY -m experiments.workshops.spatial_grounding_v1.scene_design_archive /absolute/path/new-evidence
```

To restore archived source arrays before recomputing verification:

```bash
cd /absolute/path/new-evidence
tar --zstd -xf raw-arrays.tar.zst
```

Verification refuses to overwrite an existing `verification.json`; retain it
and write an independent recheck to a different path by calling `verify(root)`.
The archive is lossless storage, not a substitute for validation.

The environment uses RoboLab `0aef241fb088ca21bb4ebd24448940ed56620d17`,
Python 3.11, Isaac Sim 5.0.0.0, Isaac Lab 2.2.0 and Torch 2.7.0+cu126.
Installed versions are retained in `evidence/runtime-freeze.txt`; each launch
also records its scene, input, source, asset and controller hashes.

## What caused trouble in the old scenes

The old LAT campaign produced only 20 all-six passes. A later translated
layout at a 0.50 m cube-center distance passed all six checks. Its starting
cube had previously been about 0.278 m from the robot root. This points to a
usable working region, not a universal optimal radius.

The original HEIGHT construction started the cube at x=0.29 m and placed its
goals at x=0.65 m, y=+/-0.23 m. Its mirrored versions had unequal pass rates.
In the retained right-side native smoke result, five trials passed and the
sixth disturbed the bowl by 7.63 mm, exceeding the unchanged 5 mm limit.
That failure is physical; it should not be reclassified or retried away.

## Smaller pieces of work

1. **Establish three working examples.** One LAT scene and both HEIGHT support
   orientations. Use `scene_design_runner`, the existing measured-TCP
   controller and the existing scorer. Inspect all views and both goal videos.
2. **Freeze the candidate list.** Only after those examples are understood,
   record the generator version, seed, bounds, candidate list and ordering.
   This is a new, disclosed campaign. Preserve the old pools unchanged.
   The provided `campaign(family, workspace, seed, count)` returns up to 100
   deterministic, distinct layout inputs. It does not launch anything or
   qualify them. Record the complete list, hash order, implementation and
   selection rule before collecting outcomes. Do not replace rejected rows.
3. **Qualify each candidate.** One fresh simulator process per scene; each goal
   across three full resets; 450 actions each. Retain failed outcomes too.
   An interrupted process is incomplete, not a failed physical trial.
4. **Select and hand off.** Select 29 distinct all-six passes per family using
   the frozen order. HEIGHT needs 2/2 support sides for development and 12/12
   for confirmation, plus one pilot. Export scene files, actual measured
   candidate poses, hashes, videos and the qualification ledger together.
   With seed 20260923 the pilot side is right, so HEIGHT needs at least 15
   right-side and 14 left-side accepted candidates. Do not reuse prototype 06
   as a new independent D/C layout: its historical geometry is already
   reserved for the existing main pilot assignment. Its role here is a
   working engineering example.

## Initial revised geometry (not yet a physical result)

- LAT: cube near x=0.46 m, y=-0.18 m; bowl 0.20 m farther forward; goals
  0.13 m to either side of the cube. Both placements stay on the table.
- HEIGHT: cube near x=0.46 m, y=-0.10 m; bowl 0.22 m farther forward.
  Goals are short 0.18 m lateral transfers onto 0.16 m square platforms.
  Cube-center goal heights are 55 mm above/below the bowl. The neutral cube
  and bowl centers remain at the same height. Support colors are neutral.
- The banana is retained away from the transfer area. Robot assets, initial
  joints, camera registration, simulator/controller configuration and scoring
  thresholds are inherited from SGW-01.

Authoring values are converted through measured center-to-root offsets.
Native captures measure the resulting geometry before qualification. No
placement is called qualified merely because the scene renders or IK finds a
solution.

## Source responsibilities

- `scene_design.py`: deterministic geometry and USD overlay generation.
- `scene_design_task.py`: import the generated overlay into the pinned DROID
  robot and camera setup; no success-based termination.
- `scene_design_runner.py`: record the actual scene and run the existing
  six-trial qualification. Outputs are immutable per attempt.
- `scene_design_verify.py`: independently recompute all six results from raw
  action/state/frame files and fully decode their videos.
- `scene_design_archive.py`: verified, reversible lossless array storage.
- `scene_design_collect.py`: bounded completion/verification of the four
  registered clean prototype jobs; never launches further experiments.
- `scene_design_batch.py`: the finite, immutable two-family qualification
  campaign; waits for verified templates and preserves rejected candidates.
- Existing `grasp_calibration.py`, `model_blind_qualification.py`,
  `robolab_lat_qualification.py`, and `scoring.py`: unchanged execution and
  measurement contracts.

See the dated registration under
`artifacts/workshops/spatial_grounding_v1/scene_design_rtx_20260923` for what
was fixed before each native attempt. The prototypes are engineering work,
not learned-policy evidence or additions to the old frozen candidate pools.
