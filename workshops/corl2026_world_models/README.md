# Spatial Instruction Following in World–Action Models

The paper asks a simple question: **when a robot fails a left/right placement
command, did the instruction fail to redirect it, or did the resulting execution
fail in that scene?**

The manuscript now presents completed Cosmos 3 Nano and DreamZero-s2 experiments.
Each model has 108 trials across two directions and two fixed object layouts.
Reflection changes DreamZero-s2 completion from 13/54 to 50/54 trials. Nano remains
highly successful, but which requested side produces greater placement depth
changes. Both models generally order their final cube–bowl positions according
to the command.

## Read the paper

- [Exact ablation specification](docs/ABLATION_SPEC.md), including the verified
  existing-experiment inventory, 232 planned core cells and separately listed
  optional guidance extension. All new cells remain unreleased.

The current draft has received a weak-reject internal review. Its DreamZero-s2
label now identifies the custom action-guidance scale 2 condition and states
that baseline scale 1 was not used. The retained Nano
stopping-sensitivity analysis is now incorporated: historical goal-dependent
endpoint responses are distinguished from the latest shared strictly
pretermination comparison and fixed-action coverage bounds. This retrospective
control uses one base scene and its reflection; it is neither prospective
fixed-duration evidence nor forecast accuracy.

A second retained H01 diagnostic rederives the first action-index entry into the
requested 45-degree cone from every step trace and preserves four no-entry
reflected-RIGHT trials as cap-censored. Its 27 matched four-cell seed blocks
quantify a layout-by-command entry-time interaction and audit three-sample
sustained-entry status without assigning goal-terminated transient cases a
common-horizon time. This is fixed-scene onset geometry, not physical-time
latency, release success, language understanding, A5--A7 or forecast accuracy.

The separate retained H05 position sweep is now authenticated and reproduced
at report level. Its prospectively frozen design has 15 matched seed blocks,
seven prescribed bowl-y positions and two commands (210 episodes). The
preregistered within-seed RIGHT-minus-LEFT depth slope is 1.125 m/m (95%
matched-seed bootstrap interval 0.719--1.562; 13/15 positive). The
[canonical JSON](paper/data/h05_lateral_dose_response.json) records exact source
Git identities, arithmetic checks and claim boundaries; the
[105-row CSV](paper/data/h05_lateral_dose_response.csv) exposes every retained
seed-level contrast. The raw 105 pair and 210 episode files are represented by
unique SHA-256 identities in the authenticated source manifest but are not in
this checkout, so this is not a fresh raw-trace audit. The seven levels are
repeated positions in one fixed simulator setup, not seven scenes, and this
retained executed-behavior result supplies no forecast-accuracy or annotation
gate evidence.

The retained H06 Nano graded-symmetry study is also authenticated from its
prospectively registered compact evidence. Its primary inventory-matched grid
contains 27 matched seeds at symmetry levels 0.25, 0.50, 0.75 and 1.00 (216
episodes/108 command pairs). The within-seed requested-depth-gap slope against
realised object-layout asymmetry averages 0.020 m per unit asymmetry (95%
matched-seed bootstrap interval 0.0004--0.0400; 20/27 positive, exact
$p=0.0192$). [Canonical JSON](paper/data/h06_graded_symmetry.json) authenticates
and strictly parses the full 4,096-episode compact V3-E004 table, while the
[108-row CSV](paper/data/h06_graded_symmetry.csv) exposes the exact primary
contrasts. Level zero is excluded because companion objects are absent there;
it is not inventory matched to positive levels. Original raw simulator files
remain external. This is historical executed behavior, not independent scenes,
position-reflection replication, equivalence, forecast accuracy or a release-
gate advance.

The retained H07/H08 endpoint package independently reconstructs the two
registered endpoint-only V3-E004 checkpoints from the same byte-authenticated
compact tables. For each of DreamZero-s2 and Cosmos Edge it rejoins 108
episodes into 54 LEFT/RIGHT pairs over 27 seeds and both endpoint levels. The
registered $s=1$ minus $s=0$ RIGHT-minus-LEFT requested-depth interactions are
-0.124 m (95% matched-seed bootstrap interval -0.173 to -0.075) and -0.192 m
(-0.260 to -0.115), respectively. The registered matched-block success-gap
interval excludes zero for DreamZero-s2 but not for Edge. Endpoint-redirection
interactions are +0.104 m (interval 0.043 to 0.165) and -0.027 m (-0.099 to
0.047), respectively. The
[canonical JSON](paper/data/h07_h08_endpoint_symmetry.json) records exact source
identities, row joins, registered-report reconciliation and all claim
boundaries; the [108-row CSV](paper/data/h07_h08_endpoint_symmetry.csv) exposes
both models' seed-level endpoint contrasts. Companion objects are absent at
level zero, so these are inventory-transition interactions in separate model
contexts, not position-reflection replications, equivalence, independent
scenes, forecast accuracy or a release-gate advance.

The retained H09/H10 wording-control package authenticates the prospectively
frozen V3-C001 registration and all 320 Nano/Edge episode rows selected by this
ablation specification. Each model contributes 20 matched seed blocks crossed
with four fixed prompt forms and LEFT/RIGHT (160 episodes): Nano succeeds in
134/160 and Edge in 123/160; the historical rows contain 1,234/1,365 model
requests and 26/37 valid behavioral failures. Against the within-seed
direct-command control, the block-balanced success contrasts for
short-command/goal-as-outcome/desired-plus-negated-opposite wording are
0.000/-0.100/-0.450 for Nano and -0.350/-0.125/-0.250 for Edge;
the requested-margin contrasts are -1.93/-6.02/-19.39 cm and
-13.17/-2.15/-10.84 cm, respectively. Requested-margin and endpoint-shift
contrasts are algebraically paired views, not independent signals. The
[canonical JSON](paper/data/h09_h10_wording_control.json) records the frozen
registration, exact compact-Git provenance, all-condition summaries, matched
discordance tables and claim boundaries; the
[120-row CSV](paper/data/h09_h10_wording_control.csv) exposes every model x
seed x non-direct-form contrast. The full registration contains 480 rows and a
third GR00T cohort; the ablation specification fixes Nano and Edge as this
retrospective retained slice, and models are never pooled.
The 20 seeds repeat across eight cells in one neutral scene, confirmatory
wording inference was not registered, and original raw PVC bytes remain
external. This is exploratory retained executed behavior, not general language
comprehension, forecast accuracy, annotation, confirmation or a release-gate
advance.

The retained H11/H12 Phase-A direct-command package separately authenticates
the completed Nano and custom DreamZero-s2 additions over seeds 8303--8329.
Each model contributes 27 matched LEFT/RIGHT reset blocks (54 behavioral
episodes): Nano succeeds in 51/54 (26/27 LEFT, 25/27 RIGHT), while DreamZero-s2
succeeds in 20/54 (3/27 LEFT, 17/27 RIGHT). Nano's 349 historical requests each
retain a decoded RGB future and returned action chunk. DreamZero-s2 instead
retains 2,554 latent futures under fixed model-noise seed 1140, with only 54
official full-reset decode MP4s and 54 rollout videos; the latent count is not a
per-request RGB-decode count or an independent-sample count. The
[canonical JSON](paper/data/h11_h12_phase_a_direct.json) authenticates the
compact Git summaries, manifests, ledgers, exact cohort selection and claim
boundaries: eight H11/H12 payload objects total 5,636,174 bytes, while two
design/overlap helpers add 18,793 bytes (ten objects and 5,654,967 bytes total).
The [54-row CSV](paper/data/h11_h12_phase_a_direct.csv) exposes one
LEFT/RIGHT contrast per model and reset block. Six preserved V2 cells per model
at seeds 8300--8302 are evidence-only and excluded. The models remain separate,
and H12's directional totals matching H07 are a numerical coincidence across
identity-distinct cohorts, not a replication or pooled confirmation. Referenced
raw Phase-A payload bytes remain external and were not revalidated. This is
retrospective fixed-scene behavior, not independent-scene, forecast-accuracy,
annotation, recovery-completion or release-gate evidence; all current activity
counters remain zero, all gates false and the 232-cell core bound unchanged.

The retained H13 evaluator-development package reopens and rehashes all 761
ledger-listed upstream source Git blobs (7,230,812 bytes), including the 752
Edge V1 chunk caches,
plus five immutable audit/annotation objects (998,249 bytes). The historical
scorer labeled 421/752 chunks but covered only 25/97 executed-positive horizons;
on that same covered subset its apparent historical-label agreement is 413/421,
versus 396/421 for constant-negative, and 331 abstentions give missingness
bounds of 413/752--744/752. The
[canonical JSON](paper/data/h13_v1_forecast_corpus.json) records exact object
identities, strict immutable reconciled-row and matched-block identity validation,
arithmetic replay, threshold and endpoint sensitivity,
and a 2,000-resample matched-block descriptive interval; the
[eight-row CSV](paper/data/h13_v1_forecast_corpus.csv) exposes the fixed
wording-by-direction cells. A frozen 160-row stratified draw is reproduced only
as a private pre-annotation design: zero media/alignment items are verified and
zero human labels are collected. The corrected-fidelity gate remains blocked on
840 required configuration/alignment/trajectory paths and current raw-media
recovery/replay. This older 80-episode/752-chunk/3,008-frame corpus is evaluator
development, not new inference, validated fidelity, annotation or confirmation;
all 13 activity counters remain zero, all nine gates false and the 232-cell core
bound unchanged.

The H01 representative-video deliverable is also packaged without selecting on
outcome: all four original/reflected x LEFT/RIGHT cells at the minimum released
seed 9400 are authenticated directly from the pinned Git commit. The compact
[media manifest](paper/media/representative_h01_videos.json) records every MP4
blob, byte count, SHA-256, historical full-stream qualification and the declared
selection rule. These side-by-side actual/local-prediction videos are
presentation media, not timing maps, physical-time alignment, annotations or
forecast-accuracy evidence.

The separate [decode-verification receipt](paper/media/representative_h01_video_verification.json)
authenticates and fully decodes all four cached MP4 Git blobs without
materializing them. It is publication QA only: the composites expose counterpart
panels and labels, so they are not blinded annotation inputs and advance no
annotation, science or release gate.

- [Research and experiment plan](docs/superpowers/plans/2026-09-12-world-model-paper.md):
  repair existing results, qualify archived recordings, then test whether
  generated futures anticipate executed motion across new object layouts.

- [Paper PDF](paper/main.pdf) and [editable LaTeX](paper/main.tex).
- [Verified paper results](results/paper_results.json).
- [Next experiment](docs/NEXT_EXPERIMENT.md): compare predicted and actual
  object positions at the same physical time.
- [Build and visual checks](docs/DELIVERY_QA.md).

The paper has an abstract, introduction, experimental setup, results, discussion,
limitations and conclusion. The schematic explains what changes between trials;
the result figure shows measured behavior. It contains no invented experiments,
forecast-accuracy claim, or planned annotation study presented as a result.

## Scope

This is a working paper for the evaluation theme of
[Do Robots Need World Models?](https://do-robots-need-world-models.github.io/).
The primary position-reflection experiment studies two world–action
configurations; retained controls include a separate Edge context, and models
are never pooled. H13 is a separate older evaluator audit, not another behavioral
cohort or a forecast-fidelity result. It does not compare world modeling against
its removal. The
final author list and submission details
remain to be settled. Nothing has been submitted or published by this task.

The measurements come from the committed steerability study at
`ce561e66f82e95055e39d3d7711691982f6b2086`. The extractor verifies the Git identity
and records a SHA256 for each source, recomputes counts and paired contrasts, and
reproduces the original numerical analyses. The separate DreamZero symmetry
experiment is retained in the extracted supporting data but is not pooled with
the reflection experiments or included in this paper's trial counts.

The two primary position-reflection layouts have fixed physical resets. Nano
varies policy-sampling seeds; DreamZero-s2 retains its released fixed noise
seed. For those primary cohorts, the manuscript reports DreamZero-s2
descriptively and uses sampling-seed intervals only for Nano. The separate
preregistered H07/H08 retained analysis reports matched-block intervals over
its registered seed blocks; it does not create independent model-noise draws.

## Rebuild

From the repository root, with `uv`, `ffprobe`/`ffmpeg`, a TeX distribution
providing `latexmk`, and the already-qualified any6d environment containing
Torch 2.7.0, create a disposable build environment that inherits Torch and add
the pinned paper-only dependencies:

```sh
WMF_TORCH_PYTHON=/home/ali/miniforge3/envs/any6d/bin/python
WMF_BUILD_ENV=$(mktemp -d /tmp/wmf-paper-build.XXXXXX)
uv venv --python "$WMF_TORCH_PYTHON" --system-site-packages "$WMF_BUILD_ENV"
uv pip install --python "$WMF_BUILD_ENV/bin/python" --requirement workshops/corl2026_world_models/requirements-build.txt
"$WMF_BUILD_ENV/bin/python" workshops/corl2026_world_models/analysis/build_package.py
```

The `--system-site-packages` boundary is intentional: `requirements-build.txt`
does not download or replace Torch, while the regression suite includes
Torch-backed qualification tests. The retained H05--H12 replays use NumPy;
H13 uses the standard library and Git object reads.
`build_package.py` fails before running the suite if any required package is
unavailable and records all five resolved versions in the build report.
Figure and PDF subprocesses receive `SOURCE_DATE_EPOCH=1786718341`, the pinned
study-source commit time, plus `FORCE_SOURCE_DATE=1`; the build report records
both values. The retained PDF still carries TeX-produced wall-clock metadata,
so this environment record is not evidence of byte-identical PDF rebuilding.

This runs the tests, recomputes results, the stopping-time control, the H01
requested-region entry diagnostic, the retained H05/H06 dose responses, the
H07/H08 endpoint interactions, the H09/H10 exploratory wording controls, the
separate H11/H12 Phase-A direct-command cohorts and the retained H13 Edge V1
evaluator audit from pinned Git objects, authenticates and
fully decodes the four representative H01 video blobs, produces figures and
compiles the official CoRL template. The build report records all retained-data
analyses, their source identities and output hashes, plus the representative-
media selection, package and decode-verification hashes. It checks page count,
citations and layout errors. Final page images are reviewed separately.

## Earlier analysis

The prior draft centered on discrepancies in a Cosmos Edge prediction scorer.
That work remains under `docs/EVIDENCE_AUDIT.md`, `docs/ANALYSIS_PROTOCOL.md`
and its associated outputs for reference; it is no longer the paper's main
contribution. Its unrun prediction-validation and annotation plans are not new
empirical results. The previous draft is preserved in commit `8657fa7`.

The user has authorized additional experiments if useful. No new model
experiments have been launched during this rewrite.
