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

The retained H02 V3-B003 cohort now has a dedicated compact-source audit. It
byte-authenticates nine immutable Git objects (831,259 bytes/1,429 lines),
strictly rejoins all 108 registered custom DreamZero-s2 episodes into 54
LEFT/RIGHT pairs and independently reproduces the 13/54 to 50/54 layout
completion shift, failure taxonomy and registered endpoint/depth interactions.
The post-result descriptive, previously unpackaged compact requested-cone
fields record 13/54 original and 53/54 reflected entry events. Of 66 entries,
63 have an observed three-sample sustained run; the other three are successful
goal-terminated entries with no observed three-sample follow-up, not classified
transient exits. All 42 no-entry traces are valid failures cap-censored at
action 450. The authenticated bridge defines the 45-degree cone and
three-sample rule. These action indices are upstream recorded fields, not
physical time or a rederivation from raw XYZ.
Its [canonical JSON](paper/data/h02_position_reflection.json) inventories 540
distinct raw path/SHA descriptors (237,163,185 declared bytes), while the
[27-row CSV](paper/data/h02_position_reflection.csv) exposes all four cells and
entry/censor contrasts for each matched environment-label block. Those external
PVC bytes were not opened or rehashed. H02 uses fixed effective model-noise seed 1140 and custom
negative-branch action guidance s2; it is neither default DreamZero nor the
official conditional-action D1 configuration. This is retained fixed-scene
behavior, not current raw recovery, physical-time/forecast accuracy, new
inference, annotation, confirmation or release evidence; all activity counters
remain zero, all gates false and the 232-cell core bound unchanged.

The retained H03/H04 V2-A015 guidance pilots are now packaged from seven
byte-authenticated compact Git objects (810,860 bytes/18,939 lines). Each pilot
compares six historical baseline cells with six post-result intervention cells
over seed labels 8300--8302 and both commands. Nano changes from g3 6/6 to g1
4/6; DreamZero changes from the conditional-action-equivalent s1 baseline 3/6
to custom negative-branch action-guidance s2 4/6. All three LEFT/RIGHT pairs
remain endpoint ordered in every configuration, but all cells share one
retained physical-reset fingerprint and DreamZero fixes effective model-noise
seed 1140. The [canonical JSON](paper/data/h03_h04_guidance_pilots.json)
records the exact source identities, baseline/intervention joins, six
source-validated paired endpoint ordering/separation/margin-gap diagnostics,
historical fixed-input gate record and claim boundaries; the
[12-row CSV](paper/data/h03_h04_guidance_pilots.csv) exposes one matched
configuration contrast per model, seed label and command. These are
exploratory post-result pilots, not powered or general guidance evidence; the
historical baselines are not counted again, models are not pooled, and H04's
historical custom-s2 arm does not select a new-confirmation D2 extension or
qualify D1. Referenced raw files
remain external and were not replayed. No current inference, episode,
annotation, confirmation, release gate or change to the 232-cell core follows.

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

The fail-closed [retained historical evidence coverage audit](paper/data/retained_historical_evidence_coverage.json)
authenticates 50 exact Git objects and maps all 13 historical inventory IDs to
nine grouped analysis packages, their focused tests, builder stages and
manuscript markers. Its [13-row crosswalk](paper/data/retained_historical_evidence_coverage.csv)
keeps H01/H02/H07 primary-paper, supplemental and presentation roles explicit
without counting those views as additional cohorts or summing episode counts.
H02/H04/H07/H12 are identified as historical custom DreamZero-s2, not qualified
official conditional-path D1.
The pinned inventory's earlier "not yet packaged" H01 review note and the
checked-in generated build report are retained historical snapshots, not current
coverage evidence, and are not rewritten. Package coverage is neither pooled
inference nor confirmation: raw/PVC recovery, physical-time pairing, independent
labels, qualified official D1 predictions and prospective varied-layout forecast
accuracy remain missing; activity stays zero, gates false and the 232-cell bound
unchanged.

A post-snapshot [H01 request-window supplement](paper/data/h01_request_window_motion.json)
adds a same-H01 execution-motion view over 738 recorded request descriptors and
108 already-counted cells; it is not a tenth grouped cohort. The Turn2909
coverage artifact above remains a sealed pre-T2910 base-snapshot crosswalk and
does not authenticate this additional analytical role. The supplement checks
recorded descriptor metadata but does not reopen forecast payload bytes, score
forecast accuracy, advance A3--A9, add activity, open a gate or change the
232-cell core bound.

The same supplement now makes LEFT/RIGHT counterpart selection explicit. It
retains 114 control and 113 reflected exact +32-action window pairs; 101 and 77
otherwise-complete single-command records lack an exact counterpart because
the paired episode terminates before their +32 endpoint (43 terminate within
that counterpart window and 135 start at or after the counterpart end).
Equal-seed signed lateral means are positive/tied/
negative for 26/1/0 control and 27/0/0 reflected seed labels, while the first
0--32 window is exactly tied in all 54 pairs. Raw window signs are mixed and a
same-request-index four-condition audit gives only 15/11/1 positive/negative/
tied reflected-minus-control seed differences. These post-hoc values telescope
to an outcome-selected request boundary divided by a variable number of
windows; they are not an independent endpoint, physical-time rate, movement
threshold, instruction-understanding result, forecast score or confirmation.

The fixed request-index profile removes that variable-window averaging without
filling stopped traces. Both layouts retain all 27 paired seeds through the
32--64 action window, where signed LEFT-minus-RIGHT window contrasts are
positive/negative/tied for 2/6/19 control and 3/8/16 reflected. At 64--96
actions support is 26/27 and 27/27, with signs 20/6/0 and 25/1/1; the exact
same-index four-condition interaction has 26 seeds and signs 15/11/0. Support
then falls to 18/27 and 17/27 at 96--128 and at most eight seeds per layout
thereafter. Recorded request indices continue through 14, but indices 10--14
have zero paired exact endpoints and are retained explicitly as 0/27 support
with null summaries. Every row reports full-27-seed strict-positive missingness bounds.
This is selection-transparent historical action-index evidence, not physical
time, independent windows, a movement threshold, A3--A9 or forecast accuracy.

The same fixed-index risk sets are also partitioned by the frozen terminal
LEFT/RIGHT pair outcomes, with every zero-population and zero-support stratum
reported explicitly. At request index 3, control support is 16
success/success pairs plus its two mixed-outcome pairs, while reflected support
is 13 success/success plus all four success-left/cap-failure-right pairs. One
of those four reflected mixed-outcome seeds remains observable through indices
4--9 and is the only reflected seed at indices 8--9; both control mixed-outcome
seeds disappear after index 3. This audit describes which historical labels
remain in each risk set only. Terminal outcomes are retrospective post-outcome
strata, availability is not censoring-independent, and no outcome-stratum
motion effect, causal moderation, language/model diagnosis, forecast score,
new cohort or gate advance is inferred.

The first-unavailability decomposition now accounts for the exact point and
trace that removes each seed from those risk sets. Control first losses across
request indices 0--14 are 0,0,1,8,10,3,4,0,0,1,0,0,0,0,0; reflected losses
are 0,0,0,10,12,1,2,0,1,0,1,0,0,0,0. The limiting command is
LEFT-only/both/RIGHT-only for 13/3/11 control pairs and 19/6/2 reflected pairs. All 63
per-layout limiting-cell appearances, including the limiters tied in nine
pairs, are success-coded: 61 contain an explicit partial
goal-stopped request and control-LEFT seeds 9406 and 9416 end successfully
exactly at action 128 without a request at index 4. The four-condition risk set
first loses 1/14/10/2 seeds at indices 2/3/4/6. These first-loss arrays are
discrete differences of the already reported support curves; only limiter
identity and terminal form are newly exposed, not an independent endpoint,
estimand, cohort or scientific signal. Successes stop at the historical goal,
whereas failures remain observed to the 450-action administrative cap, and no
pair has two failures. That structure mechanically produces zero limiting
cap-failure traces; it is not an outcome association and establishes neither
censoring independence nor failure enrichment, a motion effect, physical time,
forecast skill, language/model diagnosis, a new cohort or a gate advance.

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
configurations; retained controls include post-result H03/H04 guidance pilots
and a separate Edge context, and models are never pooled. H04's historical
custom-s2 arm is retained but does not select a new-confirmation D2 extension,
and H13 is a separate older evaluator audit, not
another behavioral cohort or a forecast-fidelity result. The paper does not
compare world modeling against its removal. The
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
Torch-backed qualification tests. The retained H03/H04 package and H05--H12 replays
use pinned Git objects; the numerical replays use NumPy, while H13 uses the standard
library and Git object reads.
`build_package.py` fails before running the suite if any required package is
unavailable and records all five resolved versions in the build report.
Figure and PDF subprocesses receive `SOURCE_DATE_EPOCH=1786718341`, the pinned
study-source commit time, plus `FORCE_SOURCE_DATE=1`; the build report records
both values. The retained PDF still carries TeX-produced wall-clock metadata,
so this environment record is not evidence of byte-identical PDF rebuilding.

This runs the tests, recomputes results, the stopping-time control, the H01
requested-region entry diagnostic, the H02 compact position-reflection audit,
the H03/H04 exploratory guidance pilots,
the retained H05/H06 dose responses, the
H07/H08 endpoint interactions, the H09/H10 exploratory wording controls, the
separate H11/H12 Phase-A direct-command cohorts and the retained H13 Edge V1
evaluator audit from pinned Git objects, rebuilds the retained historical
evidence coverage crosswalk, authenticates and
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
