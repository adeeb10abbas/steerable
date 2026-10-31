# Rewritten paper: delivery verification

Date: 2026-10-31. Status: **working research paper; not submitted**.

## Manuscript and PDF

The paper is now titled **Spatial Instruction Following in World–Action Models**.
It reports the completed Nano and DreamZero position-reflection experiments.
The previous forecast-scoring draft is preserved in commit `8657fa7`; its
supporting analysis remains available but is not presented as the current paper.

`paper/main.pdf` has **four pages total including references**, compiled with the
unchanged official CoRL 2026 style in preprint mode and an anonymous author line.

- PDF SHA256: `cfcd56eff4f8cfd22dff28cef6ae10c45d1549ad926323f580e1e454a9461c58`.
- LaTeX source SHA256: `59f64dea771fe12781ea30b5c1d76a54ee8799bea0b48625b6d524a45070522b`.
- Official style SHA256: `62f38cd8df7ad718796617595c8230da136fc83ea1d7e1ba80084113409ec957`.
- Numerical figure input SHA256: `f437fec607ea969a4c0293b157a74175c410b5f88709ac73262b3035314f1230`.
- Design figure PDF/PNG SHA256: `9896dafc73897ca19fe6558d9385a8b1512cae858ba815c88743642185ea8201` / `9f1dbac8c32485ebdbaa0df80f8dc1b40c36c9369fd118c700c4e591e4cd94df`.
- Results figure PDF/PNG SHA256: `352fc07d916fe0eb69811189ce5a33d3d4b6250ff2b5406616588d92efffcbd2` / `6f1eacb1fccc77a7b70e7b1d393816d9e87c5b570c29384e38d322ebd7ddde98`.
- Automated PDF checks passed: four pages, expected result text, resolved
  references, no overfull boxes or LaTeX errors.
- All four pages were rendered to temporary PNGs and visually inspected. The
  review renders are intentionally not checked in or cited as persistent files.
  Body text, equations, table, figures and references are readable, with no
  clipping, overlap or missing glyphs. The requested-region paragraph continues
  naturally from page 3 to page 4 without an incomplete sentence.

The result figure is generated from `results/paper_results.json`. The experimental
design figure is explicitly labeled as a schematic, not a measured trajectory.

## Numerical verification and independent review

The exact documented disposable environment uses Python 3.10.21, inherits
Torch 2.7.0+cu128 from the qualified any6d environment, and resolves NumPy
2.2.6, matplotlib 3.10.9, pypdf 6.18.1 and jsonschema 4.26.0. The clean
isolated pre-build discovery passed **1030/1030 tests in 129.990 s**
(**132.87 s** wall). The authoritative isolated build then passed all 16 stages,
including another **1030/1030 tests in 129.223 s**, in **144.25 s** wall time
with a peak resident set size of **908,820 KiB**. The post-merge protected-tree
discovery then passed **1037/1037 tests** in **126.000 s** (**128.43 s** wall;
peak resident set size **905,792 KiB**); seven of those methods belong to the protected untracked
N3 test and are not package files. The next qualified build will contain
seventeen steps; those steps will cover tests, retained-data analyses, media
packaging and four full MP4
decodes plus verification, paper results, historical replay, the private
annotation draw, figures and the official-template PDF.

Figure and PDF subprocesses receive `SOURCE_DATE_EPOCH=1786718341` for the
retained study-source commit and `FORCE_SOURCE_DATE=1`, and the build report
records both values. The retained PDF still has TeX-produced wall-clock
metadata, so these settings do not establish byte-identical PDF rebuilding.
The pinned TeX Live 2026 latexmk 4.88 and pdfTeX 1.40.29 toolchain produced both
the checked-in manuscript and the clean isolated four-page H11--H13 package with
the exact PDF SHA256 recorded above. The build had zero errors, overfull boxes,
undefined citations or undefined references. All four isolated pages were
rendered and passed visual review.

`results/validation.log` and `results/build_report.json` are generated build
products, not retained checkout evidence in this protected worktree. The
authoritative isolated H13 validation log is 466,739 bytes with SHA256
`798863915dbe72bb336842cb6ed311f2e895a6e630918ff8f54885a3dc9eff76`;
its 138,203-byte build report has SHA256
`6545f8bc8aa9ac3425d81288191b85a0820a5298ec757bc8d080daac076b3790`.
Independent inspection confirmed that the report binds the exact environment,
all 16 successful stages, the four-page PDF above, the eight retained-analysis
receipts including H13, and H13's zero media/alignment/label counts. The
H03/H04 integration must be requalified before these historical counts and
hashes are replaced; no unrun build is presented as complete. Durable
inputs and receipts are checked in under `paper/data/` and `paper/media/`; the
execution status/progress ledger records this qualification receipt without
recreating the protected deleted results.

The extractor verifies nine source Git blobs at upstream commit
`ce561e66f82e95055e39d3d7711691982f6b2086`. It recomputes counts and continuous
measurements from episode rows and reconciles them against the source summaries.
It extracts three separate 108-episode cohorts; the two position-reflection
cohorts, 216 episodes total, provide the primary manuscript experiment. The
separate DreamZero symmetry experiment is not pooled with either reflection
experiment. H05 is separately authenticated from its registered report and is
not pooled into those trial counts.

An independent reviewer checked all eight success counts, the instruction-response
and placement-depth means, intervals and measurement definitions against the pinned
episode rows. The final manuscript incorporates the following corrections:

- Instruction response compares final cube–bowl offsets, not absolute cube endpoints.
- The success cone extends 45 degrees on either side of the requested direction.
- Continuous position measurements include failed trials.
- The commands and 450-action limit are stated in the experimental setup.
- For the primary position-reflection cohorts, DreamZero results are descriptive
  because its effective model-noise seed is fixed; only Nano receives bootstrap
  confidence intervals in the primary paper figure. The separate preregistered
  retained analyses report matched-block intervals over their registered seed
  blocks; they do not create independent model-noise draws.
- DreamZero's original 41 failures are described by the recorded execution checks:
  26 pickup, 14 transport and one wrong-side placement. These are not diagnoses
  of language understanding.
- The H01 requested-region diagnostic keeps four reflected-RIGHT no-entry traces
  cap-censored, reports 104 entries and the matched entry-index interaction, and
  does not turn action indices into physical time or goal-terminated transients
  into common-horizon sustained entries.
- The retained H03/H04 package byte-authenticates seven compact Git objects
  (810,860 bytes/18,939 lines) and rejoins six baseline/intervention cells for
  each model. Nano changes from g3 6/6 to g1 4/6; DreamZero changes from the
  conditional-action-equivalent s1 baseline 3/6 to custom negative-branch s2
  4/6. All six paired-seed records independently reproduce and source-validate
  endpoint ordering, separation and right-minus-left requested-margin gaps; all
  three command pairs remain endpoint ordered per configuration, but
  every cell shares one retained reset fingerprint and DreamZero fixes effective
  model-noise seed 1140. The pilots are post-result, exploratory and descriptive;
  their historical baselines are not double counted, the models are not pooled,
  and H04's historical custom-s2 arm neither selects a new-confirmation D2
  extension nor qualifies D1. Original raw files remain external
  and were not replayed. No new inference, episode, annotation, confirmation,
  release gate or change to the 232-cell core follows.
- The retained H05 package authenticates a 15-seed by seven-level by two-command
  design, reproduces the 1.125 m/m registered slope and its 20,000-resample
  matched-seed interval, and preserves the nonmonotone level means and
  out-of-support fitted crossing. Its 315 raw-source identities are retained in
  the manifest, but the raw bytes are outside this checkout and were not
  independently rehashed here. The 210 episodes and 1,423 requests are
  authenticated historical-population counts, not current-turn activity; the
  H05 receipt records all 13 current activity counters as zero and all nine
  release gates as false.
- The retained H06 package authenticates the prospectively registered
  V3-E004 compact tables and independently rejoins the 216-episode/108-pair
  inventory-matched Nano grid. It reproduces the mean within-seed depth-gap
  slope of 0.020 m per unit realised asymmetry, its 20,000-resample interval
  and exact sign test. Level zero remains excluded because its object inventory
  differs. Compact Git rows are rehashed, but original raw simulator files are
  external; H06 is not independent-scene, position-reflection, equivalence,
  annotation, forecast accuracy or release-gate evidence. Its receipt likewise
  records all 13 activity counters as zero and all nine gates as false.
- The retained H07/H08 endpoint package independently rejoins 108 episodes and
  54 LEFT/RIGHT pairs for each of DreamZero-s2 and Cosmos Edge over the exact
  27-seed, two-level registered grids. It reproduces the registered
  $s=1$-minus-$s=0$ depth, binary-success and endpoint-redirection
  interactions, their 20,000-resample intervals and exact layout-label
  permutations. The level-zero inventory omits companion objects, so these are
  separate inventory-transition contexts rather than position-reflection
  replications. Compact rows are rehashed; original raw simulator files remain
  external. No equivalence, forecast-accuracy, annotation, confirmation or
  release-gate claim follows, and all current activity/gate counters remain
  zero/false.
- The retained H09/H10 wording package byte-authenticates the frozen 480-row
  three-checkpoint V3-C001 registration and all 320 ABLATION_SPEC-selected
  Nano/Edge compact episode rows. It reconstructs 120 same-seed direct-control
  contrast rows without pooling models or exporting raw PVC paths; the excluded
  GR00T cohort remains explicitly disclosed. Nano/Edge successes are 134/160
  and 123/160; historical requests/failures are 1,234/26 and 1,365/37. The
  requested-margin and endpoint-shift views are algebraically paired rather
  than independent signals. Confirmatory wording inference was not registered,
  so all contrasts remain exploratory descriptive repeated-block evidence. No new
  inference, episode, annotation, confirmation or release gate follows, and the
  232-cell core bound remains unchanged.
- The retained H11/H12 Phase-A package byte-authenticates the exact 54
  authorized-new direct-command cells and 27 matched LEFT/RIGHT reset blocks
  for each model while excluding six preserved V2 cells per model. Its eight
  payload objects total 5,636,174 bytes; two design/overlap helpers bring the
  authenticated boundary to ten objects and 5,654,967 bytes. Nano
  succeeds in 51/54 and reconciles 349 decoded RGB futures/action chunks;
  DreamZero-s2 succeeds in 3/27 LEFT and 17/27 RIGHT and reconciles 2,554
  latent futures but only 54 official full-reset decodes and 54 rollout videos.
  The models and interfaces remain separate, fixed model-noise seed 1140 is not
  an independent draw, and H12's numerically matching H07 directional totals
  arise from identity-distinct seed cohorts. Original raw Phase-A payload bytes
  remain external and were not revalidated. No raw recovery, forecast accuracy,
  annotation, confirmation, new activity, gate advance or change to the
  232-cell core bound follows.
- The retained H13 evaluator-development package byte-authenticates five audit
  objects (998,249 bytes) and reopens all 761 ledger-listed pinned upstream
  source Git blobs (7,230,812 bytes; 752 cache files). It strictly validates
  immutable reconciled rows and matched-block identities for 80 episodes,
  752 replan chunks, 3,008 cached forecast frames and 20 four-episode blocks,
  then independently replays the registered arithmetic.
  The historical scorer labels 421/752 chunks but covers only 25/97
  executed-positive horizons. Its same-covered-subset apparent historical-label
  agreement is 413/421 versus 396/421 for constant-negative; 331 abstentions
  imply historical-label missingness bounds of 413/752--744/752, not a fidelity
  interval. The endpoint/threshold and 2,000-resample matched-block summaries are
  descriptive evaluator diagnostics only. A frozen 160-row stratified sampling
  design reproduces exactly, but verified media/alignment items and collected
  human labels remain 0/0/0. Corrected fidelity remains blocked on 840 required
  configuration/alignment/trajectory paths plus current raw-media recovery/replay. No new
  model inference, behavioral episode, fidelity estimate, annotation or
  confirmation gate follows: all 13 activity counters are zero, all nine gates
  false and the 232-cell core ceiling unchanged.
- The four representative seed-9400 videos are authenticated and fully decoded,
  but remain presentation media rather than blinded annotation inputs, timing
  maps, forecast-accuracy evidence or a release-gate advance.

## Remaining research and submission work

The main results cover two checkpoints, one task and two fixed physical layouts;
H03/H04 reuse one retained physical reset across three seed labels and two
commands per configuration; they are not independent scenes or confirmatory
guidance evidence. H05 adds seven positions within one fixed simulator setup,
not seven scenes.
H06 adds four inventory-matched positive symmetry levels within 27 repeated
seed blocks, not 216 independent scenes. H07/H08 add two endpoint-only model
contexts over repeated seed blocks; their level-zero inventory differs from
level one. H09/H10 reuse 20 seeds across eight wording/direction cells per model
in one neutral scene; they are not 320 independent scenes or confirmatory
wording evidence. H11/H12 each reuse 27 neutral-scene reset labels across two
directions; they are separate model-specific development/recovery cohorts, not
54 independent scenes, a cross-model experiment or recovered raw evidence.
H13 is an older evaluator audit with missing raw alignment/media prerequisites,
not an 80-scene behavioral study, a fidelity estimate or annotation completion.
They establish neither prediction accuracy nor the causal benefit of world modeling.
They also do not isolate visibility, reachability or grasp geometry as the mechanism.

`docs/NEXT_EXPERIMENT.md` specifies an eight-trial recording check followed by a
comparison of generated futures with actual outcomes at matching times. It has not
been launched. New layouts, qualified model-noise sampling and independent image
labels remain future work. No new model inference, remote compute or human labeling
was performed during the rewrite.

Final authorship, publication eligibility and the workshop's final submission rules
remain to be confirmed before submission. No submission or public release occurred.
