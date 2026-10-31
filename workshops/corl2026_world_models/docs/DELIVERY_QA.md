# Rewritten paper: delivery verification

Date: 2026-10-31. Status: **working research paper; not submitted**.

## Manuscript and PDF

The paper is now titled **Spatial Instruction Following in World–Action Models**.
It reports the completed Nano and DreamZero position-reflection experiments.
The previous forecast-scoring draft is preserved in commit `8657fa7`; its
supporting analysis remains available but is not presented as the current paper.

`paper/main.pdf` has **four pages total including references**, compiled with the
unchanged official CoRL 2026 style in preprint mode and an anonymous author line.

- PDF SHA256: `1d93d5541485ec76accc67d79c2f48b59d607db37cfcba33a417557d9f2bdfcc`.
- LaTeX source SHA256: `427712c4ee02eb3e73d06001e8764d245a29395c8f804e6776a80af483b7ba8e`.
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
2.2.6, matplotlib 3.10.9, pypdf 6.18.1 and jsonschema 4.26.0. The active
protected-worktree discovery passed **994/994 tests** before H06; after the 15
H06 and five H07/H08 checks were added, the current discovery passed
**1014/1014 tests in 120.479 s** (**122.87 s** wall);
seven of those methods
belong to the protected untracked N3 test and are not package
files. An earlier H05-era clean isolated-worktree invocation passed the
authorized package's **987/987 tests in 119.102 s**. The current clean isolated
build's regression stage passed **1007/1007 tests in 121.302 s**; its full
thirteen-step build exited zero in **132.94 s**. Those steps are tests, five
retained-data analyses, media packaging and four full MP4 decodes plus
verification, paper results, historical replay, the private annotation draw,
figures and the official-template PDF.

Figure and PDF subprocesses receive `SOURCE_DATE_EPOCH=1786718341` for the
retained study-source commit and `FORCE_SOURCE_DATE=1`, and the build report
records both values. The retained PDF still has TeX-produced wall-clock
metadata, so these settings do not establish byte-identical PDF rebuilding.
TeX Live 2026 latexmk 4.88 and pdfTeX 1.40.29 validated the previous four-page
package build. The current H09/H10 manuscript was rebuilt from the same official
template with cached Tectonic 0.17.0, then checked as four pages with zero
errors, overfull boxes, undefined citations or undefined references. Its four
rendered pages passed visual review. The exact current publication hashes are
recorded above; this does not claim byte-identical rebuilding across TeX engines.

`results/validation.log` and `results/build_report.json` are generated build
products, not retained checkout evidence in this protected worktree. The
current isolated validation log is 315,693 bytes with SHA256
`6462817ccf27c19b3b35aaef41385c48e36a8c17fd137df24ea9aec82fc6dafb`;
the 75,398-byte build report has SHA256
`37cf5c0d6fbce7eae0eaa6cb30f35a7f5ebb2ca0ffcae8c12687d8b38df2c91a`.
The isolated build report was independently inspected before publication and
binds the dependency versions, four-page pdfTeX PDF SHA256
`066abe8af0f31a4bfc6e73cf78927c51d804113e12b79749fc09bf062254fc1e`,
retained-analysis hashes and media zero-activity counters. Durable inputs and
receipts are instead checked in under
`paper/data/` and `paper/media/`; the execution status/progress ledger records
the qualification receipt without recreating the protected deleted results.

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
- The four representative seed-9400 videos are authenticated and fully decoded,
  but remain presentation media rather than blinded annotation inputs, timing
  maps, forecast-accuracy evidence or a release-gate advance.

## Remaining research and submission work

The main results cover two checkpoints, one task and two fixed physical layouts;
H05 adds seven positions within one fixed simulator setup, not seven scenes.
H06 adds four inventory-matched positive symmetry levels within 27 repeated
seed blocks, not 216 independent scenes. H07/H08 add two endpoint-only model
contexts over repeated seed blocks; their level-zero inventory differs from
level one. H09/H10 reuse 20 seeds across eight wording/direction cells per model
in one neutral scene; they are not 320 independent scenes or confirmatory
wording evidence.
They establish neither prediction accuracy nor the causal benefit of world modeling.
They also do not isolate visibility, reachability or grasp geometry as the mechanism.

`docs/NEXT_EXPERIMENT.md` specifies an eight-trial recording check followed by a
comparison of generated futures with actual outcomes at matching times. It has not
been launched. New layouts, qualified model-noise sampling and independent image
labels remain future work. No new model inference, remote compute or human labeling
was performed during the rewrite.

Final authorship, publication eligibility and the workshop's final submission rules
remain to be confirmed before submission. No submission or public release occurred.
