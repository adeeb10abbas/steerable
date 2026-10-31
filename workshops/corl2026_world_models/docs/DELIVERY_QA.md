# Rewritten paper: delivery verification

Date: 2026-10-31. Status: **working research paper; not submitted**.

## Manuscript and PDF

The paper is now titled **Spatial Instruction Following in World–Action Models**.
It reports the completed Nano and DreamZero position-reflection experiments.
The previous forecast-scoring draft is preserved in commit `8657fa7`; its
supporting analysis remains available but is not presented as the current paper.

`paper/main.pdf` has **four pages total including references**, compiled with the
unchanged official CoRL 2026 style in preprint mode and an anonymous author line.

- PDF SHA256: `974b18749c838c270f34a3709e177603774e8c4df26876e3504a90c70db02644`.
- LaTeX source SHA256: `eeb85d516cb2dbfb8c2b25fc480fe4595dee0e02410c6197e2266432f9fcdd85`.
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
protected-worktree discovery passed **994/994 tests** before H06; with the 15
H06 checks added, the current discovery passed **1009/1009 tests in 123.884 s**;
seven of those methods
belong to the protected untracked N3 test and are not package
files. A clean isolated-worktree invocation passed the authorized package's
**987/987 tests in 119.102 s**. The documented package build exited zero in
**129.21 s** and covers eleven
steps: tests, three retained-data analyses, media packaging and four full MP4
decodes, paper results, historical replay, the private annotation draw, figures
and the official-template PDF.

Figure and PDF subprocesses receive `SOURCE_DATE_EPOCH=1786718341` for the
retained study-source commit and `FORCE_SOURCE_DATE=1`, and the build report
records both values. The retained PDF still has TeX-produced wall-clock
metadata, so these settings do not establish byte-identical PDF rebuilding.
TeX Live 2026 latexmk 4.88 and pdfTeX 1.40.29 validated the previous four-page
package build. The current H06 manuscript was rebuilt from the same official
template with cached Tectonic 0.17.0, then checked as four pages with zero
errors, overfull boxes, undefined citations or undefined references. Its four
rendered pages passed visual review. The exact current publication hashes are
recorded above; this does not claim byte-identical rebuilding across TeX engines.

`results/validation.log` and `results/build_report.json` are generated build
products, not retained checkout evidence in this protected worktree. The
isolated build report was independently inspected before publication and bound
the dependency versions, four-page PDF hash, retained-analysis hashes and media
zero-activity counters. Durable inputs and receipts are instead checked in under
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
- DreamZero results are descriptive because its effective model-noise seed is fixed;
  only Nano receives bootstrap confidence intervals in the paper and figure.
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
- The four representative seed-9400 videos are authenticated and fully decoded,
  but remain presentation media rather than blinded annotation inputs, timing
  maps, forecast-accuracy evidence or a release-gate advance.

## Remaining research and submission work

The main results cover two checkpoints, one task and two fixed physical layouts;
H05 adds seven positions within one fixed simulator setup, not seven scenes.
H06 adds four inventory-matched positive symmetry levels within 27 repeated
seed blocks, not 216 independent scenes; its level-zero inventory differs.
They establish neither prediction accuracy nor the causal benefit of world modeling.
They also do not isolate visibility, reachability or grasp geometry as the mechanism.

`docs/NEXT_EXPERIMENT.md` specifies an eight-trial recording check followed by a
comparison of generated futures with actual outcomes at matching times. It has not
been launched. New layouts, qualified model-noise sampling and independent image
labels remain future work. No new model inference, remote compute or human labeling
was performed during the rewrite.

Final authorship, publication eligibility and the workshop's final submission rules
remain to be confirmed before submission. No submission or public release occurred.
