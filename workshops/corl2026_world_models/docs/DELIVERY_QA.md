# Rewritten paper: delivery verification

Date: 2026-10-31. Status: **working research paper; not submitted**.

## Manuscript and PDF

The paper is now titled **Spatial Instruction Following in World–Action Models**.
It reports the completed Nano and DreamZero position-reflection experiments.
The previous forecast-scoring draft is preserved in commit `8657fa7`; its
supporting analysis remains available but is not presented as the current paper.

`paper/main.pdf` has **four pages total including references**, compiled with the
unchanged official CoRL 2026 style in preprint mode and an anonymous author line.

- PDF SHA256: `1d1f0ecbd3c54b6110590d361dc6d9d5c2f55a0fd15fc920723c505284ec72b8`.
- LaTeX source SHA256: `da75322dcdc14d1b78a26c88c4812d9a560f4c46e8c6ebffe316822442dfab87`.
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
Torch 2.7.0+cu128 from the qualified any6d environment, and layers matplotlib
3.10.9, pypdf 6.18.1 and jsonschema 4.26.0. The active protected-worktree
discovery passed **985/985 tests** in 125.701 seconds (128.11 seconds wall,
95.35 seconds user, 13.80 seconds system, 838,388 KiB peak RSS); seven of those methods
belong to the protected untracked N3 test and are not package files. A clean
isolated-clone invocation passed the authorized package's **978/978 tests** in
125.507 seconds and completed all ten `analysis/build_package.py` steps in
135.59 seconds: tests,
both retained-data analyses, media packaging and four full MP4 decodes, paper
results, historical replay, the private annotation draw, figures and the
official-template PDF.

Figure and PDF subprocesses pin `SOURCE_DATE_EPOCH=1786718341` to the retained
study-source commit and set `FORCE_SOURCE_DATE=1`. Two complete isolated-clone
builds reproduced byte-identical figure PDF/PNG files and `paper/main.pdf`; both
12,944-byte build logs are identical at SHA256
`2f3acaca524a6041a08624ef11338ae93b5207aa6b07e34a84c5a5de88477958`, and the
exact post-fix publication hashes are recorded above.

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
It extracts three separate 108-episode cohorts; only the two position-reflection
cohorts, 216 episodes total, enter the manuscript. The separate DreamZero symmetry
experiment is not pooled with either reflection experiment.

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
- The four representative seed-9400 videos are authenticated and fully decoded,
  but remain presentation media rather than blinded annotation inputs, timing
  maps, forecast-accuracy evidence or a release-gate advance.

## Remaining research and submission work

The current results cover two checkpoints, one task and two fixed physical layouts.
They establish neither prediction accuracy nor the causal benefit of world modeling.
They also do not isolate visibility, reachability or grasp geometry as the mechanism.

`docs/NEXT_EXPERIMENT.md` specifies an eight-trial recording check followed by a
comparison of generated futures with actual outcomes at matching times. It has not
been launched. New layouts, qualified model-noise sampling and independent image
labels remain future work. No new model inference, remote compute or human labeling
was performed during the rewrite.

Final authorship, publication eligibility and the workshop's final submission rules
remain to be confirmed before submission. No submission or public release occurred.
