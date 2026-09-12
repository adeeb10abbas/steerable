# Rewritten paper: delivery verification

Date: 2026-09-12. Status: **working research paper; not submitted**.

## Manuscript and PDF

The paper is now titled **Spatial Instruction Following in World–Action Models**.
It reports the completed Nano and DreamZero position-reflection experiments.
The previous forecast-scoring draft is preserved in commit `8657fa7`; its
supporting analysis remains available but is not presented as the current paper.

`paper/main.pdf` has **four pages total including references**, compiled with the
unchanged official CoRL 2026 style in preprint mode and an anonymous author line.

- PDF SHA256: `8b385fb8f63e2e76f914715223f9f45e5b51673d147554990eca47002c438580`.
- Official style SHA256: `62f38cd8df7ad718796617595c8230da136fc83ea1d7e1ba80084113409ec957`.
- Numerical figure input SHA256: `f4695ae37dfd6b981a133ee10ccd69a2b86988e71d6ff99462bd449189dd94d7`.
- Automated PDF checks passed: four pages, expected result text, resolved
  references, no overfull boxes or LaTeX errors.
- All four final rendered pages (`paper/qa_pages/final_rewrite-1.png` through
  `final_rewrite-4.png`) were visually inspected. Body text, equations, table,
  figures and references are readable. No clipping, overlap or missing glyphs
  was observed. The opening Results paragraph ends before the page break.

The result figure is generated from `results/paper_results.json`. The experimental
design figure is explicitly labeled as a schematic, not a measured trajectory.

## Numerical verification and independent review

The integrated build passed **23 tests**, extracted the paper results, replayed
the preserved historical analysis, regenerated the figures and compiled the PDF.
The logs are `results/validation.log` and `results/build_report.json`.

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
