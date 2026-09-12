# Spatial Instruction Following in World–Action Models

The paper asks a simple question: **when a robot fails a left/right placement
command, did the instruction fail to redirect it, or did the resulting execution
fail in that scene?**

The manuscript now presents completed Cosmos 3 Nano and DreamZero experiments.
Each model has 108 trials across two directions and two fixed object layouts.
Reflection changes DreamZero's completion from 13/54 to 50/54 trials. Nano remains
highly successful, but which requested side produces greater placement depth
changes. Both models generally order their final cube–bowl positions according
to the command.

## Read the paper

- [Exact ablation specification](docs/ABLATION_SPEC.md), including the verified
  existing-experiment inventory, 232 planned core cells and separately listed
  optional guidance extension. All new cells remain unreleased.

The current draft has received a weak-reject internal review. In particular,
its DreamZero results use custom action guidance at scale 2, and its endpoint
response measure needs a stopping-rule control. These revisions are planned,
not yet incorporated into the manuscript.

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
It studies executed behavior of two world–action models. It does not compare
world modeling against its removal. The final author list and submission details
remain to be settled. Nothing has been submitted or published by this task.

The measurements come from the committed steerability study at
`ce561e66f82e95055e39d3d7711691982f6b2086`. The extractor verifies the Git identity
and records a SHA256 for each source, recomputes counts and paired contrasts, and
reproduces the original numerical analyses. The separate DreamZero symmetry
experiment is retained in the extracted supporting data but is not pooled with
the reflection experiments or included in this paper's trial counts.

The two layouts have fixed physical resets. Nano varies policy-sampling seeds;
DreamZero retains its released fixed noise seed. The manuscript reports
DreamZero descriptively and uses sampling-seed intervals only for Nano.

## Rebuild

From the repository root, with `uv` and a TeX distribution providing `latexmk`:

```sh
uv run --no-project --with-requirements workshops/corl2026_world_models/requirements-build.txt python workshops/corl2026_world_models/analysis/build_package.py
```

This runs the tests, recomputes results from the pinned repository, produces
figures and compiles the official CoRL template. It checks page count, citations
and layout errors. Final page images are reviewed separately.

## Earlier analysis

The prior draft centered on discrepancies in a Cosmos Edge prediction scorer.
That work remains under `docs/EVIDENCE_AUDIT.md`, `docs/ANALYSIS_PROTOCOL.md`
and its associated outputs for reference; it is no longer the paper's main
contribution. Its unrun prediction-validation and annotation plans are not new
empirical results. The previous draft is preserved in commit `8657fa7`.

The user has authorized additional experiments if useful. No new model
experiments have been launched during this rewrite.
