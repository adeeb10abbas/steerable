# Robot world-model evaluation — CoRL 2026 workshop

Working research package for [Do Robots Need World Models?](https://do-robots-need-world-models.github.io/), primarily **Theme 6: evaluation and benchmarks**, with a secondary connection to Theme 4: model-based policy evaluation. The target is **Best Research Paper**. Awards cannot be predicted; the immediate work is to establish a contribution that withstands independent scrutiny.

The package is being built from immutable source commit `ce561e66f82e95055e39d3d7711691982f6b2086` on the local branch `research/corl2026-world-model-evaluation`.

## Reading order

1. [Four-page working manuscript](paper/main.pdf), including references, with [editable LaTeX](paper/main.tex).
2. [Evidence audit](docs/EVIDENCE_AUDIT.md) — what the available source data establishes.
3. [Analysis protocol](docs/ANALYSIS_PROTOCOL.md) — shared timing, spatial definitions, baselines, and independent annotation.
4. [Award strategy](docs/AWARD_STRATEGY.md) — the evidence needed for a competitive empirical paper.
5. [Literature and novelty](docs/LITERATURE_AND_NOVELTY.md) — nearest work and defensible contribution.
6. [Data recovery](docs/DATA_RECOVERY.md) and [model/runtime provenance](docs/MODEL_PROVENANCE.md).
7. [Annotation handoff](docs/ANNOTATION_HANDOFF.md) — private 160-chunk draw; no human labels collected.

## Research question

**Does the generated future predict what the robot will change better than the current observation alone?**

The completed audit reproduces all 752 historical rows and 3,008 cached frames.
Agreement is 413/421 (98.1%) on the historically scorable subset; an always-negative
label reaches 396/421 (94.1%) on that same subset. Only 25/97 historically positive
execution endpoints have a scorable forecast. Forecast and execution scoring use
different time aggregation, object geometry and coordinate conventions. These
are descriptive audit results, **not independently validated prediction fidelity**.

The private sampling manifest selects 160 chunks across six historical strata,
with inverse-inclusion weights representing all 752 chunks. It includes no new
annotations and must never be given to raters. Its 74 represented episodes leave
few episode-disjoint V1 holdouts if the labels are used for scorer tuning; the
independent future-cohort validation option is therefore material.

## What comes next

1. Recover original files and verify their recorded hashes: 80 trajectories,
   752 chunk metadata records, 8 camera/configuration files, and supporting media.
2. Establish the same physical time, view, and object definition on forecast,
   conditioning, and execution images. Freeze the measurement rubric before labels.
3. Obtain independent blinded labels, then compare the forecast with current-state
   persistence, retaining unknown cases and shared-seed dependence.
4. If the measurement is reproducible, validate the answer on a separately frozen
   second WAM cohort. Report a positive, negative, or inconclusive result faithfully.

Actual execution imagery is not yet established. The recovery checklist covers
RGB inside the trajectories, original execution recordings, or a later chunk's
conditioning image **only if its physical time matches the scored endpoint**.
The current manifest alone does not establish any of these routes.

The corpus was already inspected. This is a **retrospective amendment**, not a
preregistration of unseen V1 results. Corrected fidelity, human validation and
downstream control utility remain pending. No inference, remote execution,
recruitment, spending, or submission has been performed for this package.

## Reproduce locally

From the repository root, the evidence replay and sampling tests need only Python:

```sh
python3 -m unittest discover -s workshops/corl2026_world_models/tests -v
python3 workshops/corl2026_world_models/analysis/evidence_audit.py
python3 workshops/corl2026_world_models/analysis/prepare_annotation_sample.py
```

To rebuild everything, install the optional pinned figure/PDF dependencies into
an isolated environment and use a TeX distribution providing `latexmk`. With `uv`:

```sh
uv run --no-project --with-requirements workshops/corl2026_world_models/requirements-build.txt python workshops/corl2026_world_models/analysis/build_package.py
```

The build replays evidence, runs tests, regenerates the private sample and figure,
compiles the unmodified official CoRL 2026 style, and checks page count and unresolved
references. Its receipt is [build_report.json](results/build_report.json). Visual
inspection is separately recorded in [DELIVERY_QA.md](docs/DELIVERY_QA.md).

The official workshop page currently lists up to four pages, with OpenReview link,
deadline and archival status still TBD. The draft conservatively uses four pages
total including references. Author list, workshop anonymity requirements and final
submission details need confirmation when the submission site becomes available.

Scope and authorized boundaries are recorded in [SCOPE_AND_PLAN.md](docs/SCOPE_AND_PLAN.md).
