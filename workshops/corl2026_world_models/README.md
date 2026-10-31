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
DreamZero-s2 retains its released fixed noise seed. The manuscript reports
DreamZero-s2 descriptively and uses sampling-seed intervals only for Nano.

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
Torch-backed qualification tests. The retained H05 replay also uses NumPy.
`build_package.py` fails before running the suite if any required package is
unavailable and records all five resolved versions in the build report.
Figure and PDF subprocesses receive `SOURCE_DATE_EPOCH=1786718341`, the pinned
study-source commit time, plus `FORCE_SOURCE_DATE=1`; the build report records
both values. The retained PDF still carries TeX-produced wall-clock metadata,
so this environment record is not evidence of byte-identical PDF rebuilding.

This runs the tests, recomputes results, the stopping-time control, the H01
requested-region entry diagnostic and the retained H05 dose response from the
pinned repository, authenticates and
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
