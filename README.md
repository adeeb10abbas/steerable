# Steerable robotics research

This repository contains three evidence-preserving research tracks:

1. **Active: VLA/WAM language steerability.** Matched LEFT/RIGHT interventions,
   state-derived paths, complete robot videos, and imagined-versus-executed
   future analysis across DROID and RoboTwin.
2. **Bridge ablation feasibility (`RES-1`).** A structural and human-audit gate
   for temporal-language density and surface-form diversity.
3. **GR00T conditional-mutual-information diagnostic.** A fixed-observation
   LIBERO-Spatial prompt-sensitivity probe.

Do not combine results across these tracks. New agents should begin with
[`AGENTS.md`](AGENTS.md) and the active-study
[`continuation handoff`](docs/VLA_WAM_CONTINUATION.md).

## Active VLA/WAM steerability study

The current evidence includes two completed DROID reference checkpoints, three
standardized RoboTwin WAM direct gates, and a standardized π0-FAST DROID direct
gate. All four new-model gates show a one-direction-only competence split, so
the next authorized experiments are directional confirmations—not wording
sweeps.

Primary entrypoints:

- [`VLA_WAM_CONTINUATION.md`](docs/VLA_WAM_CONTINUATION.md): exact next
  experiments, commands, stop rules, and restart procedure.
- [`continuation_state.json`](artifacts/vla_wam_shared_v2/continuation_state.json):
  machine-readable queue and blockers.
- [`VLA_WAM_STEERABILITY_V2_PROTOCOL.md`](docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md):
  frozen design and claim boundary.
- [`VLA_VS_WAM_STEERABILITY_STUDY.md`](docs/VLA_VS_WAM_STEERABILITY_STUDY.md):
  reader-facing article and current evidence.
- [`VLA_WAM_STEERABILITY_VIDEO_GALLERY.html`](docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.html):
  complete paired success/failure clips.
- [`experiments/groot_droid/README.md`](experiments/groot_droid/README.md):
  controlled-access preflight for the blocked GR00T N1.7 DROID checkpoint.

Validate the active evidence package with:

```bash
python3 tools/validate_vla_wam_v2_protocol.py
```

## Bridge ablation audit

The Bridge track implements the dataset feasibility gate in Linear issue
`RES-1`: determine whether a controlled 2 × 2 ablation can vary temporal
language density and surface-form diversity while holding robot examples fixed.

### Bridge result

**Training no-go; conditional go to curation.** The pinned releases support a
clean structural construction, but not yet a scientifically valid four-cell
experiment:

- 17,580 conservative, one-to-one episode/sidecar joins.
- 12,332 trajectories pass integrity and temporal-density rules.
- A group-safe 704-trajectory target and nested 192-trajectory pilot are frozen.
- All A/B/C/D robot-row, split, language-pool, and sampling assertions pass.
- Verified task-level paraphrase groups: **0**.
- Verified subtask-level paraphrase groups: **0**.
- Generated B/D wrappers are provisional and their task/subtask surface
  distributions are not matched. They must not be used for training claims.

The detailed decision and evidence are in
[`artifacts/res1/DECISION_MEMO.md`](artifacts/res1/DECISION_MEMO.md) and
[`docs/RES-1_PROGRESS.md`](docs/RES-1_PROGRESS.md).

## GR00T language-dependence pilot

An isolated LeRobot 0.6 / GR00T N1.7 experiment probes conditional mutual
information by holding LIBERO-Spatial observations fixed, swapping all ten
suite prompts, and repeatedly sampling action chunks. The implementation and
reproduction command are in
[`experiments/groot_cmi/README.md`](experiments/groot_cmi/README.md); the pilot's
blog-ready interpretation is generated at
[`artifacts/groot_cmi/libero_spatial_ep0/BLOG_FINDINGS.md`](artifacts/groot_cmi/libero_spatial_ep0/BLOG_FINDINGS.md).

## Pinned sources

- Robot episode metadata, observations, and actions:
  [`IPEC-COMMUNITY/bridge_orig_lerobot`](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot)
  at `0e9d76d07e9df3ea3eba257b2520d4913833fad2`.
- Dense language sidecars:
  [`Embodied-CoT/steering_features_bridge`](https://huggingface.co/datasets/Embodied-CoT/steering_features_bridge)
  at `094f1f7259148e03619e73b45d7dff54995e7003`.

The LeRobot episode index is never assumed to equal the steering trajectory
ID. Identity is recovered from the sole normalized task string shared by every
subtask command pool and retained only when it is unique on both sides.

## Reproduce

The checked-in `uv.lock` is the preferred Python 3.10+ environment:

```bash
uv sync --extra dev
uv run steerable-res1 all
uv run steerable-res1 visual-audit
uv run steerable-res1 finalize-audits
```

Equivalent source-tree or pip-installed commands are:

```bash
PYTHONPATH=src python -m steerable_bridge run
PYTHONPATH=src python -m steerable_bridge visual-audit
PYTHONPATH=src python -m steerable_bridge finalize-audits
```

The workstation's global Python environment currently autoloads an unrelated,
broken `bdai` pytest plugin. The isolated repository suite is either:

```bash
uv run --extra dev python -m pytest -q
# or, without uv:
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Research artifacts

| Artifact | Interpretation |
| --- | --- |
| [`DECISION_MEMO.md`](artifacts/res1/DECISION_MEMO.md) | End-of-day go/no-go decision |
| [`annotation_inventory.json`](artifacts/res1/annotation_inventory.json) | Counts, joins, retention funnel, claim limits |
| [`join_normalization_sensitivity.csv`](artifacts/res1/join_normalization_sensitivity.csv) | Exact join sensitivity; explains why the earlier 16,945 estimate is not reproduced |
| [`temporal_density_summary.csv`](artifacts/res1/temporal_density_summary.csv) | Overall/source-level `T`, `K`, and `K/T` summaries |
| [`paraphrase_eligibility_table.csv`](artifacts/res1/paraphrase_eligibility_table.csv) | Candidate and verified intent-group capacity |
| [`split_validation.json`](artifacts/res1/split_validation.json) | Group isolation, nesting, and task-overlap assertions |
| [`manifest_validation.json`](artifacts/res1/manifest_validation.json) | Target structural assertions and scientific gate |
| [`pilot_manifest_validation.json`](artifacts/res1/pilot_manifest_validation.json) | Pilot structural assertions |
| [`surface_diversity_report.csv`](artifacts/res1/surface_diversity_report.csv) | Task/subtask length, lexical distance, and duplicate diagnostics |
| [`human_audit_summary.json`](artifacts/res1/human_audit_summary.json) | Fail-closed review completion gate |
| [`audit_lock.json`](artifacts/res1/audit_lock.json) | Immutable audit membership, runtime seed, and exact run-provenance lock |
| [`implementation_manifest.json`](artifacts/res1/implementation_manifest.json) | Per-file and combined implementation SHA-256 fingerprints |
| [`visual_audit/index.html`](artifacts/res1/visual_audit/index.html) | Locked 20-video direct-alignment review bundle |

Large reproducible row-level CSV/Parquet files, downloaded source data, plots,
and review videos are Git-ignored. Pinned input files, implementation files,
selected target/pilot cohort outputs, and review videos have sizes and content
digests in `input_manifest.json`, the implementation and audit locks, and the
generated video manifest. Other large diagnostic tables and plots are
deterministically regenerated and validated through their published counts and
assertions, but are not claimed to have individual release digests.

## Human audit workflow

Complete these sheets without changing the locked sample membership:

1. `visual_alignment_audit.csv` (20 videos).
2. `manual_sequence_audit.csv` (30 trajectory sequences).
3. `manual_command_audit.csv` (100 command pools / 838 command slots).
4. `manual_paraphrase_group_audit.csv` (100 generated groups).

Rows marked `secondary_review_required=True` make up a deterministic 20%
second-review sample. Pipeline reruns preserve a sheet once any review field is
filled. `audit_lock.json` binds all immutable sheet fields to the base seed,
pinned inputs, target/pilot splits, all target/pilot manifests, and validator
outputs. `finalize-audits` parses yes/no judgments, reports per-class taxonomy
precision and false equivalence, requires independent adjudication of actual
disagreements, and remains a no-go on blanks, malformed values, or provenance
drift. It is report-only: reviewed strings are not promoted into eligibility or
manifests without an explicit curation and full-regeneration stage.

Even perfect reviews of the current wrappers will not clear the surface-match
gate: B and D currently differ by 5.14 mean tokens and 0.133 mean pairwise
Jaccard distance. A genuine, approximately matched paraphrase generation and
adjudication pass is the next research step.
