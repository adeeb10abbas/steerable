# Publication handoff

This is the shortest path for an author preparing a research blog or paper.
Operational handoffs are evidence provenance, not article outlines.

## Read in this order

1. [`VLA_WAM_RESEARCH_BLOG.md`](VLA_WAM_RESEARCH_BLOG.md) — concise current
   narrative and claim boundary.
2. [`direct_command_cross_model_comparison.md`](../artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.md)
   — canonical arena-separated result table.
3. [`VLA_WAM_STEERABILITY_VIDEO_GALLERY.html`](VLA_WAM_STEERABILITY_VIDEO_GALLERY.html)
   — selected execution and exposed-imagination media.
4. [`media/README.md`](../artifacts/vla_wam_shared_v2/media/README.md) and
   [`vla_wam_study_stats.xlsx`](../outputs/vla_wam_research_handoff/vla_wam_study_stats.xlsx)
   — complete video roles plus formula-derived rates and evidence inventory.
5. [`VLA_WAM_STEERABILITY_V2_PROTOCOL.md`](VLA_WAM_STEERABILITY_V2_PROTOCOL.md)
   — frozen experimental rules.
6. [`publication_manifest.json`](../artifacts/vla_wam_shared_v2/publication_manifest.json)
   — exact publication assets and SHA-256 digests.

## Claims that are safe to make

- Changing only LEFT versus RIGHT changed executed actions in every matched
  pair for models whose compiled evidence reports action distinctness.
- Endpoint redirection was often stronger than task completion. Language
  sensitivity therefore does not imply competence.
- DreamZero exposed repeatable, prompt-dependent decoded futures, but completed
  only three of six behavioral episodes. Dreaming did not guarantee execution.
- DROID/RoboLab and RoboTwin outcomes must remain in separate tables.
- In the exploratory V2-A015 guidance ablation, the Cosmos `g=1` arm had lower
  total success and both requested-side margins than `g=3`; DreamZero's derived
  `s=2` arm redistributed success toward RIGHT rather than improving both
  directions.

## Claims not supported

- A general VLA-versus-WAM ranking.
- Pooled success rates across simulators.
- Zero-valued imagination scores for models with missing, latent-only, or
  action-only future interfaces.
- Treating infrastructure-invalid attempts as behavioral failures.
- Presenting the current-stack π0 replication as the missing historical-code
  wording experiment.
- Presenting static Cosmos-Reason2 diagnostics as robot behavior.
- Presenting either Cosmos3 base-model interface probe as behavioral evidence,
  or presenting its generated video as simulator execution.

## Current additions

The original bounded study is complete. π0-FAST V2-A008 is closed without
behavioral evidence: its fixed-observation probe repeated LEFT exactly but
returned an identical action tensor for LEFT and RIGHT (RMS 0.0). Its 60
registered cells remain unrun and are not failures or zeros. A later,
separately labeled addition is complete:

- V2-A010: six valid π0.5 current-stack direct cells — LEFT 1/3, RIGHT 3/3,
  three aligned endpoint pairs, and three distinct action pairs. Its selected
  seed-8300 media is actual simulator execution only, not recovered historical
  π0.5 footage and not an imagined future.

- V2-A011: six valid Cosmos3 Nano Policy DROID current-stack direct cells —
  LEFT 3/3, RIGHT 3/3, three aligned endpoint pairs, three distinct-action
  pairs, and 37 retained decoded futures. Its selected seed-8300 card shows
  actual simulator execution beside a clearly labeled model prediction; the
  prediction is not execution or an additional trial.

- V2-A013: Cosmos3 Edge base passed its three-request fixed-observation
  action-plus-video interface probe. Repeat LEFT was bit-identical and RIGHT
  differed, but the exact CuRobo mapping audit blocked execution. Its selected
  gallery media is prediction-only; actual rollout is explicitly unavailable.

- V2-A012/V2-A014: Cosmos3 Super base passed the corresponding image-only
  action-plus-video diagnostic. It has no robot state, controller execution, or
  released behavioral cell and cannot support a DROID execution claim.

- V2-A015: two separate six-cell, post-result DROID guidance ablations are
  complete at seeds 8300–8302. Cosmos3 Nano `g=3 → g=1` changed success from
  `6/6 → 4/6` (LEFT `3/3 → 1/3`; RIGHT `3/3 → 3/3`), and the all-cell paired
  mean requested margin decreased by `0.1302 m`. DreamZero conditional-action-equivalent
  `s=1 → s=2` changed success from `3/6 → 4/6`, but LEFT fell `2/3 → 1/3`
  while RIGHT rose `1/3 → 3/3`. Report the latter as directional
  redistribution under derived CFG-style negative-branch action guidance,
  not as an official DreamZero action-CFG feature or a powered general gain.
  The gallery contains all six intervention executions per arm, every complete
  DreamZero official decode, and all 64 Cosmos local prediction horizons.

The V2-A010 and V2-A011 results may be described only with their current-stack
labels and separate DROID denominators. Nano’s two setup failures occurred
before policy requests and are not behavioral failures. Do not add either base
arm to behavioral result tables; their interface media must remain explicitly
non-execution.

## Repository map

- `docs/` — reader-facing blog, protocol, gallery, and operational handoffs.
- `artifacts/vla_wam_shared_v2/results/` — compiled tables and figures.
- `artifacts/vla_wam_shared_v2/media/` — complete role-aware video catalog and
  bounded selected publication media.
- `outputs/vla_wam_research_handoff/` — shareable research statistics workbook.
- `artifacts/vla_wam_shared_v2/pilot/` — registries, compact episode evidence,
  and invalid-attempt/intervention ledgers.
- `experiments/` — model adapters and launch documentation.
- `tools/` — validators, compilers, media selection, and manifest builders.
- `handoff/repo_bundles/` — pinned external repository history.

Raw rollouts, checkpoints, and Python environments belong on the PVC and must
not be committed.

## Historical material

[`VLA_VS_WAM_STEERABILITY_STUDY.md`](VLA_VS_WAM_STEERABILITY_STUDY.md) is a
long-form technical record spanning v1 and early v2 evidence cutoffs. The
continuation documents preserve operational history. Older blog and social-copy
drafts are not sources for current aggregate claims.
