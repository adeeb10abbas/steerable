# RES-1 decision memo: CONDITIONAL GO TO CURATION / TRAINING NO-GO

Date: 2026-07-23  
Pinned annotations: `094f1f7259148e03619e73b45d7dff54995e7003`  
Pinned LeRobot data: `0e9d76d07e9df3ea3eba257b2520d4913833fad2`

## Decision

**Continue curation, but do not train the current B/D manifests.** A common
robot-row set can generate all four structural views, but the bottleneck cell
has **zero verified paraphrase groups**. The generated strings quote the
canonical instruction inside six prompt wrappers; they are useful for testing
the plumbing, not yet evidence of genuine lexical surface diversity.

## Four-gate result

1. **Sidecar join — conditional pass.** The release has 53,192 trajectory
   keys and 38,454 densely annotated trajectories. Recovering the trajectory
   task as the sole normalized string shared across all subtask pools and then
   requiring a one-to-one normalized-task match yields **17,580**
   episode pairs. Only **403** pairs retain the
   same raw index, so index equality would be wrong. Every accepted pair has
   contiguous annotation steps `0..L+1`, direct released-code coverage
   `frame i -> annotation step i`, and valid command pools. Physical boundary
   alignment still requires the 20-video review.

2. **True surface paraphrases — fail pending curation.** The sidecar contains
   **1,730,644** command slots, mixing task text, subtask
   wording, coordinates, paths, strict motion/gripper commands, hybrids, and
   unclear strings. Automatic rules can identify candidates but cannot certify
   semantic equivalence. Blank audit fields remain failures by construction.

3. **Scale — temporal pass, language fail.** Integrity and
   temporal rules retain **12,332**
   trajectories, enough to freeze both `{'train': 128, 'validation': 32, 'test': 32}` and
   `{'train': 512, 'validation': 64, 'test': 128}` structurally. Verified language capacity is zero, so
   these are provisional curation cohorts rather than authorized training sets.
   New task-level paraphrases are mandatory; matched subtask paraphrases must
   also be generated or individually verified.

4. **Matched manifests — structural pass; scientific fail.**
   A/B/C/D have 25,910 rows and the same
   pinned trajectory, frame, split, observation reference, action reference,
   and reference hash. High-diversity training/held-out pools are disjoint and
   deterministically sampled. However, audit status is pending and B/D are not
   distribution-matched: mean total-length difference is
   **5.14 tokens**
   and mean Jaccard-distance difference is
   **0.133**.

## Eligibility bottleneck

| Level | Origin | Intent groups | At least 6 non-canonical candidates | Verified |
| --- | --- | ---: | ---: | ---: |
| subtask | generated_verbatim_wrapper | 64,976 | 64,976 | 0 |
| subtask | released | 206,243 | 0 | 0 |
| task | generated_verbatim_wrapper | 12,332 | 12,332 | 0 |

Candidate counts exclude the canonical text and remain diagnostic, not training
eligibility.

## Frozen exclusions and split rules

- Reject one-pool or multi-intersection task recovery, missing/ambiguous task
  matches, malformed/non-contiguous steps, absent command pools, non-string or
  empty used commands, fewer than three semantic segments, any one-frame
  segment, and unknown source collection.
- Assign each original Bridge `out.npy` group to exactly one split, then sample
  the 704-trajectory target with seed `20260725`. Select the nested,
  role-preserving 192-trajectory pilot with seed `20260724`. Language
  surfaces, paraphrase review, and visual review use base seed `20260723`;
  the broader sequence audit uses `20260753`, while command-pool
  membership is deterministic without a separate runtime seed. Stratification uses
  source, instruction length, trajectory length, and segment count; task family
  is retained as a diagnostic because exact task labels are globally unique.
- Coordinates, traces, unresolved placeholders, strict atomic commands,
  hybrids, or any candidate that changes an object, relation, direction,
  gripper state, temporal order, or other constraint cannot enter the primary
  surface-diversity treatment.
- Exact joined task strings are globally unique. Therefore val/test trajectory
  metrics also test new tasks. To isolate wording, evaluate each row's explicit
  `selected_heldout_instruction` against the same immutable robot references.
- Bridge v1 and FLAP each contribute only one density-eligible trajectory and
  are absent from the deterministic 704-trajectory target. Reported cohorts
  cover Bridge v2, RSS, and ICRA only; this is not silently generalized.

## What the diversity number means

The automatic annotation-anatomy figure is a provisional rule audit, not a
claim that residual strings are paraphrases.

![Automatic annotation anatomy](plots/annotation_anatomy.png)

## Tomorrow's first action

Open `visual_audit/index.html`, then complete `visual_alignment_audit.csv`,
`manual_command_audit.csv`, `manual_sequence_audit.csv`, and
`manual_paraphrase_group_audit.csv` with independent second review on the
pre-marked 20 percent. Run `steerable-res1 finalize-audits`. Because the current
B/D language distributions are unmatched and wrappers are low-strength, this
command only reports the fail-closed gate; it does not promote reviewed strings
or rewrite manifests. The expected next step is a genuine matched paraphrase
generation/adjudication and full-regeneration pass, not model training.
