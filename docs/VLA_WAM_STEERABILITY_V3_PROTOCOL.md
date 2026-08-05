# VLA/WAM language steerability v3

Status: frozen before any v3 model request or behavioral inference. This is a
disclosed prospective extension after all v2 and V2-A015 results were known.
It does not revise, pool, relabel, or rerun valid v2/V2-A015 evidence.

The machine-readable source of truth is
[`protocol.json`](../artifacts/vla_wam_shared_v3/protocol.json). The companion
amendment, arena registries, taxonomy, and analysis plan are all required and
validated together.

## Core boundaries

- DROID/RoboLab and RoboTwin remain separate arenas: raw successes, failures,
  and denominators are never pooled.
- Prompts are static for an episode. Oracle actions, coaches, prompt switches,
  and progress-conditioned instructions are prohibited.
- Every valid failure remains evidence. Setup/infrastructure attempts are kept
  in a separate ledger and may only be repaired at the identical registered
  cell.
- Viewport video, executed action traces, and raw JSONL are required for every
  behavioral cell.

## Phase A: direct-command replication

DROID targets 30 exact matched LEFT/RIGHT environment-and-sampling pairs per
checkpoint, seeds 8300–8329. The direct sentences and release-inside-the-45°
predicate are unchanged. π0-FAST seeds 8300–8309 remain preserved historical
evidence, while 8310–8329 are blocked until the exact missing OpenPI/RoboLab
commits are recovered. Its V2-A008 current-stack prompt-sensitivity failure is
not a release. For other DROID checkpoints, 8303–8329 are new additions;
8300–8302 may only contribute as preserved evidence when their full pinned
runtime identity matches exactly, never by rerun or replacement.

Priority is π0-FAST, GR00T N1.7, Cosmos3 Edge Policy DROID, Cosmos3 Nano
Policy DROID, π0.5 current-stack, then DreamZero.

RoboTwin retains the seven core anchor scenes pairs03–09. Pair `p` has scene
seed `4300000+p`; both requested directions run in the identical anchor reset.
Sampling replicate `r=0` uses `8400+p` and is preserved v2 evidence. For
`r=1..9`, the seed is `8400+p+100*r`. This creates 126 new episodes per model
and 378 new episodes across Efficient-WAM-RT, FastWAM, and LingBot-VA.
The generated Phase-A JSONL queue and its hash-bearing manifest are required
artifacts: they fix 780 total cells (360 DROID, 420 RoboTwin), including 648
authorized-new, 50 preserved-candidate, 42 preserved-r0, and 40 blocked-π0
rows.

## Separately gated work

Confound ablations change one named factor only, use contemporaneous randomized
matched control/intervention cells, and retain separate denominators. Their
numeric levels are intentionally unreleased: a model-blind fixture calibration
must first freeze the coordinates and relation margins in a new amendment.

Phase C freezes the exact existing four DROID strings for direct command, short
command, goal-as-outcome, and desired-plus-negated-opposite. It uses 20 shared
matched seeds 8500–8519 and the first three non-π0 DROID priorities: GR00T
N1.7, Cosmos3 Edge Policy DROID, and Cosmos3 Nano Policy DROID. That is
`3 × 20 × 4 × 2 = 480` separately gated episodes. π0-FAST is optional only
after exact revision recovery and a new model-specific sensitivity release.

Phase D is lower-priority and applies to every released Phase-A registered
(scene/reset, instruction) condition whose runtime exposes a real effective
stochastic policy seed. Each eligible condition receives the same 16 shared
sampling-seed indices. Deterministic/no-effective-seed runtimes are ineligible,
not padded with fake repetition. The 16 rollouts are nested within a condition
and are never analyzed as 16 independent scenes.

## Failure classification

The old frozen failure stage is always retained. V3 assigns one additional
precedence class: `correct`, `pick_failed`, `wrong_side`, `release_failed`, or
`transport_failed`. Correct uses the frozen success predicate; a missing
three-sample +3 cm pickup is `pick_failed` before either final-region class can
apply. `wrong_side` and `release_failed` therefore require a picked object and
a sustained final opposite/requested region, respectively. `release_failed`
also requires the separately recorded frozen-scorer `final_detached_release`
boolean to be false; it is never inferred from requested success or a gripper
command. A requested-success false record with sustained requested placement
and detached release true is a technical-invalid scorer inconsistency, not a
behavioral class. Alongside margins,
pickup, and release fields it
retains signed final lateral offset, cone/native-region entry and kind,
episode length, contact availability/reason, and object path length. The full
codebook and continuous fields are in
[`failure_taxonomy.json`](../artifacts/vla_wam_shared_v3/failure_taxonomy.json).
The signed lateral offset is the frozen raw robot-frame
movable-minus-reference `delta_y` (positive robot LEFT), matching v2; requested
margin is `+delta_y` for LEFT and `-delta_y` for RIGHT. It is never replaced by
a negated reader-display coordinate.
Contact status is explicitly `observed`, `not_observed`, or
`instrumentation_unavailable`: the contact step is an integer only when
observed, and a reason is required only for instrumentation unavailability. A
retained all-false contact stream means `not_observed`, never unavailable.

## Execution order

Freeze and validate these artifacts first. Then run only model-blind fixture,
runtime-identity, and output-write preflights; next run the fixed-observation
repeat/sensitivity gate. Behavioral direct replication releases only after its
model-specific gate passes. Later blocks require their own independent release.

Run `python3 tools/validate_vla_wam_v3_protocol.py` before any v3 inference.
