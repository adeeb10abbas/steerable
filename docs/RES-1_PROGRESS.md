# RES-1 research log

Last updated: 2026-07-23 (America/New_York)

## Research question

Can one frozen Bridge cohort generate four matched training views that vary
only temporal semantic density and genuine surface-form diversity?

The implemented answer is: **robot-row construction yes; scientifically valid
language treatment not yet.** The current state is a training no-go and a
conditional go to continue curation.

## Pinned inputs

| Input | Revision | Records |
| --- | --- | ---: |
| `Embodied-CoT/steering_features_bridge` | `094f1f7259148e03619e73b45d7dff54995e7003` | 53,192 trajectory keys; 38,454 dense sidecars |
| `IPEC-COMMUNITY/bridge_orig_lerobot` | `0e9d76d07e9df3ea3eba257b2520d4913833fad2` | 53,192 episodes / 1,893,026 frames |

The downloader verifies hosted file size and records a local SHA-256 digest.

## Decisions and evidence

### D1 - Identity is a conservative string join, not index equality

The annotation release has no explicit LeRobot episode ID. For multi-pool
steering trajectories, the original task normally appears in every subtask
command pool. The join retains a task only when exactly one normalized string
is shared across all pools and that task identifies exactly one trajectory on
both releases.

- Accepted pairs: 17,580.
- Accepted pairs with equal raw indices: 403.
- Tier B pairs when LeRobot vocabulary disambiguates steering records: 17,601.
- The shared-chat estimate of 16,945 is not reproducible using documented
  form-only normalization. Stopword deletion approaches it but is rejected as
  unsafe identity logic.

### D2 - Direct timestep mapping is the code-consistent candidate

All accepted pairs have sidecar steps `0..L+1` for a LeRobot episode of length
`L`. The released training loader performs direct `frame i -> sidecar step i`
lookup. The Bridge conversion makes LeRobot frame `i` correspond to raw Bridge
observation `i+1`; these are compatible statements, not an off-by-one conflict.

Direct versus `i+1` changes 80,486 of 621,480 used labels. The 20-video audit
therefore remains a real scientific gate. The software packages the evidence
but never fills human judgments.

### D3 - Automatic taxonomy is candidate evidence, not semantic truth

The 1,730,644 released command slots are provisionally classified as:

| Automatic class | Slots |
| --- | ---: |
| Task language | 208,275 |
| Subtask same-intent candidate | 244,281 |
| Atomic motion/gripper | 99,091 |
| Point grounding | 364,922 |
| Multi-point trace | 194,076 |
| Hybrid/added semantics | 246,163 |
| Malformed or semantically unclear | 373,836 |

Rules preserve object, relation, and direction constraints, treat ordinary
single-coordinate grounded actions as grounding, and leave unsupported cases
unclear. No residual class is silently promoted to a paraphrase.

The 100-pool audit allocation is fixed at 32 Bridge v2, 23 Bridge v1, 21 RSS,
13 ICRA, and 11 FLAP pools. Twenty percent is pre-marked for independent second
review.

### D4 - Temporal scale is ample; verified language scale is zero

- Integrity-eligible: 17,580 trajectories / 621,480 frames.
- Density-eligible (`K >= 3`, no one-frame segment): 12,332 trajectories /
  454,153 frames / 64,976 semantic segments.
- Release-only subtask groups with at least 1/2/4/6 non-canonical automatic
  candidates: 17,367 / 2,157 / 6 / 0.
- Final verified task groups: 0.
- Final verified subtask groups: 0.

Generated wrappers preserve each canonical string verbatim and provide four
training plus two held-out strings. They are marked provisional and pending
human review. They test the data path, not genuine lexical diversity.

The released-pool thresholds explicitly exclude the canonical text itself. A
candidate count is not a semantic-equivalence judgment and does not confer
training eligibility.

### D5 - Split isolation is broader than trajectory ID

The target is selected first, with every original Bridge `out.npy` capture
group assigned to one split. The pilot is a role-preserving nested subset.

| Cohort | Train | Validation | Test | Frames |
| --- | ---: | ---: | ---: | ---: |
| Pilot | 128 | 32 | 32 | 7,074 |
| Target | 512 | 64 | 128 | 25,910 |

There is no trajectory leakage, capture-group leakage, pilot/target role
switching, or held-out-language/train-pool overlap.

The conservative join makes exact normalized tasks globally unique. Therefore
val/test trajectory metrics also measure new-task generalization. Isolated
wording evaluation must instead use `selected_heldout_instruction` on the same
immutable row references; it cannot be inferred from trajectory split alone.

### D6 - Structural equality and scientific validity are separate gates

All four target views contain 25,910 identical robot rows and reference hash
`2e4f6f1f80a75d8bd46529d1d622d753c3c9f39cf2907e6edfdc187c6631e6d1`.
The validator adversarially checks condition metadata, canonical and intent
identity, master-pool equality, 4/2 pool sizes, disjointness, selected-string
membership, deterministic selection, update semantics, splits, and robot refs.

Scientific language validity is false because:

- all human audits are blank;
- generated strings are low-strength prompt wrappers;
- embedding distance is not computed for unverified surfaces;
- B/D mean surface length differs by 5.139 tokens; and
- B/D mean pairwise Jaccard distance differs by 0.133.

### D7 - Human review is locked to one exact run

The four audit templates are protected by `audit_lock.json`: immutable row
fields, membership, secondary-review assignment, row/unit counts, runtime
seeds, and sheet hashes must match. The same lock binds the pinned input
manifest, split validation, target/pilot splits, all target/pilot master and
A/B/C/D Parquet files, and both manifest validators.

- Implementation fingerprint:
  `64317d31a734b7d6fea56294e81c1222fbed5c0708c4cd26df0b6d546fcbcff3`.
- Run-provenance fingerprint:
  `2b3d661f65070dd316154a625b30635c327c9b7b40a5ef77bc13bb13c36a1280`.
- Locked audit units: 100 command pools / 838 slots, 100 paraphrase groups,
  20 visual sequences, and 30 broader sequence checks.
- Every locked provenance field currently validates; human judgments remain
  blank, so the scientific gate stays false.

`finalize-audits` is report-only. It never promotes reviewed strings into the
eligibility table or rewrites A/B/C/D; promotion requires an explicit curation
step followed by complete regeneration.

## Implementation status

- [x] Revision-pinned download and digest manifest.
- [x] Conservative join plus exclusion ledger and normalization sensitivity.
- [x] Complete frame-language annotation manifest.
- [x] Temporal-density metrics, source summaries, and plots.
- [x] Seven-way automatic taxonomy and locked 100-pool audit.
- [x] Candidate/verified paraphrase eligibility table.
- [x] Group-safe target and nested pilot splits.
- [x] Target and pilot A/B/C/D structural manifests.
- [x] Adversarial manifest validation and split validation.
- [x] Twenty-video captioned visual-audit bundle.
- [x] Fail-closed audit finalizer and decision memo.
- [x] Immutable audit sample, implementation, seed, and run-provenance locks.
- [ ] Human video/sequence review.
- [ ] Human command taxonomy review.
- [ ] Generate and adjudicate genuine matched task/subtask paraphrases.
- [ ] Compute embedding-distance report on verified surfaces.
- [ ] Re-run the final gate; training remains unauthorized until it passes.

## Primary artifacts

| Artifact | Status |
| --- | --- |
| `artifacts/res1/DECISION_MEMO.md` | Complete |
| `artifacts/res1/annotation_inventory.json` | Complete |
| `artifacts/res1/quality_issues.csv` | Complete, generated/ignored |
| `artifacts/res1/trajectory_density.csv` | Complete, generated/ignored |
| `artifacts/res1/temporal_density_summary.csv` | Complete |
| `artifacts/res1/command_taxonomy.csv` | Complete, generated/ignored |
| `artifacts/res1/manual_command_audit.csv` | Template ready; review pending |
| `artifacts/res1/eligible_intents.csv` | Complete, generated/ignored; verified count zero |
| `artifacts/res1/pilot_split.json` | Complete |
| `artifacts/res1/target_split.json` | Complete |
| `artifacts/res1/split_validation.json` | Pass |
| `artifacts/res1/manifest_validation.json` | Structural pass / scientific fail |
| `artifacts/res1/pilot_manifest_validation.json` | Structural pass / scientific fail |
| `artifacts/res1/visual_audit/index.html` | Complete; review pending |
| `artifacts/res1/human_audit_summary.json` | Fail-closed; training ready false |
| `artifacts/res1/audit_lock.json` | Pass; sample and exact run locked |
| `artifacts/res1/implementation_manifest.json` | Complete |

## Next research action

Review the videos at `artifacts/res1/visual_audit/index.html`, fill the four
locked CSV sheets, and run:

```bash
PYTHONPATH=src python -m steerable_bridge finalize-audits
```

This command reports review results but does not promote new surface pools.

In parallel, replace the prompt wrappers with genuine task- and subtask-level
paraphrases generated to a shared length/divergence target, then adjudicate
them without admitting coordinates, atomic motions, traces, abstraction
changes, or added constraints.
