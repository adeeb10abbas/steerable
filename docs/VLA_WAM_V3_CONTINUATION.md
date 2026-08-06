# VLA/WAM v3 expansion continuation

Updated: 6 August 2026, after the complete Phase-A launch-authorized queue, the
separately authorized π0-FAST compatibility bridge, the cross-version
measurement-coverage audit, and the model-blind release plus pre-inference live
runtime implementation for Nano V3-B001 were compiled. The machine-readable source of truth is
[`continuation_state.json`](../artifacts/vla_wam_shared_v3/continuation_state.json).

## Status

All **648 cells marked `authorized_new`** in the frozen Phase-A queue are
complete valid evidence: 270 DROID/RoboLab episodes and 378 RoboTwin episodes.
The queue's 40 `blocked_pi0` rows remain frozen under their unavailable exact
historical runtime identity. Amendment V3-A002 instead authorized a distinct,
publicly reproducible π0-FAST compatibility cohort: all **20 matched pairs / 40
episodes** are now complete valid evidence. That cohort is never pooled with
the blocked rows or preserved V2 evidence.

No Phase-A or V3-A002 inference remains. Exactly one Phase-B ablation is now
released: Nano V3-B001, a positions-only movable-object reflection with 108
registered cells. Its release followed model-blind calibration and occurred
before any Phase-B model request or behavioral episode. Every other Phase-B
ablation, all Phase-C wording cells, and all Phase-D stochastic repetitions
remain unreleased.

### Measurement-coverage gate for Phase B

The committed [measurement audit](../artifacts/vla_wam_shared_v3/measurement_coverage_audit.json)
checks all **982 unique behavioral episodes** across V1, V2, Phase A, and
V3-A002. Requested-side margin and signed final lateral offset are available
for **982/982** episodes, including failures; no value is imputed from binary
success. Older aliases are either explicit or algebraically exact. In
particular, older RoboTwin object-minus-target `x` is standardized as `-x`, so
positive remains robot LEFT, and its arena is still never pooled with DROID.
No legacy behavioral rerun is required for these two measurements.

The same audit reproduces Nano's 27-pair Phase-A margin gap: mean RIGHT-minus-
LEFT requested margin `0.123601 m`, with 23 positive and 4 negative paired
gaps and an exact two-sided sign-test `p = 0.0003107`. GR00T is already complete
at 27 matched pairs / 54 episodes and must not be rerun. Phase B must log both
fields for every new valid episode: signed offset supplies the full-sample
analysis, while any success-conditional margin analysis must name its reduced
subset explicitly.

### Nano Phase-B position-reflection release (V3-B001)

V3-B001 asks whether Nano's directional bias changes when the center positions
of the movable objects are reflected about the robot sagittal plane. It is a
**positions-only movable-object intervention, not a full-scene mirror**: the
robot, cameras, and nonmovable geometry remain fixed. Initial quaternion sources
are identical across layouts; any measured post-settle orientation difference
is retained as a downstream physical mediator rather than described as fixed.

The frozen queue contains **27 prespecified seeds, 9400–9426, × four cells per
seed = 108 cells**: `control` and `position_mirrored`, each under the exact
static prompts:

> Put the Rubik's cube to the left of the bowl.

> Put the Rubik's cube to the right of the bowl.

At release, calibration recorded **zero model requests and zero behavioral
episodes**. The primary analysis uses signed final lateral offset for the full
sample, including every valid behavioral failure. Requested-side margin is
secondary and success-conditional: it may be analyzed only on the named
`nano_v3b001_all_four_cells_correct` complete-case subset, containing seeds for
which control LEFT, control RIGHT, position-mirrored LEFT, and
position-mirrored RIGHT all pass the frozen success predicate. Missing values
are never imputed and unmatched successful cells are never mixed.

| Frozen release artifact | SHA-256 |
| --- | --- |
| [`model_blind_calibration_report.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/model_blind_calibration_report.json) | `112716acada89050561c9488d93a333a300b1675c2305329c8e5aceeb4e6da71` |
| [`nano_mirror_v3b001_cells.jsonl`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl) | `018b8b6ae76ac46f2f89eef83c4b16d7a4ff3d1ff15d91527b96fb56b5432c5a` |
| [`nano_mirror_v3b001_manifest.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_manifest.json) | `5c82268739feb41281435a51dcd848b575218cd9fbe5839d9ad130d1a7888830` |
| [`post_result_nano_mirror_v3b001_amendment.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/post_result_nano_mirror_v3b001_amendment.json) | `9d88c29733fa3b24a154977bc25d04d2d77df5be59e3213f0c3a6cfbe3edc6a0` |

These files authorize the exact queue after the required live identity and
output-path rechecks; they do not report Phase-B behavioral outcomes.

### Nano V3-B001 live runtime boundary

The fail-closed live implementation is in reset-gate repair: V3-B001 has 108
released cells and zero completed valid behavioral cells. The first excluded
450-action smoke produced 15 model requests and exposed the export-finalization
defect. After that repair, a 233-action smoke compiled with eight requests, but
the following control cell exposed a coordinate-frame defect: the released
fixture is explicitly robot-frame, while the bridge compared it to world-frame
poses. The compiled smoke was invalidated before analysis. Both complete raw
attempts, the deterministic zero-request control failures, and all media are
preserved and hash-ledgered in
[`live_infrastructure_ledger.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_infrastructure_ledger.json).

The current repair compares released and observed positions in the robot frame,
writes bridge exceptions durably before Isaac closes, requires the export
before compilation, and keeps RoboLab output in the immutable attempt-local PVC
directory. A new live identity must be bound after this source change before
retrying the same first cell. The implementation binds a new
Phase-B live runtime identity to the verified Phase-A Nano identity and exact
release, pins the Nano server CLI, performs the released positions-only reset,
and preserves every decoded future, executed action, viewport video, state
capture, reset attestation, and compiled JSONL record outside ordinary Git.

Before the first request, the bridge must reproduce the model-blind reset gate:
60 hold-action settle steps followed by a 15-step stability window, the frozen
linear and angular speed thresholds, neutral requested-relation predicates, and
an episode counter of 75 that is then reset to zero. A failed gate, partial
attempt, or pre-existing output path remains outside the behavioral denominator
and cannot be overwritten. The launcher preserves the frozen queue order,
records `OMNI_KIT_ACCEPT_EULA=YES`, and does not use a thermal guard.

The exact next cell remains:

> `v3b001:nano:seed9400:position_mirrored:right`

> Put the Rubik's cube to the right of the bowl.

Its repaired live-bound runtime identity must be created and verified on the
ali-owned PVC after the committed implementation is synchronized. The excluded
and invalidated attempts do not change the scientific status: no valid Phase-B
result exists yet.

## Exact intervention

DROID changed only these episode-static prompts inside each matched seed:

> Put the Rubik's cube to the left of the bowl.

> Put the Rubik's cube to the right of the bowl.

RoboTwin used the same sentence frame with pair-specific object names. The
exact fourteen rendered sentences are recorded in
[`continuation_state.json`](../artifacts/vla_wam_shared_v3/continuation_state.json)
and the frozen [`phase_a_cells.jsonl`](../artifacts/vla_wam_shared_v3/phase_a_cells.jsonl).
LEFT and RIGHT always shared the registered reset, scene seed, sampling seed,
runtime, controller, and horizon.

## DROID/RoboLab Phase A

Each completed checkpoint contributes **27 new matched pairs / 54 valid V3
episodes** at seeds 8303–8329. Seeds 8300–8302 remain separate V2 evidence
because their complete V3 runtime identity could not be established.

| Checkpoint | LEFT | RIGHT | Success discordance B/L/R/N | Endpoint aligned | V3 taxonomy C/P/T/W/R | Future interface |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| π0.5 current stack | 5/27 | 24/27 | 4/1/20/2 | 25/27 | 29/6/14/5/0 | actions only |
| GR00T N1.7 | 3/27 | 0/27 | 0/3/0/24 | 21/27 | 3/49/2/0/0 | actions only |
| Cosmos3 Edge Policy DROID | 18/27 | 25/27 | 16/2/9/0 | 27/27 | 43/2/7/2/0 | 452 decoded futures |
| Cosmos3 Nano Policy DROID | 26/27 | 25/27 | 24/2/1/0 | 27/27 | 51/0/1/0/2 | 349 decoded futures |
| DreamZero action-guidance `s=2` | 3/27 | 17/27 | 1/2/16/8 | 25/27 | 20/20/14/0/0 | 54 official decodes; 2,554 latent futures |

`B/L/R/N` means both succeeded, LEFT-only, RIGHT-only, neither. `C/P/T/W/R`
means correct, pick failed, transport failed, wrong side, release failed. All
27 matched action-trace pairs differed for every completed checkpoint. The
summary artifacts retain Wilson intervals, continuous measurements, exact
paired tests, hashes, and infrastructure exclusions.

### Post-result π0-FAST compatibility cohort (V3-A002)

V3-A002 evaluates a public old-name OpenPI configuration without representing
it as the missing historical system. It pins OpenPI
`235044ed8a1502c0a18338eedc5d7adfe705af05` (tree
`03a4387bedbc0fa1467c367c60fc24e28b61ec6c`), config
`pi0_fast_droid_jointpos`, and RoboLab
`0aef241fb088ca21bb4ebd24448940ed56620d17`. Seeds 8310–8329 use the exact
static prompts quoted above and identical resets within every matched pair.

| Cohort | LEFT | RIGHT | Success discordance B/L/R/N | Endpoint ordering A/X/T | V3 taxonomy C/P/T/W/R | Future interface |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| π0-FAST public old-name config | 0/20 | 12/20 | 0/0/12/8 | 16/3/1 | 12/22/4/0/2 | actions only |

The direction imbalance is statistically identifiable within this cohort:
exact two-sided McNemar `p = 0.000488`; Wilson 95% intervals are `[0, 0.161]`
for LEFT and `[0.387, 0.781]` for RIGHT. The object-relative final lateral shift
(`RIGHT − LEFT`) was aligned with the requested ordering for 16 of 20 pairs,
with mean `−0.166 m` and median `−0.197 m`; an exact two-sided sign test
excluding the tie gives `p = 0.00443`. All 20 common executed-action prefixes
differed (mean RMS `0.335`, median `0.341`). RMS is descriptive in the native
mixed 8-D action coordinates and is not a distance or path-length measure.
Contact-timing instrumentation was unavailable in all 40 episodes and remains
null rather than being interpreted as no contact. There were no infrastructure
exclusions or runtime interventions.

The result supports prompt sensitivity and physical redirection, but not
symmetric directional competence: the same policy that redirected most
matched endpoints completed 12 RIGHT requests and no LEFT requests. Because
the runtime identity and denominator are distinct, these counts must not be
combined with the earlier 10-pair π0-FAST evidence.

The bounded [seed-8311 paired actual rollout](../artifacts/vla_wam_shared_v3/media/pi0_fast_old_name_config_v3a002/pi0_fast_v3a002_seed8311_paired_actual.mp4)
shows a matched LEFT failure and RIGHT success with both exact prompts printed
in the video. Its [media manifest](../artifacts/vla_wam_shared_v3/media/pi0_fast_old_name_config_v3a002/media_manifest.json)
binds the source-video hashes and H.264 publication output. All 40 full raw
viewport videos remain on the ali-owned PVC; π0-FAST is action-only and has no
imagined-future video.

## RoboTwin Phase A

Each model contributes **63 new matched pairs / 126 valid V3 episodes**:
seven scenes × nine new sampling replicates × two directions. Replicates are
nested within scenes and are not 63 independent scenes. Each model's seven
preserved V2 r00 pairs are reported separately and never merged with V3.

| Model | LEFT | RIGHT | Success discordance B/L/R/N | Endpoint aligned | V3 taxonomy C/P/T/W/R | Future interface |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Efficient-WAM-RT | 26/63 | 28/63 | 7/19/21/16 | 42/63 | 54/28/38/6/0 | 126 decoded futures |
| FastWAM | 24/63 | 20/63 | 1/23/19/20 | 39/63 | 44/31/36/15/0 | action-only test interface |
| LingBot-VA | 19/63 | 19/63 | 1/18/18/26 | 47/63 | 38/9/64/15/0 | 126 latent-only futures; no decoded video |

All 63 matched action-trace pairs differed for every model. Historical r00
coverage is Efficient-WAM-RT LEFT 3/7 and RIGHT 2/7, FastWAM 1/7 and 1/7,
and LingBot-VA 3/7 and 4/7; these are coverage layers, not additions to the V3
denominators.

## Scientific boundary

The completed data support a narrow result: changing the static language often
changes the executed trajectory and frequently redirects the endpoint, while
requested task completion and failure mode remain checkpoint- and
direction-dependent. Language sensitivity is therefore not equivalent to
reliable directional control.

The separate π0-FAST bridge sharpens that distinction: 20/20 action responses
and 16/20 endpoint redirections coexist with a 0/20 versus 12/20 success split.
This is evidence of strong direction-conditioned behavior and a large residual
directional asymmetry, not robust bidirectional control.

Phase A does **not** identify training distribution, geometry, reachability,
starting side, or object role as the cause of an asymmetry. V3-B001 now
provides a frozen test of whether Nano's directional-bias contrast changes
under one positions-only movable-object reflection, but its release is not a
behavioral result. Even a future interaction would not by itself identify
training data as the cause or establish full-scene symmetry. An exposed
prediction is also not evidence that the prediction caused successful
execution.

## Historical identity blocker and unreleased work

π0-FAST has 10 preserved V2 matched pairs (20 cells) and 20 frozen
`blocked_pi0` pairs (40 cells, seeds 8310–8329) under the exact historical
identity. Releasing those original rows would require recovery of OpenPI commit
`9e46d3aea26417bfb564227734b95d010aa827e5` and RoboLab commit
`11142d4319e44401e0464866bb5fedf7ec8a8927`. The current-stack V2-A008 probe
returned identical LEFT/RIGHT actions (RMS 0.0), so it is not a substitute.
The completed V3-A002 compatibility cohort is also not historical recovery and
must remain separate. It is complete and must not be rerun.

- Phase B: Nano V3-B001 alone is released as 108 exact cells after model-blind
  calibration; every other Phase-B ablation remains unreleased.
- Phase C: 480 registered episodes, not released; each wording requires its
  independent byte-hash, repeat, and prompt-sensitivity gates.
- Phase D: not released; it requires an effective stochastic-seed probe for
  each exact runtime.

## Restart checklist

Read this file, the V3 continuation state, and the V3 protocol before the older
V2 handoff. Raw outputs remain under `/data/users/ali/vla_wam/raw/v3` on the
ali-owned PVC; checkpoints and environments remain outside Git.

```bash
git status --short
.venv/bin/python tools/validate_vla_wam_v3_protocol.py
.venv/bin/python tools/validate_vla_wam_v2_protocol.py
git diff --check
```

Do not rerun a valid Phase-A or V3-A002 cell. V3-B001 is the only released
Phase-B queue; launch only its exact hash-bound rows after the required live
identity, viewport-writer, and raw-output-path rechecks. Use the eight original
committed summary/evidence-manifest/infrastructure-ledger triplets plus the
separate V3-A002 triplet under `artifacts/vla_wam_shared_v3/results/` for
completed-result analysis. Do not infer current experiment state from the
older article, website, gallery, figures, or chat.
