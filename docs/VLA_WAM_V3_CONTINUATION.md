# VLA/WAM v3 expansion continuation

Updated: 6 August 2026, after the complete Phase-A launch-authorized queue, the
separately authorized π0-FAST compatibility bridge, the cross-version
measurement-coverage audit, and the completed 27-seed Nano V3-B001
position-reflection ablation were compiled. π0.5 V3-B002 is now registered
before inference, and the existing-log failure-mode split is complete. The
machine-readable source of truth is
[`continuation_state.json`](../artifacts/vla_wam_shared_v3/continuation_state.json).

## Status

All **648 cells marked `authorized_new`** in the frozen Phase-A queue are
complete valid evidence: 270 DROID/RoboLab episodes and 378 RoboTwin episodes.
The queue's 40 `blocked_pi0` rows remain frozen under their unavailable exact
historical runtime identity. Amendment V3-A002 instead authorized a distinct,
publicly reproducible π0-FAST compatibility cohort: all **20 matched pairs / 40
episodes** are now complete valid evidence. That cohort is never pooled with
the blocked rows or preserved V2 evidence.

No Phase-A, V3-A002, or Nano V3-B001 inference remains. Nano V3-B001 is complete
at **27 matched seeds / 108 valid behavioral episodes** under one hash-bound
runtime identity. Its release followed model-blind calibration and occurred
before any Phase-B model request or behavioral episode. π0.5 V3-B002 is a
separate 108-cell replication registered with zero V3-B002 requests and zero
V3-B002 behavioral episodes; its model-specific runtime gate is the exact next
step. Every other Phase-B ablation, all Phase-C wording cells, and all Phase-D
stochastic repetitions remain unreleased.

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

| Nano release/runtime artifact | SHA-256 |
| --- | --- |
| [`model_blind_calibration_report.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/model_blind_calibration_report.json) | `112716acada89050561c9488d93a333a300b1675c2305329c8e5aceeb4e6da71` |
| [`nano_mirror_v3b001_cells.jsonl`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl) | `018b8b6ae76ac46f2f89eef83c4b16d7a4ff3d1ff15d91527b96fb56b5432c5a` |
| [`nano_mirror_v3b001_manifest.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_manifest.json) | `5c82268739feb41281435a51dcd848b575218cd9fbe5839d9ad130d1a7888830` |
| [`post_result_nano_mirror_v3b001_amendment.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/post_result_nano_mirror_v3b001_amendment.json) | `9d88c29733fa3b24a154977bc25d04d2d77df5be59e3213f0c3a6cfbe3edc6a0` |
| [`live_reset_semantics_correction.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_reset_semantics_correction.json) | `167494ff48b075c41ff64fce4c18c78ae6650d1bcfd63d0646cf386a0a82875b` |
| [`live_reset_preflight_report.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_reset_preflight_report.json) | `075a3d78fff99a74bfbb3981e77c20b6de20a9d5123e5ed4f6dea10ee2f61bc9` |
| [`live_queue_snapshot.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_queue_snapshot.json) | `bfd8f0e095fc83dd2d09ba254c0860d8c2d8bbdf02205173369c50e56013e52c` |
| [`live_infrastructure_ledger.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_infrastructure_ledger.json) | `33c83da878782f77aca60a95d0c7382982af33b2210403d96c17637d41cc0418` |
| [`nano_v3b001_results_manifest.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_results_manifest.json) | `ab9b9849e04bc15e65ed9d8d55d13e8aad62dbef5f5e2fa16b8eaf486a6ea517` |
| [`nano_v3b001_summary.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_summary.json) | `f43636a03caade5f3dc65de6736808c8257c78eacb07ba4cb963bfc6a0e36578` |
| [`nano_v3b001_final_evidence_manifest.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_final_evidence_manifest.json) | `ed16c120f58b89fe67227c544a7ac7b20610e802a0a19276fcb0f7796cef5270` |
| [`nano_v3b001_publication_media_manifest.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/publication_media/nano_v3b001_publication_media_manifest.json) | `2a9023b976b6a877b2da266ce1333f6639d89a971a4e082f50a7af1f0cf246ea` |

The release files authorize the exact queue under the verified runtime identity.
The two `live_*` evidence files remain immutable historical records of the
12-cell prefix. Completed-result analysis must use the final results manifest,
summary, aggregate JSONL, and hash-closed evidence manifest.

### Nano V3-B001 completed result

V3-B001 completed all **108/108 prespecified cells**: 27 matched seeds, two
layouts, and the exact LEFT/RIGHT prompt pair. All 108 valid behavioral
episodes—including every model failure—enter the denominator. The outcome
taxonomy is 102 `correct`, five `transport_failed`, one `release_failed`, and
zero `pick_failed` or `wrong_side`. Condition-level success is:

| Layout | LEFT | RIGHT |
| --- | ---: | ---: |
| Control | 26/27 | 26/27 |
| Position-reflected | 27/27 | 23/27 |

The full-sample primary estimands use signed final lateral offset for every
seed, including failures. Changing the prompt redirected endpoints in the
requested LEFT-to-RIGHT order for **27/27 control seeds** and **27/27
position-reflected seeds**. Median separation was `+46.7 cm` in control and
`+43.7 cm` after position reflection. Their prespecified interaction was small:
median `+0.7 cm`, mean `+0.5 cm` with paired-bootstrap 95% CI `[-5.6, +6.6] cm`,
and exact two-sided sign-test `p = 0.701`.

The requested-side depth contrast changed sharply. In control, RIGHT finished
deeper in its requested side than LEFT by median `+14.8 cm`; after reflecting
movable-object center positions, that contrast reversed to median `-8.8 cm`.
The prespecified reflection interaction was median **`-24.6 cm`** and mean
`-24.8 cm` with paired-bootstrap 95% CI `[-32.4, -17.3] cm`; **24/27** seeds
were negative and the exact two-sided sign-test was **`p = 4.923e-05`**.

The success-conditional secondary analysis contains only the named 21-seed
all-four-cells-correct subset. It does not encode failures as zero or combine
unmatched successful cells. Its requested-margin interaction was median
`-22.8 cm`, mean `-21.2 cm` with 95% CI `[-29.1, -13.5] cm`, 18/21 negative,
and exact sign-test `p = 0.00149`.

The supported inference is precise: under this checkpoint and simulator, the
static language probe robustly redirected endpoints in either layout, while the
registered movable-object position reflection changed which requested direction
received greater side depth. This establishes a causal effect of that physical
intervention on the measured contrast. It does **not** identify training data as
the mechanism, isolate reachability from downstream physical mediators, or test
a full-scene mirror, base rotation, start-side interaction, or role swap.

The compiler retained 738 decoded local predictions across 21,972 executed
actions. The bounded publication clips show the complete actual rollout beside
every exposed 33-frame request-local prediction in order; the stitched right
panel is not a continuous full-task imagination and does not receive a success
score.

#### Historical live-boundary and reset audit

The committed live snapshot freezes the first three complete four-cell seed
blocks (9400–9402): 12 valid behavioral cells, 10 `correct`, one
`release_failed`, and one `transport_failed`. It is retained as a historical
prefix, not current study state. Four earlier complete behavioral attempts were
invalidated before analysis, and all 39 prior model-sampler requests remain
infrastructure evidence in the hash-bearing
[`live_infrastructure_ledger.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_infrastructure_ledger.json).

Attempt 10 used all four wrappers in calibration order, yet its control-LEFT
reset failed the frozen 3 mm gate before any model request. Registration context
is therefore falsified as the sufficient explanation. The confirmed defect is
the interaction between RoboLab's episode-length mutation order and the live
proxy. The pinned runner calls `env.reset()` twice. After call 1, the
pre-correction proxy zeroed `episode_length_buf`; call 2 then saw `ep_len=0`,
entered RoboLab's `artifact_ids` branch, and performed a fresh second physical
reset. Ordinal-1 live calibration evidence had validated the analytic frozen
fixture coordinates; the pre-correction gate instead attested ordinal 2 and
observed the `0.0034789443 m` error.

Calibration repeat semantics are now disclosed precisely. Repeat 0 was one
fresh physical reset followed by 60 settle and 15 stability steps. Repeats 1
and 2 saw `ep_len=75`; RoboLab marked them frozen before computing active IDs,
so they did not perform fresh physical resets. Their near-identical states are
valid model-blind stability observations, but not three independent resets.

The prospective operational correction is frozen in
[`live_reset_semantics_correction.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_reset_semantics_correction.json).
The exact bridge must receive two runner reset calls but perform one physical
reset and one settle gate. Call 1 evidence is provisional and not persisted.
The immediate call 2 is idempotent, returns cached observation/info, keeps the
frozen flags false, and persists the final `2/1/1/true` attestation. That
attestation must bind `[75] → [0]` and zero model requests. Prompts, seeds,
analytic fixture coordinates, the success predicate, 60+15 steps, and the 3 mm
tolerance are unchanged.

Attempts 08 and 09 remain excluded server failures: a nonwritable Triton home
(`247d772c…`) and NFS cleanup under a PVC-backed `TMPDIR` (`ff2159bf…`).
Persistent caches stay on the ali-owned PVC; server `TMPDIR` is pod-local, and
`OMNI_KIT_ACCEPT_EULA=YES` remains explicit. Attempt 10's first mirrored-RIGHT
cell compiled provisionally as a success after 169 actions. Its raw JSONL
(`a51380c9…`), video (`92e26891…`), and manifest (`bca5342c…`) are preserved but
invalidated because the runtime source changes. The zero-request control
failure retains fixture (`573f1df5…`), settle (`588297f9…`), and bridge
(`1f921e18…`) evidence.

The zero-sampling exact-bridge preflight has now passed under runtime-attempt-08
identity `2c5e314a…`, bound to clean pushed commit `39f19b57…`; the manifest file
hash is `a609300c…`. The bridge accepted a transport request packet on each
layout and received the explicit no-inference response. No model server or
checkpoint was loaded, no model sample was drawn, and no action was executed.

Both the first released layout
`v3b001:nano:seed9400:position_mirrored:right` and a control-LEFT diagnostic
invoked through the exact `cell_plan` + `run_cell` path passed the final
`2/1/1/true` reset attestation with `[75] → [0]`, zero requests during the gate,
matching positions, and a neutral reset relation. The maximum position errors
for mirrored RIGHT were `0.244`, `0.420`, and `0.609 mm` for banana, bowl, and
cube. Control LEFT measured `0.244`, `0.022`, and `2.758 mm`, respectively,
remaining within the frozen 3 mm tolerance. The control diagnostic did not
advance the frozen queue. Complete hashes are retained in
[`live_reset_preflight_report.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_reset_preflight_report.json).

The completed queue used only that verified runtime identity. The preflight's
mirrored-RIGHT cell became one of the 108 valid rows. Split ali-owned RTX pods
were an operational load-isolation choice, not a scientific factor.

Within the three complete blocks, condition-level requested success is 2/3 for
control LEFT, 2/3 for control RIGHT, 3/3 for position-mirrored LEFT, and 3/3 for
position-mirrored RIGHT. These remain descriptive historical-prefix counts and
must not replace the completed result above. Signed final lateral offset and
requested-side margin are retained for all 12 cells in
[`live_queue_snapshot.json`](../artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_queue_snapshot.json).
The prefix's all-four-cells-correct margin subset contains only seed 9400
(`n=1`); no inference is made from that prefix.

### π0.5 position-reflection registration (V3-B002)

V3-B002 is the exact scene-matched π0.5 replication of the Nano design: the
same 27 seeds `9400–9426`, the same control and position-reflected fixtures,
and the same randomized four-cell execution order within every seed. The
frozen queue therefore contains **27 matched seeds × 4 cells = 108 episodes**.
No seed was resampled. Each seed keeps all four cells under one runtime lane.
The prompt bytes remain:

> Put the Rubik's cube to the left of the bowl.

> Put the Rubik's cube to the right of the bowl.

The registration records three predictions before π0.5 V3-B002 inference.
H1 predicts an endpoint-redirection interaction near zero, following Nano's
mean `+0.5 cm` result (`p = 0.701`). H2 predicts a strongly negative
requested-side-depth interaction, following Nano's mean `-24.8 cm` result
(`p = 4.923e-05`). H3 is a two-sided exact test of whether reflection changes
π0.5's prior binary RIGHT-over-LEFT gap (`24/27` versus `5/27`); its direction
was deliberately not filled in after registration. Continuous interactions use
20,000 matched-seed bootstrap resamples with master seed `3104159`, medians,
and exact sign tests. H3 reports the per-seed difference-in-differences on
`{-2,-1,0,1,2}`, its exact within-seed permutation test, and the 2×2 success
table.

The independent existing-log failure-mode split is already complete and adds
no new inference. Counts below are `correct / pick / transport / wrong-side /
release`:

| Checkpoint | LEFT counts | RIGHT counts | Failure-only exact p |
| --- | ---: | ---: | ---: |
| π0.5 | 5 / 6 / 11 / 5 / 0 | 24 / 0 / 3 / 0 / 0 | 0.3822 |
| DreamZero | 3 / 10 / 14 / 0 / 0 | 17 / 10 / 0 / 0 / 0 | 0.001722 |
| Cosmos3 Edge | 18 / 1 / 6 / 2 / 0 | 25 / 1 / 1 / 0 / 0 | 0.6182 |

DreamZero's failure-only distribution differs by requested direction in this
cohort. π0.5 and Edge do not reject a shared failure shape, but their RIGHT
failure rows contain only three and two episodes; this is not evidence of
equivalence. Raw 162-row JSONL, row-normalized proportions, exact-test
enumeration, and all source/output hashes are retained under the V3-B002
analysis directory.

| V3-B002 registration/analysis artifact | SHA-256 |
| --- | --- |
| [`post_result_pi05_mirror_v3b002_amendment.json`](../artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/post_result_pi05_mirror_v3b002_amendment.json) | `8e56365fdc306adbef2d2f2f8357653d1a683ff17f879f624d7eefd4acb1abb6` |
| [`pi05_mirror_v3b002_cells.jsonl`](../artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_cells.jsonl) | `0db680a3aee04c991bcc78904cb572b7d962971e04fbc879e828354da30dafee` |
| [`pi05_mirror_v3b002_manifest.json`](../artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_manifest.json) | `8aaaa38302f6a654090250b3e12cd8735fab28a74027d50193325ffa9d0dddea` |
| [`failure_mode_split_episodes.jsonl`](../artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_episodes.jsonl) | `94d078c16d579019c216a7d05c0e4748d049d4bc91ab1879ca38b8bc78da8735` |
| [`failure_mode_split_report.json`](../artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_report.json) | `f6289333d77538ed9235a567cb98d70333dd50549f7e52d13bfe5d34a10bdf96` |
| [`failure_mode_split_manifest.json`](../artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_manifest.json) | `5a0e95af238536dbbcf7a6c81bf721f095cea1a5e178ddfbaecda8105a1e06ab` |

V3-B002 is registered but not behaviorally released. Before its first model
request, the new runtime identity must bind the exact OpenPI/RoboLab revisions,
checkpoint, registration manifest, fixture wrappers, and queue code, then pass
zero-request physical reset, fixture, real-renderer, and raw-writer gates plus
fixed-observation exact-repeat and prompt-sensitivity checks.

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
provides a completed test of whether Nano's directional-bias contrast changes
under one positions-only movable-object reflection. V3-B002 prospectively asks
whether the same intervention changes π0.5's continuous contrasts and its much
larger binary success gap; registration alone is not a behavioral result. Even
a future interaction would not by itself identify training data as the cause
or establish full-scene symmetry. An exposed prediction is also not evidence
that the prediction caused successful execution.

## Historical identity blocker and unreleased work

π0-FAST has 10 preserved V2 matched pairs (20 cells) and 20 frozen
`blocked_pi0` pairs (40 cells, seeds 8310–8329) under the exact historical
identity. Releasing those original rows would require recovery of OpenPI commit
`9e46d3aea26417bfb564227734b95d010aa827e5` and RoboLab commit
`11142d4319e44401e0464866bb5fedf7ec8a8927`. The current-stack V2-A008 probe
returned identical LEFT/RIGHT actions (RMS 0.0), so it is not a substitute.
The completed V3-A002 compatibility cohort is also not historical recovery and
must remain separate. It is complete and must not be rerun.

- Phase B: Nano V3-B001 is complete at 108/108 exact cells. π0.5 V3-B002 is
  registered at 108 exact cells with zero completed episodes and awaits its
  model-specific runtime release gate. Every other Phase-B ablation remains
  unreleased.
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

Do not rerun a valid Phase-A, V3-A002, or Nano V3-B001 cell. Nano V3-B001 is
complete at 108/108 valid episodes; the three-block live snapshot is historical
only. Use its final results manifest, summary, aggregate JSONL, and hash-closed
evidence manifest for analysis. V3-B002 is the exact next experiment, but no
cell may start before its new runtime/reset/renderer/writer/repeat/sensitivity
gate is hash-bound. All other Phase-B, Phase-C, and Phase-D cells remain
unreleased. Use the eight original committed
summary/evidence-manifest/infrastructure-ledger triplets plus the separate
V3-A002 triplet under `artifacts/vla_wam_shared_v3/results/` for earlier
completed-result analysis. Do not infer current experiment state from the older
article, website, gallery, figures, or chat.
