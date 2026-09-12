# Ablation specification: world-model predictions and executed motion

Version 1.0 · 12 September 2026 · Draft specification; no model runs released.

**Research question:** When an object arrangement changes, does a world–action model predict the motion produced by its own actions?

**Decision:** Use two model configurations and 24 new original/reflected layout pairs. Run one fixed-duration episode per command, model and layout. This is **192 confirmation episodes**, plus **32 development episodes** and **up to eight recording-pilot episodes**. The maximum core workload is **232 new episodes**. Most ablations below reuse these recordings and require no additional inference.

The scientific matrix is fixed here. Physical coordinates, frame/time mappings and execution-machine details must be verified before launch; they are explicitly listed in Section 8. This is not a claim that the recordings, annotations or new experiments already exist.

Companion files:

- [Machine-readable spec](../experiments/forecast_layout/ablation_spec.json).
- [All 232 planned core cells](../experiments/forecast_layout/planned_cells.csv).
- [116 optional guidance cells, not selected](../experiments/forecast_layout/optional_guidance_cells.csv).
- [Existing-evidence inventory](../results/existing_ablation_inventory.json).
- [Scientific rationale and analysis details](superpowers/plans/2026-09-12-world-model-paper.md).

This specification replaces the earlier choice of 12–24 confirmation layout pairs with **24**. Reducing that number requires a documented design revision before confirmation begins.

## 1. What we already have

All historical records below were checked against source commit ce561e66f82e95055e39d3d7711691982f6b2086. Counts are completed episodes, not independent scenes. Results are kept in their original cohorts; the table is not a pooled benchmark.

| ID | Completed experiment | Design and result | Use in this paper |
| --- | --- | --- | --- |
| H01 | Nano g3 position reflection, V3-B001 | 108 episodes: 27 blocks × original/reflected × LEFT/RIGHT. Original 52/54 successes; reflected 50/54. 738 decoded-future request records. | Existing behavioral result and prediction-recovery source |
| H02 | DreamZero custom action guidance s2 position reflection, V3-B003 | 108 episodes with fixed effective noise 1140. Original 13/54; reflected 50/54. | Existing behavioral result; never label as default DreamZero |
| H03 | Nano guidance pilot, V2-A015 with V2-A011 baseline | Six g3 baseline episodes: 6/6 successes. Six g1 episodes: 4/6. Three paired labels × two commands per configuration. | Already-completed exploratory guidance ablation |
| H04 | DreamZero guidance pilot, V2-A015 | Six conditional-action-equivalent s1 episodes: 3/6 successes. Six custom s2 episodes: 4/6. | Already-completed exploratory guidance ablation |
| H05 | Nano bowl-position sweep, V3-B005 | 210 episodes: seven prescribed bowl-y levels × 15 blocks × two commands. LEFT 93/105, RIGHT 99/105. 1,423 decoded-future records. | Existing dose-response evidence and development source |
| H06 | Nano graded symmetry, V3-E004 | 2,246 episodes at five prescribed levels. Endpoint levels each have 521 command pairs; intermediate levels each have 27. | Supporting geometry study, not 2,246 scenes |
| H07 | DreamZero-s2 symmetry, V3-E004 | 108 episodes. Reference LEFT 3/27, RIGHT 17/27; symmetric LEFT 26/27, RIGHT 25/27. | Separate replication context; not H02 |
| H08 | Cosmos Edge symmetry, V3-E004 | 108 episodes. Reference LEFT 18/27, RIGHT 23/27; symmetric LEFT 25/27, RIGHT 26/27. | Supporting third WAM; no new Edge runs in core |
| H09 | Nano four phrasings, V3-C001 | 160 episodes: 20 blocks × four phrasings × two commands; 134/160 successes. | Existing wording control; do not rerun a wording sweep |
| H10 | Cosmos Edge four phrasings, V3-C001 | 160 episodes with the same factorial; 123/160 successes. | Existing wording control, separate checkpoint |
| H11 | Nano V3 Phase-A direct commands | 54 new episodes, excluding the six historical V2 baseline episodes. 51/54 successes. | Additional development/recovery source |
| H12 | DreamZero-s2 V3 Phase-A direct commands | 54 new episodes; LEFT 3/27, RIGHT 17/27. 2,554 latent-future records and 54 official full-reset decodes recorded. | Recovery source; latent records are not per-request decoded RGB |
| H13 | Cosmos Edge V1 forecast corpus | 80 episodes, 752 action/forecast chunks and 3,008 cached frames. Historical scorer replay exists. | Evaluator development only; no validated forecast-accuracy result |

Important boundaries:

- H03/H04 include previously collected baselines; do not count those baselines again when totaling another cohort.
- H06–H08 change object inventory between symmetry level 0 and positive levels. The inventory-matched graded comparison uses levels 0.25, 0.5, 0.75 and 1. This is not the same intervention as position-only reflection. No reported equivalence claim passes.
- H05 has seven controlled positions, not 210 independent arrangements. V3-B004 was a failed fixture preparation; H05 uses the completed V3-B005.
- H09/H10 direct/short/outcome/desired-plus-negated-opposite phrasings are already observed. New confirmation uses only the two literal direct commands.
- The V1 historical agreement score is not prediction fidelity: its temporal/geometric comparisons and missing execution evidence remain unresolved.
- No historical collection above is an unseen confirmation set for the new forecast metric.

### Availability versus existence of records

Compact counts, summaries and many trace fields are accessible through pinned Git objects. Nano H01 retains step positions in its episode table. DreamZero H02 requires the original source episode files for the full step traces.

Raw H01/H02 files referenced under the old execution paths are absent on this machine. The checked references comprise 2,341 distinct absolute raw paths for H01 and 540 for H02. Their manifests are recovery pointers, not proof that the original host still has the files.

Four H01 and six H05 comparison videos are tracked in the pinned repository but absent from this sparse working tree. They concatenate local forecasts and may hold final frames for presentation. They must not be used as scientific timing maps.

The inventory JSON provides exact source paths and Git object identities for each table entry.

### Existing analysis that still needs packaging

The review checked Nano before either paired episode terminates and found ordered LEFT/RIGHT offsets in 27/27 pairs in both layouts. Treat this as a retrospective sensitivity analysis until reproduced by a committed analysis script.

Early fixed steps contain little cube motion; later steps lose early-terminating episodes. The historical check does not replace the new fixed-duration control.

**What is missing:** validated physical-time pairing of generated and execution images; independent image labels; qualified default/conditional DreamZero predictions in the proposed design; and a prospective varied-layout test of forecast accuracy.

## 2. Models and constants

| Config ID | Primary new configuration | Constants |
| --- | --- | --- |
| N3 | Cosmos3-Nano-Policy-DROID, g3 | Checkpoint revision 6706d7680581c255ff61e0f3bb49d90eac55c79e; guidance 3; four denoising steps; shift 5; history length 1; conditioning FPS 15; resolution setting 480; state input; joint-position actions; 32×8 returned chunks; execute up to 32 actions before replanning |
| D1 | DreamZero-DROID, conditional-action path | Checkpoint revision 96ad344138c66e82536422432ad742f015784942; video guidance 5; 16 configured inference steps, with actual cache/step behavior recorded; executed horizon 8; effective noise 1140; no custom s2 action extrapolation |
| D2 | DreamZero-DROID, custom negative-branch action guidance s2 | Optional extension only. Same D1 constants where verified; change action guidance from conditional-equivalent s1 to s2 |

N3's values are required by the historical V3-B001 server validator. Fifteen conditioning FPS and 32 returned actions do not themselves establish a video/action-time mapping.

D1 must use the official conditional-action path. A patched scale-1 path may be used only after its actions and generated latent outputs are verified against the official path on identical inputs. Otherwise it is a separately described configuration and cannot silently occupy D1 cells. D2 is not an automatic fallback if D1 fails qualification.

D2 guidance changes action flow during denoising, not just a final action tensor. It does not hold the generated future fixed. A D1/D2 study can test guidance sensitivity but cannot be described as a fixed-video intervention.

Held fixed within a comparison:

- Robot, cameras, nonmovable scene geometry, assets, model checkpoint, inference settings and control timing.
- The same starting state for LEFT and RIGHT in each layout, verified by reset and initial observation records.
- Reflection changes cube, bowl and banana center positions (x,y,z) to (x,-y,z) in robot coordinates. Initial orientations use the same sources; any post-settling differences are recorded, not falsely called held fixed.
- Static commands, with exact punctuation:
  - Put the Rubik's cube to the left of the bowl.
  - Put the Rubik's cube to the right of the bowl.
- One model execution per cell. Nano uses a matched effective policy seed within each four-condition layout block. DreamZero's 1140 is fixed; block labels are not independent noise draws.
- Stop after 450 actions, logging success as an event rather than ending at first success. Preserve model-failure traces and separately identify technical invalidity or a safety abort.
- No oracle coaching, prompt changes or simulator-state input beyond the model's declared observation interface.

The shared action count is a within-model diagnostic. It does not establish equal physical duration or compute between N3 and D1.

## 3. Exact core run matrix

| Stage | Layout blocks | Configurations | Conditions per block/config | Episodes |
| --- | ---: | ---: | ---: | ---: |
| Q: recording pilot | One historical original/reflected pair, P00 | N3, D1 | Original LEFT, original RIGHT, reflected LEFT, reflected RIGHT | 8 |
| D: development | Four new pairs, D01–D04 | N3, D1 | Same four conditions | 32 |
| C: confirmation | 24 new pairs, C01–C24 | N3, D1 | Same four conditions | 192 |
| Maximum core | | | | **232** |

Pilot Q can be waived per configuration only if an already verified recording demonstrates the same new runtime, goal-independent stopping and synchronized camera logger in all four conditions. Old success-terminated episodes alone do not satisfy that waiver. A waiver reduces the count; no waived cell is recorded as an executed experiment.

Development determines measurement feasibility, not the desired result. Its four layout pairs and the historical scenes are excluded from C01–C24. Confirmation has exactly 24 independent base-layout units per model, each with its reflection and both commands.

Run the four conditions as an intact block on the same isolated worker type. Fully reset and verify all model temporal/cache state before every episode, as well as resetting the simulator; a shared server must never carry one condition's context into the next. Use all 24 permutations of the four-condition order exactly once over the 24 confirmation blocks, assigned by a frozen hash of the study namespace and block ID. N3 and D1 use the same assigned order for a given layout block. Freeze ordering before any confirmation request. Do not interleave episodes on a DreamZero server whose temporal state is global.

Candidate Nano effective seeds: 2026091000 for P00; 2026091101–2026091104 for D01–D04; 2026091201–2026091224 for C01–C24. Check that the runtime accepts them and that the intended seed keys are unused in the pinned study before release. These are chosen new seed values, not claims about historical runtime behavior. DreamZero remains 1140.

The CSV reserves all 232 scientific cells. It intentionally has no executable command and marks every row not released. Its row order is an inventory, not the finalized execution order.

If only one model qualifies, the other model's entire development/confirmation branch remains not run; do not substitute a different model or configuration. One model then has 16 development and 96 confirmation episodes, plus its pilot or any already-spent pilot checks on the other model. Such a reduced study must be labeled explicitly.

### Nonbehavioral qualification requests

For each primary configuration use one saved observation and six forward requests: LEFT without decoding, repeat LEFT without decoding, RIGHT without decoding, and the same three inputs with decoding. Reset temporal context and effective noise identically for each comparison. The no-decode path may skip only rendering of retained latent outputs; it must not skip video prediction within a jointly denoised model. If the released interface always decodes, compare its standard output with offline redecoding of the same retained latent and report that the decode-toggle probe was inapplicable; do not invent an action-only model mode.

This is **12 generation requests across the two primary configurations**, not 12 robot episodes. If D1 uses a patched s1 implementation, add three official-path reference requests for equivalence. Decode toggles must not change actions; fixed-input repeats distinguish input sensitivity from nondeterminism. Count these calls separately from the 232 behavioral cells.

## 4. Required ablations and controls

A1/A2 are runtime interventions. A3–A9 are analyses of those same recorded episodes. They do not create independent datasets or multiply the sample size.

| ID | Comparison | Question answered | Extra episodes beyond Section 3 |
| --- | --- | --- | ---: |
| A1 | Original versus reflected object centers | Does scene layout change actual motion, and does the forecast track that change? | 0; built into the factorial |
| A2 | LEFT versus RIGHT from matched reset states | Does the instruction change executed behavior under each layout? | 0; built into the factorial |
| A3 | Generated future versus persistence | Does prediction improve over assuming no change? | 0 |
| A4 | Generated future versus constant velocity from prior observations | Does it improve over a simple observation-history extrapolation? | 0 |
| A5 | Common action-450 positions versus first-success positions from the same trace | How much does goal-dependent measurement time change the instruction-response conclusion? | 0 |
| A6 | Full uniform sample versus moving/stationary prefixes | Are apparent forecast gains dominated by easy stationary cases? | 0 |
| A7 | Relative cube–bowl motion versus cube and bowl motion separately | Is a relation change incorrectly attributed to cube transport? | 0 |
| A8 | Observable-case estimates versus full-layout missingness bounds | Could missing or ambiguous predictions reverse the conclusion? | 0 |
| A9 | Primary qualified horizon versus earlier supported horizons inside the executed prefix | Does performance change with forecast horizon? | 0 model episodes; may add annotation |

A3 replaces the prediction for scoring, not the world model inside the policy. It cannot identify a causal benefit of world modeling for action generation.

A4 uses the last two timestamped original observations before the forecast request. At the initial request it reduces to persistence. Because N3 uses history length 1, A4 has additional temporal information relative to that model input; identify it as an observation-history baseline, not an equal-input ablation. Missing prior images are explicit; do not estimate velocity from the future. The recording contract must retain prior frames for this comparison.

A5 is a within-trace measurement control. Define first-success endpoint as the first recorded success event, or action 450 if no success occurs. Define the common endpoint as action 450 for both commands. This is not a second physical rollout. Safety-truncated traces remain censored; do not carry the last state forward to action 450.

A9 compares H with exactly one earlier target: the earliest strictly positive qualified exposed time below H. If no earlier target exists, mark A9 unsupported for that configuration. This is a secondary analysis only when decoded targets already map to those physical times. Do not extend an action chunk to obtain a longer evaluable forecast. Do not use unsupported or unexecuted future frames.

Deliberately mismatched task-end scoring is not a required scientific ablation. Use wrong-time and post-reset pairings as validation tests that the scorer rejects, not as a second accuracy metric.

## 5. Optional intervention: guidance

We already have the small H03/H04 guidance pilots. No new guidance sweep is required for a paper about the prediction quality of fully disclosed configurations.

Enable a new D1 versus D2 comparison only if the paper will claim that custom action guidance changes the relation between prediction and execution. Decide before confirmation, not after observing the result.

| Additional D2 cells | Episodes |
| --- | ---: |
| Recording qualification on P00 if needed | 4 |
| D01–D04, both layouts and commands | 16 |
| C01–C24, both layouts and commands | 96 |
| Maximum additional | **116** |
| Core plus full D2 extension | **348** |

The already-planned D1 cells are the comparator; do not run or count a second copy of them. Interleave D1/D2 conditions contemporaneously within the same scene blocks using separate isolated contexts. Only action guidance may differ for an isolated guidance claim. If different code paths also change caching, scheduling or observations, report a configuration comparison.

The optional CSV lists these 116 cells separately; they are not selected or released by default. Selecting D2 also requires the same six nonbehavioral decode/repeat generation requests, unless exact-runtime qualification is already documented. These calls are separate from its 116 behavioral cells. If this extension is proposed after main confirmation has been inspected, the existing scenes support exploratory analysis only; a new independent confirmation design is required.

Other extensions are excluded from this version: Nano g1 replication, extra phrasings, additional WAM checkpoints, VLA comparisons, new embodiments, longer action chunks, training a world-model-free policy and a forecast-guided controller. Existing experiments already cover some of these axes; none is needed to answer the specified question.

## 6. Measurements, images and analysis

### Alignment

For each model choose one primary horizon H after physical mapping is verified: the longest strictly positive exposed target time within its unchanged executed action prefix for which matching original camera observations can be recorded. Freeze H before confirmation. If models support the same physical H, use it; otherwise report separate within-model studies and no accuracy ranking.

The action prefix is at most 32 actions for N3 and eight for D1/D2, further reduced by truncation. Generated frame number is not action number. DreamZero's recorded interface may return 24 actions while only eight execute; the remaining future is not eligible for the primary comparison.

Use matching camera, crop and time. Require timestamp error no greater than half a control step and half a captured-frame interval, taking the smaller tolerance. Save the measured residual. If that tolerance cannot be met, retain the sample as time-ineligible; do not interpolate or fabricate an execution image.

### Primary endpoint

For selected request j, let r0 be the image-plane cube-minus-bowl centroid offset before execution, rhat the generated offset at H, and r the actual offset at H. Normalize coordinates by the image diagonal.

Forecast error = distance(r,rhat).
Persistence error = distance(r,r0).
Primary skill = persistence error − forecast error.

Positive skill favors the generated future. A4 uses rCV = r0 + H times the velocity measured only from the two preceding observations.

The main intervention contrast uses motion magnitudes q = distance(r,r0) and qhat = distance(rhat,r0). Within each command/layout pair, compare reflected-minus-original mean q with the corresponding mean qhat. Show the paired discrepancies across scenes. Matching average magnitudes cannot establish correct directions or individual predictions; the per-request error and separate cube/bowl measurements remain necessary.

Actual instruction response uses simulator-relative cube–bowl positions at the common action endpoint; include first-success endpoints as A5. Do not treat monocular image offsets as full 3D success, contact or release.

### Sampling and labels

Select up to four requests per episode by a frozen, uniform, metadata-only draw. Freeze the algorithm and seed before confirmation; generate the actual request-selection manifest after execution provides the request inventory and before any image labels are inspected. Eligibility uses camera/time/action-prefix correspondence, not object visibility or forecast quality. A selected unclear image is not replaced by a clearer request.

Two independent raters mark object centers under a rubric fixed on development images, without counterpart images, model labels, instructions or success outcomes. Adjudicate disagreement and report it. Keep missing objects and uncertainty explicit. Use the 95th percentile of duplicate development label disagreement in relative coordinates as the operational movement-resolution threshold for A6; it is a measurement convention, not a physical no-motion test.

The primary observation distribution is uniformly sampled request starts within each episode. Average within episodes, then equally over four conditions, then equally over independent layout pairs. Use 10,000 paired-layout bootstrap resamples, a fixed analysis seed, and two-sided 95% intervals. Report both models separately and both primary intervals; do not pool them or promote only the favorable model.

A continuous four-condition mean is undefined when any condition has zero observable samples. Report the continuous estimate on fully represented layout pairs with its contributing count. Never average only the remaining conditions without disclosure.

For the full planned design, compute lower/upper bounds on the fraction of comparisons where the forecast strictly beats persistence: treat unresolved cases first as non-wins, then as wins. Preserve episode, condition and layout weighting; a zero-eligible episode contributes [0,1] uncertainty. This does not impute continuous object positions. Report coverage and censoring by condition.

### Annotation workload

Confirmation alone has at most 192 × 4 = 768 sampled requests.

- Current / predicted / executed images: at most 2,304 images, 4,608 individual judgments by two raters.
- A4 may require one additional preceding image per request: maximum **6,144 judgments** total before adjudication.
- Across all 232 planned core episodes, the four-image upper bound is **7,424 judgments**, excluding archive demonstrations, extra A9 horizons and repeated adjudication.
- If A9 is supported for every sampled request, its earlier prediction/execution pair adds at most two images per request: the total becomes **9,216 confirmation judgments** or **11,136 judgments across all 232 core episodes**, before adjudication. These are workload ceilings, not an assertion that every image requires a new label.
- Identical images can reuse a label after identity checks. Record actual throughput during development.

No new video-generation calls are required for A3–A9, but additional labels or offline decoding may consume resources. State that cost separately.

## 7. Deliverables and completion rules

Before new model execution:

1. Correct model labels and package the existing stopping sensitivity analysis.
2. Recover and validate one complete Nano and one DreamZero archive episode, or record exactly which source assets are unavailable.
3. Resolve the nonbehavioral runtime checks and Q recording checks.
4. Freeze the design, scene manifest, statistical code and budgets after development.
5. Only then release C01–C24.

Required paper outputs:

- Actual scene and frame/action-time diagram.
- Per-model baseline errors, forecast skill, full coverage and layout-cluster uncertainty.
- Predicted-versus-executed reflection contrasts, not just a single success-rate table.
- The A5 stopping control and A6/A7 explanations.
- Model/configuration and sample-size table; representative videos chosen by a declared rule.

A successful study does not require positive forecast skill. A repeatable mismatch is publishable evidence if it is measured correctly and distinguished from alignment or annotation error. An imprecise result does not establish equivalence or absence of information.

## 8. Exact prerequisites still unresolved

These are launch blockers with defined outputs, not permissions to invent values:

| Item | Required output before its dependent stage |
| --- | --- |
| Raw recovery | Original files matching recorded identities, or an explicit unavailable result |
| Model/runtime restoration | Actual source/weight/patch/config identities and declared effective seed behavior for N3/D1 |
| D1 equivalence if patched | Matching official-path action and latent outputs on the fixed input probes |
| Physical scenes | Frozen asset identities and pose arrays for P00, D01–D04 and C01–C24; visibility/collision/reset checks; both success predicates false after settling; matched command resets; candidate acceptance blind to target-model outcomes, with rejected-candidate list |
| Forecast mapping | Per-model generated index → physical time → executed action index mapping, H and actual timestamp tolerance |
| Camera mapping | Primary camera, resolution/crop mapping and control-step capture synchronization |
| Repeatability | Isolated temporal contexts; fixed-input repeats; documented reset/observation/simulator variation |
| Annotation | Before confirmation: frozen landmarks, ambiguity rules, operational movement threshold, draw algorithm and seed. After execution but before labeling: request-selection manifest from recorded metadata |
| Execution order | Assigned block permutations and seed-compatibility/collision check |
| Resources | Measured runtime, memory, storage and annotation costs; selected execution host and bounded run authorization |

Model-blind physical qualification may use simulator state to check fixtures; policy episodes retain only declared model inputs. Do not edit old V2/V3 registries or replace old valid episodes. All new work has a workshop-local namespace.

This spec is complete as a scientific run matrix. It is deliberately not an executable release: no GPU access, new policy episode, annotation collection, or paper submission occurred while writing it.
