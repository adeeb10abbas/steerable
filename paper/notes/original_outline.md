> **Historical planning document only.** Several counts and claim formulations
> below were superseded by the evidence audit. Do not use this file as a source;
> use `EVIDENCE_MAP.md` and the current section files instead.

# Paper Outline — Writing Guide

**Working title:** Move the Objects, Move the Bias: Two Mechanisms Behind Directional Failure in Language-Conditioned Manipulation

**One-sentence thesis:** Directional left/right failure has two causes — scene configuration and a smaller residual policy prior — and single-layout binary success rates cannot tell them apart.

**Scope, stated everywhere:** all causal results are DROID/RoboLab, four checkpoints, simulation only, relational placement on one spatial axis.

---

## I. Introduction (\~0.75 page)

- VLAs are reported to ignore spatial instructions and follow visual shortcuts; remedies target language conditioning (CAG, CAST, language feedback policies).
- We test that premise with matched pairs: identical scene, seed, checkpoint, controller; one word changed.
- **Turn:** the premise does not hold. Policies respond every time and mostly move the right way. Completion still splits by direction.
- Corpus does not explain it — both channels audited, both near balanced.
- So we moved the objects. Reflection inverts the advantage; a lateral sweep grades it; symmetry removes most of it.
- **But not all.** In the powered cohort a residual survives and fails its preregistered equivalence margin.
- Contributions list (4 bullets: protocol, screening, causal interventions, audits).

**Watch for:** an NVIDIA scientist read the summary as "the model isn't learning language conditioning" — the opposite of Claim 1. Make the sensitivity-vs-competence distinction survive a fast read. Put it in the turn paragraph explicitly.

---

## II. Related Work (\~0.5 page, three paragraphs, no subsections)

- **VLAs and language conditioning.** π₀/π₀.₅, OpenVLA, world-action models. Note checkpoints span architecture families.
- **Counterfactual instruction failure.** CAG/LIBERO-CF, CAST, language feedback policies. Position: these assume language is the weak link. 
  - Distinguish spatial *reference* (position selects the object — their CF-Spatial, their identical-cup task) from spatial *relation* (position is the goal — ours). Scope your claim.
  - Their CF-Focused result supports you sideways: removing one distractor moved average grounding 39.0% → 64.4%, success 20.0% → 40.0%, same weights, same prompts.
  - Note they randomized object positions and still measured 20%/30% grounding on left/right. Randomization is not a geometry control.
- **Evaluation practice.** Binary success under a frozen predicate is the default. Show it is insufficient here.

---

## III. Method (\~1 page) — this is a contribution, write it as an instrument

- **Matched-pair intervention.** Exact prompt strings. One static instruction at reset. No oracle, no mid-episode switching, no progress-conditioned language.
- **Four measurements, four questions:** 
  - action distinctness → did the policy respond
  - paired endpoint shift (signed) → did it point the right way
  - task success (frozen predicate) → did it complete
  - requested-side depth (continuous) → how well, when success saturates
  - State the exact distinctness criterion once, normatively.
- **Failure taxonomy:** pick / transport / wrong-side / release / correct.
- **Statistical protocol:** Wilson for marginals; seed-level bootstrap (20k resamples) plus exact sign tests for paired contrasts; exact within-seed layout-label permutation for interactions; TOST with registered margins for nulls.
- **Discipline paragraph:** registered predictions before launch, frozen predicates, hashed artifacts, behavioural failures stay in denominators, NR never becomes zero, arenas never pooled. This is why the reversal is believable — do not bury it.
- **Cohorts and allocation rule.** Eleven checkpoints screened; five in the powered cohort; allocation by dynamic range on the measure being tested. State the rule *before* either table appears so uneven n reads as design.

---

## IV. Language Reaches the Policy (\~0.4 page)

- 324/324 matched pairs action-distinct (135/135 DROID, 189/189 RoboTwin).
- Endpoint ordering correct 125/135 and 128/189.
- Vision-shortcut account predicts near-identical traces. Opposite observed.
- **Caveat, load-bearing:** distinctness has no same-prompt baseline. Under a stochastic policy traces differ regardless. Forward-reference Section V.

---

## V. Prompt Effect: Magnitude and Orientation Dissociate (\~0.6 page)

- Fixed settled observations, matched sampling seeds, 336 requests, zero behavioural episodes. All 12 exact repeats bit-identical, RMS 0.
- π₀.₅: effect is 4% of same-prompt sampling variation (0/27 above noise p95), yet strongly oriented — cosine 0.540/0.652, permutation p < 1e-5.
- Nano: effect exceeds sampling variation (ratios 0.72/1.19) but nearly unoriented — cosine 0.027/0.079, control layout p = 0.167 (n.s.).
- DreamZero: deterministic at these observations, ratio undefined not infinite, 27/27 above p95.
- **Neither predicts closed-loop behaviour.** π₀.₅ has the smallest effect and orders 25/27; Nano's control-layout effect is indistinguishable from random orientation and orders 27/27.
- Reconciliation (small coherent bias accumulating over replanning) stated as untested hypothesis.
- Scope: action-space, not verified FK. 27 sampling draws at fixed states, not independent scenes.

---

## VI. The Directional Gap Is Not Stable (\~0.6 page)

- DROID: large in π₀.₅ (5/27 vs 24/27) and DreamZero (3/27 vs 17/27); inconclusive in Edge (p = 0.065); absent in Nano; reversed in GR00T at floor.
- RoboTwin: no gap in three checkpoints at 63 pairs each.
- **Revised interpretation — important.** Two of three RoboTwin checkpoints later failed an endpoint-redirection positive control. So the RoboTwin null is *uninterpretable*, not evidence of arena-dependence. Do not claim "doesn't replicate in a second arena."
- Nesting caveat: 63 pairs = 7 scenes × 9 replicates.

---

## VII. Reflecting Object Positions Reverses the Advantage (\~1 page) — CORE

- Design: 27 seeds × 4 cells = 108 episodes per checkpoint. Movable-object centres reflected about the sagittal plane. Robot, cameras, fixed geometry, prompts, seeds held constant.
- **Redirection invariant:** Nano interaction +0.5 cm (p = 0.701), π₀.₅ −1.12 cm (p = 0.701).
- **Depth reverses in all three:** π₀.₅ −34.6 cm (27/27 seeds, p = 1.49e-8), Nano −24.8 cm (24/27, p = 4.92e-5), DreamZero −14.1 cm (23/27, p = 3.11e-4).
- **Completion inverts in π₀.₅:** 4/27 vs 25/27 → 25/27 vs 9/27, DiD −1.37, exact p = 5.96e-8.
- State plainly why Nano (ceiling) and DreamZero (floor) cannot register a binary reversal; claim no binary effect there.
- Asymmetry note: reversal is \~88% of what perfect symmetry predicts. Report it — a reviewer who computes it and finds you silent will trust less.

---

## VIII. The Dependence Is Continuous (\~0.4 page)

- Seven registered bowl positions, 15 seeds each, 210 episodes, Nano.
- Slope 1.12 m/m, CI [0.72, 1.56], 13/15 seed slopes positive, p = 7.39e-3.
- Binary completion under LEFT declines 15/15 → 10/15; RIGHT stays at ceiling.
- **Keep the caveat:** fitted zero crossing is outside registered support, so no in-support crossing is claimed.
- **Run before publishing:** refit excluding the two extreme levels. If the slope survives on the middle five, the response is genuine; if it collapses, soften to "differs at the extremes of the tested range."

---

## IX. Three More Scene Factors (\~0.4 page)

- Target start-side: 26/27 vs 22/27 (target starts left) → 23/27 vs 27/27 (starts right); interaction +0.296, p = 0.0156.
- Role swap: depth contrast +0.13 m, CI [0.06, 0.21], p = 1.51e-3.
- Fixed-scene stochastic repeats: LEFT 41/216 vs RIGHT 197/216; seed-level mean difference +0.72, CI [0.68, 0.77]. Rules out policy noise.

---

## X. Neutralizing Geometry Removes Most of It (\~1 page) — CORE

- 4,096 registered episodes, 2,048 matched pairs, five checkpoints, symmetry residual 0.
- **π₀.₅, powered:** binary gap 74.2 → 20.2 pp, interaction −51.9 pp (p = 0.00786); depth 17.0 → 5.75 cm, interaction −12.4 cm (p = 0.00348). Endpoint redirection preserved (+23.9 cm, 335/341 positive pairs).
- **Residual survives:** binary 90% CI +15.0 to +25.2 vs ±15.56 margin; depth +4.85 to +6.66 cm vs ±4.15 cm. MDE gate passed — powered, not just non-significant.
- **Depth interaction significant in every checkpoint** (−0.124 to −0.192 m). Binary interaction only where a gap existed.
- **DreamZero eliminates completely:** +51.9 → −3.7 pp, interaction −0.556 (p = 0.00085). So the residual is π₀.₅-specific, not general. Protect this — it is what keeps the claim honest.
- **Nano is the ceiling case:** binary interaction exactly 0.0 (p = 1.0), depth interaction −0.139 m (p = 1.8e-4) over 1,123 pairs. Strongest single argument that binary success is insufficient.
- **Edge underpowered, not null:** −0.148, p = 0.40 at n = 27. Say underpowered.
- **Failure signature shifts:** π₀.₅ control 6 pick / 14 transport / 5 wrong-side → symmetric 0 / 1 / 13. Qualitative change, not a rate change.

---

## XI. The Corpus Does Not Explain It (\~0.3 page — compress hard)

- Language: 75,144 episodes × 3 annotations; 23,869 left vs 26,079 right (+9.26%); training subset +2.33% raw, +5.75% weighted. Two independent implementations.
- Motion: 57,639 episodes, 18,691,281 frames, pose only. Endpoint Δy 50.45/49.55; cumulative lateral path 50.022/49.978; language-neutral episodes slightly left-skewed.
- Claim only that one-for-one inheritance is inconsistent with the magnitudes. Conditional structure untested.

---

## XII. Limitations (numbered, reads as rigor)

1. Simulation only.
2. Causal results are DROID-only; two RoboTwin checkpoints failed the endpoint-redirection positive control, so cross-arena replication is open.
3. Reflection moved movable-object centres only — does not isolate embodiment handedness, camera symmetry, or action-decoder asymmetry.
4. Base-rotation control failed closed (wrist camera inherits base transform); no null was manufactured from it.
5. Model-blind reference controller failed at pickup in 108/108 — negative control, not a feasibility claim.
6. Residual demonstrated in one checkpoint under adequate power.
7. Checkpoint provenance not fully auditable.
8. One task family, one spatial axis, static instruction only. Closed-loop corrective language untested — cite the language-feedback-policy line.
9. Corpus audit is marginal.

---

## XIII. Conclusion (\~0.2 page)

- Language reaches these policies and redirects them with geometry-invariant magnitude.
- Whether redirection becomes completion depends largely on where the objects are, and partly on a residual policy prior.
- Recommendation: matched pairs, a continuous placement measure alongside success, and a geometry control before attributing a directional effect to a policy.

---

# Figures

| # Content Supports Notes  |                                                                                                        |              |                                                                                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------ | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1**                     | Three layouts (control / reflected / symmetric), same prompts, success bars under each                 | Whole thesis | Use real top-down renders, not schematics. Add failure taxonomy under panel 3. Caption: "Same policy, same prompts, same seeds — only object positions differ." |
| **2**                     | Reflection interaction, three checkpoints, per-seed dots + bootstrap CIs; depth panel and binary panel | §VII         | Already built. Widen to double-column.                                                                                                                          |
| **3**                     | Dose-response: depth contrast and binary completion vs bowl position, seven levels                     | §VIII        | Already built. Consider folding into Fig 1 as a 2×2.                                                                                                            |
| **4**                     | E004 symmetry: gap at s=0 vs s=1 per checkpoint, binary and depth                                      | §X           | **New — build this.** The powered result has no figure yet.                                                                                                     |
| **5**                     | Four measures on Nano — three read balanced, depth does not                                            | §X, §III     | Already built. This is what sells the instrument.                                                                                                               |
| **6**                     | Failure taxonomy stacked by direction, control vs symmetric                                            | §X           | Shows the qualitative shift.                                                                                                                                    |
| **7**                     | Cross-checkpoint screening, paired success with Wilson intervals, DROID and RoboTwin panels            | §VI          | Already built.                                                                                                                                                  |

**Supplement:** phrasing × direction (exploratory); FastWAM and LingBot-VA positive-control failures; E002 model-blind controller; graded Nano dose-response; full checkpoint tables; geometry and visibility gates.

**Cut first if over length:** Figure 7, then Figure 3 (fold into Fig 1).

---

# Before submission

- [ ] Refit dose-response excluding extreme levels (§VIII)
- [ ] π₀.₅ control-layout depth contrast on its own (needed for §X ratio)
- [ ] Wrong-side directionality: instructed vs realised side, 2×2, all cohorts — decides whether §X says "directional prior" or "commitment failure"
- [ ] DreamZero determinism: model property or harness seeding artifact (§V)
- [ ] Exact action-distinctness criterion, written normatively
- [ ] Embodiment specs for both arenas
- [ ] Verify RA-L abstract character limit on PaperPlaza
