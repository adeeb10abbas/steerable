# Independent scientific design review

**Review status:** prospective protocol review; no V4 experiment has been executed or empirically validated by this review. The allocation, source audit, metrics, and execution contracts were reviewed together. Implementers must satisfy the recorded qualification gates before treating this as a released campaign.

## 1. Verdict and purpose of the campaign

The design extends the existing work coherently. The historical reflection and wording results explain why binary manipulation success needs geometric interpretation. The new study asks whether behavior changes appropriately when a scene object moves during an otherwise unchanged instruction. It adds a reference-naming comparison and a controlled execution-schedule intervention rather than repeating the old checkpoint screen.

The strongest prospective contribution is **language-dependent correction under a specified scene-intervention protocol**. Motion alone is not the contribution. The named-reference experiment, simultaneous reporting of task completion and geometric behavior, and the execution-cadence comparison provide the scientific leverage.

The allocation of 17,664 episodes is substantial but bounded: two main checkpoints receive the central comparisons; a third checkpoint receives one second-stack bridge; static controls are reused only where their exact conditions and episode identities match. The protocol does not require another model leaderboard, fine-tuning, arbitrary extra seeds, or hardware experiments to close this campaign.

The review does **not** establish that the selected runtimes will grasp successfully, that the new fixtures are already implemented, that the hypotheses will hold, or that the data will support a strong publication. Those remain empirical and engineering questions.

## 2. Critical issues and required resolutions

These requirements are part of the release review. The authoritative experiment and measurement documents must agree with them before confirmatory execution.

| Issue | Why it matters | Required resolution |
| --- | --- | --- |
| Primary response time anchored to the first visibly changed observation | A policy can release before observing movement. Excluding it makes a nominally pretreatment-eligible comparison depend on a post-treatment event. | Anchor primary C2 geometric response at the **planned event onset plus 2.0 s**, with 1.0 s and 4.0 s sensitivity horizons. Keep early terminal states using the specified absorbing analysis convention. Actual observation-to-response timing remains a secondary diagnostic with explicit competing events. |
| Actual reference motion stops at each branch's first release | Naming a different reference can change release time, hence actual displacement and exposure. The complete realized visual stimulus is not necessarily identical across prompts. | Describe C2 as an interaction with the **same prescribed movement and first-placement stopping protocol**. Record realized paths, exposure, early stops, and the common physically observed pre-release window. Do not claim an identical complete sensory history or a direct internal binding mechanism. |
| C2 requires both named-reference targets to be feasible | Four relation words describe eight distinct placement cases when either A or B can be the named reference. Checking only one misses a source of apparent language selectivity. | Scripted checks cover four relations × two named references. Across six fixtures this gives 23 goal cases: 4 + 8 + 2 + 1 + 4 + 4. For nine preselected reset cases and original/midpoint/destination layouts: **621 stationary checks**, plus **23 moving checks**, for 644 checks per final geometry candidate. Rejected geometry attempts are retained separately. |
| A simulator snapshot may omit policy history and queued actions | Apparent treatment effects can originate from different pre-intervention state instead of the scene intervention. | Qualify complete prefix restoration/replay, including model cache/history, RNG, observations, controller state, queues, and clocks. An independent-rollout fallback needs a separately frozen valid estimator and cannot inherit exact-prefix response-time claims. |
| C2 eligible samples can differ across relations | Requiring every relation to reach the trigger can select an unusually capable subset; pooling all available pairs can overweight easier relations. | Estimate each relation over its explicitly reported matched-eligible reset population, then equally average the four relation estimates. Bootstrap original reset blocks with their eligibility masks intact. Do not silently drop an unestimable relation. |

The design documents and configuration incorporate these resolutions. They remain implementation acceptance requirements; this review does not certify the unimplemented runner.

For C2, failure to receive a changed observation is an exposure outcome, not permission to delete an otherwise eligible pair. A planned-event timestamp remains available from the frozen clock even when first release prevents the external motion from being delivered. The primary metric describes the full declared intervention protocol, including such stopping behavior. Its components and exposure counts must accompany the aggregate.

## 3. What each comparison identifies

| Comparison | Supported interpretation | Interpretation that would exceed the design |
| --- | --- | --- |
| C1 movement versus same-prompt sham | Effect of assigning the movement protocol in the selected policy/task/runtime | A failure located uniquely in language understanding |
| C1 planned-destination static versus moving | Whether the policy can complete the task when the planned destination is already present | An exact counterfactual for the moving episode's history or, after truncated movement, its attained endpoint |
| C1 wording × movement | Whether the movement cost differs between the two fixed equivalent descriptions | General language invariance or a pure reference-frame effect |
| C2 naming × movement protocol | Whether sham-relative improvement toward the current goal depends on which reference is named | Identical sensory histories after language-dependent stopping; proof of a particular internal representation |
| C3 prescribed movement profiles | Behavior under the tested finite durations, peak speeds, paths, and exposure | A universal maximum tracking speed or an isolated speed effect when duration and acceleration also change |
| C4 movement cost under two query cadences | Whether the declared execution schedule changes sensitivity to movement | A pure inference-speed effect, a new training method, or native real-time deployment performance |
| C5 supported vertical instructions | Above/below placement and correction under horizontal reference translation | Tracking a vertically moving reference or gravity-symmetric reflection |
| C6 containment | Completion and approach under a moved container and equivalent argument orderings | Equating a projected approach score with completed insertion |
| C7 joint object-pair substitution | Recurrence under the specified new manipulated/reference pair | An isolated appearance, texture, or manipulated-object-only effect |
| C8 second policy/robot/simulator stack | A focused combined-stack replication | A causal architecture or embodiment comparison |

A positive C2 difference is insufficient by itself: it can arise because movement harms the other-reference condition. Report useful improvement with A named, the effect with B named, their difference, completion, terminal goal violation, and realized exposure together.

## 4. Measurements that must survive adversarial checks

### A relational goal is a region

The policy need not preserve one particular offset or copy the reference's displacement. If its existing landing region remains valid after movement, little or no correction can be appropriate. The primary dataset must not exclude those cases using the observed sham endpoint. A disclosed retrospective diagnostic may identify them, but it cannot replace the registered estimate.

The same principle applies to overlapping A/B goal regions in C2. A placement that satisfies both references does not establish which reference the policy followed. Keep the `both` category and do not force a misleading reference-confusion label.

### Relative motion is not robot correction

Record manipulated-object and reference world trajectories separately. Moving only the reference changes their relative vector. A deterministic fixture with a stationary manipulated object must yield zero sham-relative manipulated-object response even though its relation and task score change.

The geometric improvement metric evaluates both manipulated-object outcomes against the same goal set from the moving branch. This removes the trivial coordinate-change explanation. Contact can still produce passive object motion; retain contact traces and classify that alternative mechanism instead of calling all movement visually mediated correction.

### Success, geometry, and timing have different denominators

All-assigned completion includes valid failures to grasp or trigger. The online response estimate uses its declared pretreatment-eligible population. Correct physical placement, observation of the intervention, and observed correction are not interchangeable outcomes.

A low geometric violation does not establish release, stable support, or completed containment. A wrong but stably released placement is not a grasp failure. Intermediate movement through a wrong-side region is not failure when the task specifies only a terminal relation.

Response-time summaries must show nonresponse and competing events. Grasp loss and release before response are not independent censoring. Mean latency among responders alone is an incomplete description.

### Empty goal sets remain behavioral evidence

Planned paths must have nonempty legal goal sets before release. If a policy subsequently creates an arrangement with no legal placement, retain the episode as a behavioral failure. Use the frozen capped-distance rule, show empty/capped counts, and preserve all raw geometry. Do not report the cap as an exactly measured physical distance, silently discard the episode, or switch scoring definitions after results are known.

### First-placement stopping changes the target population of tasks

The first confirmed detachment ends the policy phase in every condition. The protocol therefore studies correction **before a first placement attempt**, including failed drops. It does not estimate eventual completion with repeated recovery or regrasping. This restriction is deliberate and must appear in the paper's methods and limitations.

The reference is frozen only when the two-tick detachment criterion is actually detected. There is no retrospective physics rewind. The passive settling interval supplies an outcome; it must not introduce an oracle correction or count passive post-release movement as a policy response.

## 5. Engineering review before expensive dispatch

The full campaign should start only after the same small end-to-end pipeline can produce a correct negative result as reliably as a success. Qualification must include the following concrete cases:

1. Reference-only movement changes relative geometry but not manipulated-object world motion.
2. Sham/sham replay preserves the entire relevant state and produces the expected null response.
3. A pending old action executes during the emulated delay; no returned action uses a future observation.
4. Different wall-clock inference times reproduce the same declared simulated-time schedule.
5. First release before any changed observation remains in the intended completion and response denominators.
6. Early release truncates movement, and the recorded actual path differs honestly from its planned endpoint.
7. A wrong first placement cannot be rescued by continued policy requests during adjudication.
8. A correct relation with a held object fails release; partial containment fails full containment.
9. A policy-induced collision, empty goal set, or no-grasp outcome is retained as behavior, while a corrupted reset or dropped mandatory stream is infrastructure-invalid.
10. Interrupting a lane preserves completed cells and does not allow a retry to replace a valid failure.

The 240 excluded policy pilots are a technical qualification allocation, not a promise of positive policy competence. Any miniature campaign must reuse those pilot records/replays or synthetic harness fixtures where possible; additional actual policy executions must be separately identified as engineering work. A pilot showing no natural carrying can leave online estimates unavailable. Do not solve that by manufacturing a carried-object state or selecting another model after inspecting the results.

The two-reference physical validation must cover both named-reference goals and all registered diagonal direction states. The supported shelves must remain stationary. The container opening and all support footprints must be eroded by the manipulated object's dimensions. Validate swept paths, not only their endpoints.

## 6. Sample allocation and statistical limits

The independent unit is the reset block. Additional conditions, cached continuations, video frames, sampling seeds on the same physical reset, or multiple changes within one episode do not create independent scenes.

The primary allocation of 128 blocks supports useful precision for substantial effects, but a wording-by-movement interaction can be considerably noisier than one binary success rate. The 64-block secondary families should not be treated as high-precision equivalence studies. Preserve the actual paired estimators, sample counts, eligibility masks, and uncertainty limits in the final report.

The four registered primary tests need one consistent multiplicity rule. Pointwise intervals remain pointwise. C3/C4 and the bridges are prespecified secondary comparisons; a promising result there cannot silently become the campaign's sole confirmatory claim.

The analysis must remain informative under null, ceiling, floor, zero-discordance, and sparse-eligibility outcomes. Show the individual terms of interactions so that baseline floor or ceiling effects are not misdescribed as robustness. Do not resolve insufficient precision with extra outcome-driven seeds.

## 7. Conditions for moving directly to paper writing

No new experimental family is required by this review. The existing allocation is enough to support a coherent paper if its primary runtimes and measurements qualify and its outcomes answer the questions.

The evidence package is ready for writing when:

- Every declared primary cell is accounted for, including valid failures and unresolved infrastructure attempts.
- Existing V2/V3 results remain separate and retain their original scorer, cohort, and provenance.
- Completion, geometric outcomes, C2 components, exposure, and eligibility all reconcile to the same immutable episode inventory.
- The movement and clock manipulations are verified from actual traces.
- All planned profiles and bridges are reported, including blocked or uninformative ones.
- Each proposed claim states whether it concerns all assigned tasks, a pretreatment-eligible subset, a specific execution protocol, or a combined robot stack.

The final decision should be driven by the completed comparisons. Positive selectivity would support a behavioral language-conditioning result; a scheduler improvement would support a practical execution intervention; a null or low-eligibility result would impose a narrower conclusion. None of these outcomes justifies silently redesigning the frozen campaign.
