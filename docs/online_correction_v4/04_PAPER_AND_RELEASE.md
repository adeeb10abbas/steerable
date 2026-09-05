# Paper and evidence release after V4

Status: writing and reporting contract. Every V4 empirical statement below remains **TODO: run the registered experiment and insert its estimate, interval, and denominator**. Do not write a results-oriented abstract that assumes correction, failure, or a cadence benefit in advance.

## 1. Paper structure

A suitable working title is **Spatial Instructions During Execution: Evaluating Online Correction in Robot Policies**. Revise it to match the completed evidence; avoid claiming a general solution to grounding or all forms of steerability.

| Section | Content to write | Evidence required |
| --- | --- | --- |
| Abstract | Concrete example, narrow evaluation problem, two main policies plus a focused stack bridge, matched interventions, three main outcome types | TODO: primary estimates and scope actually completed |
| Introduction | Explain “place the cube left of the bowl,” why a moved bowl can require a new action, and why static success alone is ambiguous; state at most three research questions | Historical reflection figure plus C1/C2 design |
| Related work | Spatial-relation evaluation, moving-object manipulation, action-chunk execution; state what prior work already does | Primary papers listed in the protocol, with bounded novelty wording |
| Experimental design | Policy/checkpoint identities, fixed prompts, task frames and supported goals, natural-grasp trigger, same prescribed motion/reference naming, controlled clock, first-placement scoring | Frozen allocation, runtime lock, geometry figure, exact prompt table |
| Static foundation | One compact subsection retaining the cleanest historical results and their limits | Unchanged V2/V3 sources; no pooling into V4 inference |
| Online correction | C1 movement and wording effects; C2 reference-selectivity effect and its components | TODO: all-assigned completion, goal-region error, eligible response and exposure counts |
| Continued motion and execution cadence | C3 profile responses and C4 schedule effect relative to each schedule's own sham | TODO: traces, achieved timing, uncertainty, early-release exposure |
| Focused replications | Vertical semantics, containment, object pair, second stack | TODO: C5–C8; explicitly name unavailable or blocked scope |
| Discussion and limitations | What the evidence identifies; ambiguity that remains; simulation, private checkpoint, geometry, eligibility, first-placement and timing limits | Completed results, failed and null outcomes included |
| Appendix | Historical checkpoint screen, exact prompts/geometry, scorer sensitivity, provenance, failure examples, controller validation, complete allocation and missingness | Reproducible exports, not selective screenshots |

Write academically by stating the question, defining the comparison, and reporting the measured outcome with its uncertainty. Introduce a term only when it is needed, using a concrete example first. Prefer “difference in final position,” “task success,” and “response after reference motion” to unexplained invented labels. Define goal-region distance once with an equation and show its geometry. Keep simulator internals and bookkeeping in methods/appendix unless they affect interpretation.

## 2. Fixed main tables

All table generators must read the locked ledger and contrast registry. Each table states the policy artifact, population, independent reset count, and units. A missing/blocked estimate is an explicit `not run`, `blocked`, or `not estimable` entry, not zero.

### Table 1 — Study and checkpoint scope

Rows: Nano DROID, existing private pi0.5 DROID, GR00T Bridge/WidowX. Columns: exact checkpoint ID/hash, release access, simulator/robot, input views/history, predicted horizon, achieved standard/fast period and delay, fixtures/families, allocated and completed blocks. Include a note that C8 changes the combined stack and that architecture differences are not isolated.

### Table 2 — Historical static evidence (separate from V4)

Rows: pi0.5, Nano, DreamZero for clean original/reflected layouts. Columns: original L/R success counts, reflected L/R counts, corresponding continuous geometric contrast and interval, source registry. Add the carrier-confounded inverse-wording result as a separate panel or supporting table. Use the source audit's verified quantities and signs; do not call the nonsignificant interaction evidence of equivalence. Do not claim a held-out reversal prediction from the historical sweep.

### Table 3 — C1 horizontal outcomes

Rows: policy × physical goal × direct/inverse wording. Columns: sham/move-stop/destination-static success `x/n`, capped goal-region distance with interval, trigger eligible `n/N`, delivered/observed counts, early-release motion truncation. A second panel reports moved-minus-sham and wording-by-movement contrasts. Keep per-goal rows in the appendix if space requires, but provide the prespecified equal-goal aggregate and denominators in the main text.

### Table 4 — C2 reference selectivity

Rows: each main policy and the four physical goals, plus the registered aggregate. Columns: named-A goal improvement H, named-B H, their selectivity contrast, pointwise interval, raw and Holm-adjusted primary p-value, effective matched-eligible blocks, valid first placements, physical-contact and truncated-motion counts. Show supporting world displacement. A positive interaction without a helpful named-A component is not presented as successful spatial correction.

### Table 5 — C3/C4 movement and cadence

Rows: policy × profile × schedule. Columns: achieved query period/delay, first-placement success, goal violation, helpful response rate, latency distribution/competing events, motion fraction observed, and actual inference wall time. Report schedule × movement contrast relative to schedule-specific sham. C3/C4 are prespecified secondary; do not turn the best profile into a new primary test.

### Table 6 — C5–C8 scope replications

Rows: above/below, inside/contains, sponge/tray, Bridge/WidowX, separated by policy. Columns: exact fixture distinction, sham/moved/destination-static outcomes, movement contrasts and intervals, valid/eligible counts, blocked status. Above/below tests horizontal motion of the middle reference on fixed shelves; the table must not imply vertical target motion. Inside/contains is one physical goal with inverse descriptions.

### Required supporting tables

- Full C1–C8 allocation and coverage, including every missing cell and infrastructure attempt.
- Frozen language, coordinate transforms, object dimensions, goal geometry, thresholds, and scoring sensitivity.
- Scripted feasibility results, including historical E002 failure versus the newly qualified controller.
- Behavior failure stages crossed with event-delivery/observation states, without conditioning on success.
- Contact availability, state-log completeness, optional future availability, and scorer coverage.
- Independent reset counts versus correlated condition episodes and reused controls.
- Every primary estimate with interval, null-test implementation, multiplicity handling, and not-estimable reason.

## 3. Fixed figures

1. **Task and intervention schematic.** Actual simulator renders showing the manipulated object, two candidate references, task-frame arrows, valid goal regions, natural carry event, sham versus prescribed movement, and direct/inverse descriptions. Clearly distinguish commanded reference trajectory from measured object trajectory. Do not generate synthetic experimental screenshots.
2. **Static foundation.** A small historical reflection panel with original/reflected scenes, L/R success counts, and continuous positions. It motivates the diagnostic; it is not the main new result.
3. **Primary event-aligned response.** For C1/C2 show named-reference motion, manipulated-object world motion, and distance to current valid region, with reset-level uncertainty. Mark observation capture, delayed action availability, first actual corrective response where measurable, and release. Include the named-B condition. Relative coordinates alone are insufficient.
4. **Motion-profile response and cadence.** Display the prescribed and measured waveforms, actual exposure, standard/fast trajectories, and first-placement outcomes. Separate simulation-time reaction from GPU wall time. Show failure/no-response mass alongside latency, not just the median among responders.
5. **Scope and failure composition.** A compact plot of C5–C8 movement contrasts with intervals and denominators, plus a failure-stage panel. Do not pool DROID and WidowX raw rates or hide blocked scope.

Keep trajectories representative by a declared selection rule, such as the median absolute primary contrast within each prespecified outcome class, not hand-picked dramatic videos. Include at least one successful correction, one unnecessary response, one correct no-response when the goal remains valid, one nonresponse with invalid placement, and one manipulation failure **if those classes occur**. Missing classes are stated rather than fabricated. Archive all videos so examples are auditable.

## 4. Claim-to-evidence rules

| Possible observation | Supported statement | Unsupported shortcut |
| --- | --- | --- |
| Static layout changes the success gap | Performance depends on the tested scene and criterion | A checkpoint lacks one spatial concept |
| Moved-reference performance is lower than sham, with good destination-static performance | There is a movement-associated cost beyond poor performance at that planned static destination, under this protocol | Online grounding is the only cause; destination-static is an identical-prefix control |
| C2 naming contrast plus a positive helpful named-A component | The response to the same prescribed movement depends on the named reference and improves registered goal geometry | Internal semantic representation or general understanding is proved |
| Robot moves in the same direction as the reference but goal distance does not improve | Motion responsiveness without measured geometric benefit | Correct relation following |
| Little movement, but valid placement remains | No correction was needed for this outcome | Language failure or blindness to motion |
| Direct/inverse interaction | Behavioral sensitivity to equivalent relational description under matched carrier | A viewpoint/reference-frame change was isolated |
| Fast schedule reduces the motion cost relative to its own sham | The registered execution-cadence intervention improves this outcome under controlled delay | Faster architecture, real-time deployability, or an identified neural cause |
| C8 reproduces an effect | The effect extends to the tested second stack | Embodiment alone explains the difference |
| Confidence interval includes zero | The estimate is uncertain at the achieved precision | Invariance, equivalence, or no possible effect |

The main intervention is matched as a prescribed path. First-placement termination can truncate achieved motion differently across conditions. Always report this exposure difference and restrict “identical physical stimulus” wording to the common active portion where it was actually identical. Do not select only fully exposed successful episodes to make the stronger claim.

## 5. How reviewer feedback changes this design

| Feedback from Mark / prior discussion | Concrete response |
| --- | --- |
| Too much unfamiliar jargon | Concrete placement example first; only three main outcome types; exact prompt and geometry tables |
| Left/right alone is too narrow | Four horizontal directions in the main design; supported above/below and containment as focused replications |
| Use a principled relation taxonomy | Six directional predicates; containment inverse explicitly distinguished; support/contact not conflated with above |
| Task, object, and embodiment diversity matter | Containment task, sponge/tray pair, and a single Bridge/WidowX stack bridge |
| Prefer 2–3 meaningful policies to 11 shallow rows | Two deeply evaluated existing policies plus one targeted stack replication; historical screen in appendix |
| At most 2–3 clear research questions | Goal/reference selectivity; wording/scope consistency; execution cadence |
| Simulation needs a stronger justification | Exact reset/control comparisons, measured geometry, model-blind feasibility, and deterministic timing; simulation limitations remain explicit |
| More dimensions have priority over hardware at this stage | No new hardware collection required in this campaign; no physical-robot generalization claimed |
| Avoid fine-tuning if it changes the question | Fixed checkpoints and static episode prompts throughout |
| EmbodimentSemantic and newer moving-object work overlap | Avoid a generic robustness or first-dynamic-relations claim; focus on matched policy behavior, named-reference selectivity, and controlled execution |

The earlier Overleaf comment history may contain additional line-level feedback beyond the material available in this audit. This package addresses the supplied comments and repository evidence; it does not claim that every live Overleaf thread has been rechecked or resolved during this task.

## 6. Interpretation if results are weak or mixed

The campaign is designed to yield an interpretable paper decision without demanding a desired result.

- **Strong static behavior and selective online correction:** report where relational use survives scene movement, and use failures/profile effects to define limits.
- **Strong static behavior and weak online correction:** quantify the movement cost and distinguish no updated observation/action, delayed response, unhelpful motion, and physical execution failure. Do not collapse all into language failure.
- **Sensitivity depends on wording:** report the interaction and actual components. A secondary bridge that does not reproduce it remains part of the scope boundary.
- **No cadence effect at the achieved precision:** report the interval and timing audit; do not declare architecture invariant or add more cadence conditions after inspection.
- **Few natural carries or invalid new fixtures:** all-assigned results remain informative about the tested system, but the eligible online question or intended scope may be unresolved. State that clearly. A complete ledger is not a guarantee of a strong full-paper claim.
- **Little reference selectivity despite valid execution:** report the estimate and competing explanations. Do not compensate by searching the eleven historical checkpoints for a favorable replacement.

Write only claims supported by the completed independent comparisons. If the primary question remains unresolved, produce a bounded report with the valid static and online evidence instead of initiating an unregistered follow-up campaign automatically.

## 7. Evidence and GitHub release checklist

The coordinating agent closes the campaign with one immutable results manifest linking all raw input hashes, accepted attempts, planned rows, runtime locks, scorer/analysis commits, and exported tables/figures. Preserve all failed infrastructure attempts and the exact reasons for missing or blocked planned cells. Reused controls must point to the same accepted source episode IDs in every derived comparison.

Commit compact material under an explicitly new V4 artifact namespace, for example `artifacts/online_correction_v4/`. Include:

```text
campaign.json and runtime_lock.json
planned_episodes.jsonl or its compact registry plus content hash
accepted_results.jsonl (compact records, not arrays)
coverage_by_cell.csv and audit_report.json
primary_results.csv and all registered supporting tables
results_manifest.json and raw-storage URI/hash inventory
scorer, analysis, plotting code and dependency lock
figure PDFs/PNGs/SVGs with source table hashes
continuation_state.json and final decision memo
```

Keep weights, large arrays, complete videos, decoded futures, and simulator collections on durable experiment storage. Validate that referenced URIs exist and their contents match recorded hashes; a syntactically valid hash string is not verification. Include a public/private availability map, especially for the existing pi0.5 checkpoint. Redact credentials and private cluster access details from public artifacts while retaining scientific identities and accessible evidence locations where permitted.

Run the planning helper, actual scorer tests, actual simulator timing/replay checks, full accepted-ledger audit, frozen statistics, figure regeneration, and source-to-table reconciliation. Mark which checks were performed by the design author versus the execution agents. Do not describe a documentation unit test as robot validation.

Commit and push on a dedicated V4 branch and open a reviewable pull request. Preserve V2/V3 evidence and unrelated changes. Do not force-push shared history. After review, the project's normal merge workflow can promote the new protocol; no need to merge merely to prepare a reviewable campaign.

## 8. Writing handoff after the last accepted cell

The analysis agent supplies a two-page evidence memo answering the three registered questions, each with effect sizes, uncertainty, denominators, and material limitations. The writing agent then fills the abstract/results TODOs from the generated tables, checks every numerical statement against its source row, and updates the manuscript without inventing conclusions.

There should be no need to rerun a robot episode to recover an omitted timestamp, discarded action chunk, missing reference pose, or unrecorded failure. Those fields are required in the initial storage contract. A later exploratory reanalysis may use the same complete raw evidence, but must remain labeled exploratory and must not alter the frozen primary analysis.
