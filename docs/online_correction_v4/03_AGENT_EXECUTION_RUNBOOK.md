# Agent execution runbook: one coordinated V4 campaign

This is an implementation and execution contract for the experiment design in this directory. It does **not** claim that a V4 runner already exists or that any V4 episode has been run. The immediate deliverable is an audited, frozen campaign that agents can run across Ali's authorized clusters without repeatedly redesigning it after observing results.

Run all released families through one dependency-aware campaign. Engineering pilots precede the main queue; passing recorded gates releases the next stage automatically. This is one coordinated campaign, not permission to bypass validation or repeatedly tune conditions until results look interesting. A missing credential or unverified resource owner blocks only the affected resource. A scientific or implementation failure must remain visible.

The experiment keeps policy weights and each episode's prompt fixed. Simulator state may drive the external intervention scheduler, physical feasibility checks, and scoring; it must never enter the learned policy through an undeclared state channel, spatial oracle, prompt update, coach, or privileged scene graph.

## 1. Start from the committed study, not from a conversational reconstruction

Read the following at the exact checkout used to implement V4:

1. Root `AGENTS.md`.
2. `docs/VLA_WAM_V3_CONTINUATION.md` and `artifacts/vla_wam_shared_v3/continuation_state.json`.
3. `docs/VLA_WAM_STEERABILITY_V3_PROTOCOL.md` and its immutable protocol artifact.
4. `docs/VLA_WAM_CONTINUATION.md`, the V2 continuation state, and `docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md`.
5. `docs/WORK_LAPTOP_B200_HANDOFF.md` and the external-repository bundle manifest/readme.
6. `deploy/k8s/v3_lane_bundle/README.md`, `QUALIFICATION.md`, and `spec.example.json`.
7. The V4 overview, experiment matrix, measurement specification, and `docs/online_correction_v4/campaign.json` supplied with this runbook. The matrix expands to **17,664 planned new policy episodes** across C1–C8, excluding pilots and privileged-controller checks. Counts and release status come from the machine-readable configuration, not a manually maintained launch list. C1/C2 use 128 independent reset blocks; secondary families use 64. This allocation is fixed: do not adaptively resize it from observed effect sizes or add seeds to chase significance. Precision for interaction effects may remain limited and must be reported.

The infrastructure audit for this document used repository commit `ce561e66f82e95055e39d3d7711691982f6b2086`. The old work-laptop handoff is a restoration guide, not the latest scientific queue. The V3 continuation files and later experiment decisions supersede its stale completion summaries. Record the implementation commit separately in the V4 freeze.

Important verified facts:

- Existing V2/V3 cohorts and protocols are historical evidence. Do not rerun completed cells, modify their registration, replace their outputs, or label V4 as recovery of a historical runtime.
- The lane bundle contains real immutable Kubernetes infrastructure. Its checked-in example is **qualification-only**: the simulator ultimately executes `/usr/bin/true`, so it produces no behavioral episode.
- `QUALIFICATION.md` reports an August 12 infrastructure run, followed by readiness/teardown fixes requiring fresh qualification. It does not establish that the current revised runtime or V4 intervention is scientifically qualified.
- V3-E006-R012 accepted no constructed grasp/carry state and ran zero behavioral episodes. A reachable end-effector pose is not evidence of a valid carried-object state. V4 must use natural acquisition and verified replay/branching instead of assuming that diagnostic solved state construction.

## 2. Version boundaries and authority

Create a new V4 namespace. Preserve V1/V2/V3 raw evidence and protocol files byte-for-byte. Add a small, explicit root `AGENTS.md` routing amendment for V4 rather than leaving agents to infer whether older restrictions permit the new intervention.

The amendment must specify:

- the active V4 protocol and continuation paths;
- that external reference motion and its state-based timing are declared experimental interventions;
- that policy input remains the released observation/action interface plus the fixed prompt;
- that pilots and main results have separate identities and disjoint reset families;
- that V3's closed queues stay closed;
- how the direct-command/interface gate applies to V4 without selecting policies for a desired language effect;
- that automatic continuation is allowed after machine-verifiable gates pass.

Do not rewrite a frozen protocol to make a failed gate pass. Before main execution, fixes create a new implementation freeze and preserve failed pilot receipts. After any main results exist, a substantive change to the policy, simulator, task, intervention, scorer, or eligibility rule creates a disclosed amendment and a distinct cohort. Derived-analysis bug fixes can reuse raw logs if their provenance and effect on results are recorded.

The campaign's two main policies are Cosmos3 Nano and the exact retained π0.5 artifact; GR00T N1.7 Bridge/WidowX is a focused second-stack bridge. Their exact identifiers, accessible checkpoint manifests, and runtimes must be resolved in the freeze. The historical π0.5 artifact's access limitations stay explicit; do not substitute a public checkpoint under the same identity.

Use a clear status lifecycle:

```text
DESIGN -> IMPLEMENTING -> QUALIFYING -> PILOTING -> FROZEN -> RUNNING -> VERIFYING -> COMPLETE
```

`BLOCKED_ACCESS`, `BLOCKED_RUNTIME`, and `BLOCKED_SETUP` are explicit side states with a reason and next action. `COMPLETE` requires the terminal artifact contract below, not merely an empty scheduler queue.

Release can cover an explicit subset of families. Keep the full 17,664-row inventory and list every withheld family with its blocked reason; do not delete unrun rows or silently lower the planned denominator. C3 requires its C1 controls; C4 requires its C1 and C3 controls. A physically invalid vertical, containment, alternate-object, or second-stack fixture does not block an otherwise qualified primary horizontal/reference-binding family. The release lock records the exact released/blocked family sets and qualifying policy/fixture identities.

## 3. Reuse existing components deliberately

The following are audited reuse candidates; no V4 behavior is implied by their existence.

| Existing repository component | Reuse in V4 | Required extension or check |
| --- | --- | --- |
| V3 Nano Phase-B runtime adapter, server, RoboLab bridge, queue launcher, and cell compiler | Released policy interface, checkpoint identity, reset discipline, raw action capture | Add V4 event scheduling, clocks, observation provenance, state-complete replay/branching, and V4 schema adapters |
| V3 π0.5 Phase-B contract/runtime/bridge/queue/compiler | Same purpose for the exact π0.5 runtime | Preserve checkpoint and controller semantics; add the same V4 contracts |
| Existing GR00T or other selected policy integration | Avoid unnecessary reimplementation of the model adapter | Verify the actual selected embodiment, checkpoint, input history, control rate, and reset behavior |
| Existing V3 compilers and evidence finalizers | Hash manifests, separation of invalid infrastructure attempts, auditable figures | Build a new V4 compiler; never stretch a lateral-only result schema by silently renaming fields |
| `tools/render_v3_k8s_lane_bundle.py` and `tools/validate_v3_k8s_lane_bundle.py` | Fresh Jobs, immutable launch specs, health readiness, file bindings, startup evidence | Generalize only where a selected GPU/runtime requires it; add corresponding validation; retain a separate V4 launch identity |
| External integration bundles in `handoff/repo_bundles/` | Exact dependencies and local integration patches | Restore their pinned prerequisites; record any necessary V4 commits separately |

The V3 bundle is explicitly pinned to A40 simulation and A100 policy roles. A B200, H100, or another cluster is a new hardware/runtime qualification, not a harmless label edit. If the renderer hardcodes GPU labels or schema version, implement and review a prospective V4 renderer rather than hand-editing generated YAML.

### Missing V4 implementation that agents must supply

Use these as module responsibilities, not fictitious commands already available in the repository:

| Responsibility | Required deliverable |
| --- | --- |
| Registry/compiler | Expand the authoritative matrix into immutable episode/group rows; validate counts, factor coverage, disjoint seeds, and shared controls |
| Scheduler | Atomic group leases, lane placement, budget accounting, retry classification, resumption, automatic gate transitions |
| Simulator adapter | Fixed-step external object trajectories; natural trigger; state/reset audit; physical feasibility assertions; no hidden policy input |
| Policy adapter | Session reset, RNG reset, observation and action-chunk metadata, independent request IDs, optional audited state snapshot/replay |
| Measurement recorder | Complete state/action/timing/trigger records; exact scorer inputs; lossless raw arrays and video index |
| Validator | Required-field, clock, pairing, physical, denominator, and hash checks; negative fixtures for common failures |
| Compiler/renderers | Freeze-driven analysis and paper-ready tables/figures/media from retained logs |
| Campaign launcher | One documented entrypoint that executes the gate graph and resumes safely; emit its actual `--help` and example invocation only after implementation |

The supplied `tools/online_correction_v4.py` supports campaign validation, manifest generation, release-lock checking, and result checking. It is a planning/validation tool, **not an inference runner**. Do not place a fabricated `run_v4.py --all` command into a handoff. The implementation agent must add the actual supported execution invocation, its working directory, environment, exact version, and dry-run output to the V4 continuation file after the entrypoint exists and passes its contract checks.

## 4. Freeze the inputs before releasing main work

Store compact artifacts under a new directory, for example `artifacts/online_correction_v4/`. The exact directory name must match the campaign configuration. Keep raw evidence on authorized persistent storage outside ordinary Git.

Required freeze artifacts:

| Artifact | Required contents |
| --- | --- |
| `protocol.json` | Research questions, estimands, populations, family IDs, exact decision rules, failure taxonomy, exclusions, shared-control mapping, stopping rule |
| `runtime_manifest.json` | Full study and dependency SHAs; image digests; checkpoint/statistics/tokenizer hashes; camera/controller/normalization/action conventions; inference execution mode |
| `setup_manifest.json` | Assets, coordinate transforms, task geometry, support/release predicates, reset bounds, reachable trajectories, settled-state tolerances |
| `prompt_manifest.json` | Exact UTF-8 bytes, goal/reference identity, carrier, relation, viewpoint convention, SHA-256 for each prompt |
| `motion_manifest.json` | Exact paths, start/end positions, amplitude/speed/acceleration/duration, smoothing, onset/stop/reversal schedules, sham definitions, reference/distractor identity |
| `scoring_manifest.json` | All numeric thresholds and units, dwell windows, response detection rule, clock definitions, censoring, uncertainty method, figure specifications |
| `seed_manifest.json` | Independent reset-family IDs; geometry, policy, scheduling and intervention seeds; excluded-pilot and main namespaces |
| `queue.jsonl` | One registered attempt-independent episode identity per row and a group identity for paired analysis |
| `queue_manifest.json` | Row counts by family/policy/setup, unique IDs, expected budget, SHA-256 of exact queue bytes |
| `gate_report.json` | Every prerequisite with passed/failed/blocked status, evidence path/hash, timestamp, qualifying runtime identity |
| `launch_matrix.json` | Authorized cluster/pool mapping, fixed runtime stratum, resource budget, immutable lane specifications |

No numeric `TODO`, placeholder path, unbound checkpoint alias, unresolved prompt, or unspecified scorer may survive in a launch-critical artifact. Unknown values are valid in the design document; they are a release failure in the generated main queue.

The main queue is frozen before the first main episode. Pilot choices are logged and the final design is labeled prospectively fixed after engineering pilots. Do not call these pilots confirmatory evidence.

## 5. Qualification and pilot gates

The goal is to catch expensive mistakes before thousands of parallel episodes run. Tests below resolve concrete measurement or execution risks; they are not additional scientific ablations.

| Gate | Check | Release consequence |
| --- | --- | --- |
| G0: source and access | Exact repository/dependency/checkpoint identities; authorized resources; persistent storage; quota and egress | Block affected lane until access or identity is resolved |
| G1: infrastructure | Actual CUDA inference, renderer frame, image/camera pipeline, health endpoint, video encode/decode, output fsync; graceful termination and durable receipts | Infrastructure only; does not authorize behavioral analysis |
| G2: state and coordinates | Reset restores registered positions, orientations, velocities, joints, camera, contacts/support; rendered left/front/up agree with logged transforms | Block affected setup if ambiguous or inconsistent |
| G3: motion and feasibility | Scripted/privileged physical execution validates both static endpoints and full motion paths; no object intersection, unmodeled collision, inaccessible target, or reference contact with robot | Freeze feasible paths; preserve failures without blaming the learned policy |
| G4: policy session | Exact-repeat behavior, reset independence, cache/RNG isolation, action units/chunks, observation hashes, static prompt bytes | A broken interface blocks; a valid policy with weak language response remains a scientific outcome |
| G5: trigger and branch/replay | Natural stable grasp detector, time of exposure, complete prefix reconstruction, equality/tolerance checks, allowed fallback mode | Release only the branch mode actually verified |
| G6: measurement | Known-motion synthetic fixtures, reference-only motion with stationary cube, off-diagonal placement, missing-log cases, censoring, clock/action latency, video alignment | Any sign, frame, or denominator bug blocks main queue |
| G7: excluded engineering pilot | Each selected policy/fixture group receives 16 static engineering episodes and 8 motion probes; all pilot viewport videos retained and inspected | Freeze parameters and release registered eligible families automatically |
| G8: full miniature campaign | Run a small disjoint rehearsal through scheduler, failed-attempt retry, interruption/resume, compiler, figures, and evidence closure | Main release only after the complete pipeline is demonstrated |

The miniature campaign is an engineering rehearsal, not an extra model-selection stage. Reuse the 240 pilot records/replays or synthetic harness cases where possible; any additional actual policy executions are separately disclosed engineering work, not an uncounted scientific cohort. It must include at least one valid behavior failure, one deliberate infrastructure failure, one no-trigger episode, and one interrupted lane as fixtures or controlled harness cases so the denominator and resume paths are exercised.

Enumerate pilot groups explicitly from the configuration: the two main policies on the horizontal, named-reference, vertical, and containment fixtures, plus Nano on the alternate object fixture and GR00T on the second-stack fixture. This is **10 policy/fixture groups and 240 policy pilot episodes**. Do not double-count a shared fixture just because several main families use it. Any additional genuinely distinct physical fixture needs its own declared pilot group. These pilot starts, motion probes, and privileged-controller checks are outside the 17,664 main-episode budget.

Direct-command checks precede wording sweeps. They verify the frozen command path and report baseline competence; they must not silently discard a selected model because its response is null. If a policy cannot naturally reach the trigger, report its trigger rate and static outcomes. Its online correction estimate may be unavailable or too imprecise; do not manufacture carried-object starts to fill the table.

Select the geometric displacement scale before policy inference using model-blind feasibility and the prespecified candidate ladder. Excluded pilots validate the implementation and detector/scorer operation; they do not select geometry or prompts for behavioral success. Any necessary scientific change is a disclosed pre-confirmatory amendment, never chosen from the direction or significance of an online effect. Once the main freeze is committed, run every registered condition and report all outcomes.

## 6. Natural acquisition and valid matched continuations

### 6.1 Default start

Start each task from its ordinary registered reset. The policy must acquire and transport the object naturally under the episode's fixed prompt. Detect a stable-grasp event using a frozen combination of lift height, contact/support, object-to-gripper stability, and dwell time. Do not use a completion-success check to decide retrospectively which episodes receive a perturbation.

The trigger occurs before final placement/release. Freeze its latest allowed occurrence and the post-trigger evaluation duration. Record early release, no grasp, no trigger, and timeout separately. Do not move the goal after the task has already completed and score the model as if it had been instructed to maintain the relation indefinitely.

The configured active-policy cap is 60 simulated seconds. The latest eligible natural trigger is 40 seconds; an event-exposed episode stops no later than `min(60 s, t_event + 20 s)`. With no event, the active cap remains 60 s. The fixed 1.0 s passive adjudication interval follows the active phase and lies outside that cap, so the full recorded trajectory is at most 61 simulated seconds. Do not grant extra active retries during passive adjudication.

### 6.2 First-placement terminal contract

The **first qualifying detachment after verified carry ends the policy phase, regardless of whether the placement is correct**. Use the same rule in every motion, sham, static-destination, wording, and runtime condition. A wrong placement, drop, or unstable first attempt is retained as a failure; the policy cannot make a rescue/regrasp attempt.

Detachment must persist for two native control ticks. Log two distinct times: `t_detach_onset`, the first qualifying tick, and `t_detach_detected`, the second tick when the criterion is known to hold. The actual intervention controller acts at **detection**, not retrospectively at onset:

1. At `t_detach_detected`, freeze the imposed reference trajectory at its actual current pose; retain the path phase and realized displacement.
2. Stop policy requests and stop applying newly predicted chunks. Hold the last safe robot controller target using the frozen non-oracle hold rule.
3. Continue ordinary passive physics for exactly **1.0 simulated second**. Do not move the object to a favorable location or stabilize it with a hidden helper.
4. Score relation, allowed support/containment, and stability from this settling interval against the reference pose frozen at detection. Retain every predicate and the settled endpoint, including wrong/unstable outcomes.

The one-tick detection delay is observable and bounded; no physics rewind, earlier freeze, or future-information detector is permitted. A one-tick contact interruption that does not satisfy the two-tick criterion does not terminate the episode. Before-detachment success guesses must not stop motion or the policy.

If the registered task deadline occurs before a qualifying detachment, stop under the frozen timeout rule and score the observed behavior; do not synthesize a release. Keep the passive-settling interval distinct from active policy time and record whether it follows release or an explicitly defined terminal timeout.

This is a **first-placement evaluation**. The resulting completion rate is not the probability of eventual success with unlimited retries. Early motion truncation is an observed exposure outcome; report it for correct and incorrect first attempts alike.

### 6.3 Preferred matched-prefix modes

Within the **same policy, prompt, reset, and sampling identity**, create matched continuations only if all relevant state can be reproduced. Choose one verified mode per registered family:

1. **Fresh restart with complete deterministic prefix replay (default).** Restart the policy session and simulator, reproduce the same observation/request history with the same RNG and timing, and require the same natural trigger state. Replaying only arm actions while omitting policy history is insufficient. Record all request/action hashes and state deviations. Each replay is computation and must be counted in the wall/GPU budget. Each registered continuation has one episode identity; shared-prefix caches must not create extra independent samples.
2. **Complete snapshot branch (only if independently qualified).** Snapshot simulator state and the policy session at the natural trigger, then restore each continuation from the same state. This includes object/robot positions and velocities, actuator/gripper state, simulator time/RNG, contact-related state supported by the simulator, observation history, policy RNG, recurrent/cache state, queued chunks, and in-flight request disposition. Demonstrate replay equivalence within frozen tolerances before using branches. An ordinary simulator pose snapshot is not enough.

Shared random seeds alone do not prove identical prefixes. Contact solvers, hidden caches, server history, and asynchronous chunk scheduling can break matching. Conversely, matching need not be bitwise if the protocol prospectively defines and verifies physically meaningful state tolerances; report that tolerance-based matching honestly.

Give every shared natural prelude a `prefix_id`. Its continuations remain correlated observations within their reset/prefix group. A no-trigger prelude may prevent all of its registered continuations from receiving an intervention; preserve that common cause and report the registered all-start statuses without counting copied no-trigger records as extra independent samples. The campaign's planned episode total is a queue/work allocation, not the number of independent statistical units.

The policy RNG seed and common-prefix identity must not depend on an intervention that starts only after the natural trigger. In particular, scenario and C4's post-grasp query schedule must not perturb the pre-trigger random stream. Derive geometry/counterbalance/phase assignments from the registered reset identity, and reuse them across its compared policies/wordings/scenarios. A different initial scene such as `destination_static` has a distinct prefix even if its sampling seed is shared.

Different prompts may cause different pre-trigger trajectories. Never claim a shared physical branch across direct/inverse wording or left/right instructions unless it was actually generated and verified. Their common reset supports a reset-block comparison; it does not make their grasp states identical.

The `destination_static` condition begins with the reference at its **planned** destination. It has a different initial scene and its own natural prelude. Pair it by registered reset family for the declared analysis; do not pretend it shares the original-scene grasp state. When first detachment truncates a motion early, this static condition is a planned-destination diagnostic, **not an identical terminal-scene control**. Record the actual endpoint/displacement of every movement and disclose truncation in that comparison. C2's named-reference prompts likewise have separate natural preludes. C4 changes query period only at the registered natural-grasp checkpoint, so its pre-trigger execution remains identical to its standard-period comparator.

Reuse control **records**, not nominally similar reruns. C3 and C4 reuse specified C1/C3 controls through explicit episode IDs and the same first-64 reset-family mapping. The registry/compiler must reject a row that silently reruns or double-counts such a control.

### 6.4 Prospective fallback if full branching is unsupported

Use independent natural rollouts from matched reset families. Assign perturbation/sham condition prospectively with a randomized order. Apply it only after the frozen natural trigger, so the external intervention has no pre-trigger effect. Preserve all starts and all no-trigger outcomes.

Call this **randomized event-triggered evaluation**, not exact counterfactual branching. Analyze within-policy/prompt event-eligible outcomes with reset clustering, report trigger counts/rates separately, and retain end-to-end outcomes over all registered starts. Do not compare post-trigger means across policies or prompts as though they describe the same selected population.

Choose the fallback before main collection and record it in the freeze. Do not mix snapshot and fallback episodes in one nominal condition without a declared stratum. Do not keep retrying a valid no-trigger episode until it yields a convenient grasp.

C2 is an explicit exception to the independent-natural-rollout fallback: its primary H selectivity requires verified common prefixes within each prompt. If deterministic fresh-session replay and complete-state restoration both fail, C2 remains blocked. The fallback is limited to C1 completion/geometric contrasts and permitted secondary outcome summaries; it cannot supply exact-prefix response or latency claims.

## 7. Keep the time axis meaningful across clusters

Every record must distinguish simulator time, monotonic host time, observation exposure, policy request/response time, action availability, and actual action application. Log the physics step and control tick explicitly. Do not use video frame numbers as the sole clock.

Required timing events include:

- external motion command and first nonzero reference displacement;
- rendered observation capture time and the state/frame it represents;
- first observation containing the scene change;
- request submission, server receipt, inference start/end, and client receipt;
- action chunk identity, horizon, consumed indices, first application, and replacements/discards;
- first action informed by a changed observation;
- detected corrective object/robot movement, task release/completion, and deadline.

Use monotonic timestamps within each host and request IDs across hosts. Raw timestamps from unsynchronized simulator and policy machines cannot be subtracted as if they shared a clock. Measure network-inclusive wall delay on the simulator client clock; keep native server timing as an internal duration unless clock offsets and uncertainty have been calibrated.

**The main campaign uses a controlled simulation clock.** The nominal emulated observation-to-action delay is **0.10 s**, quantized upward to the adapter's native control period. Standard queries occur every **0.50 s**; the C4 intervention uses **0.25 s** after the natural-grasp checkpoint. Freeze each runtime's realized quantized values and ensure their relation to the simulator/control grid is explicit. Do not change the physical controller rate as a shortcut for changing query period.

The registered event phases are the fractions `0, 0.25, 0.5, 0.75` of the **standard** query schedule, balanced by reset block. C4 and its standard comparator must have the same simulated motion onset. Do not recalculate the phase from C4's shorter period and thereby change both movement timing and action scheduling. Record requested and realized quantized phases, same-timestamp event ordering, and the common checkpoint that anchors the schedule.

Implement this causal event order for a query scheduled at simulated time `t`:

1. Finish all physics/intervention updates through `t` according to the frozen same-timestamp ordering; capture the observation and its state/frame hash.
2. Submit **that observation** to the policy. Pause simulated time while actual computation completes; measure wall-clock compute/network cost separately.
3. Retain the response without applying it. Advance physics and the registered reference trajectory through `[t, t + delay)` using the **previously available action queue** and the frozen empty-buffer fallback.
4. At `t + delay`, make the retained response available and apply the registered queue-replacement rule. The response was generated from the observation at `t`; it must not be recomputed from a future frame captured during the emulated delay.
5. Continue the ordinary physics/control/intervention schedule until the next query. Repeat with an independent request ID and the actual observation then available.

The empty-buffer rule must be fixed in the adapter and tested—for example holding the last valid controller command, with a defined initial command at reset. It must not invoke an oracle controller or query the model early. Freeze whether a newly available chunk replaces the remaining queue or appends; the standard and fast-query comparison must differ only in the declared query schedule.

This pause during wall-clock computation does **not** eliminate response delay: the full emulated interval is subsequently simulated with old actions while the reference moves. Conversely, do not simulate the emulated interval twice. Both errors materially change the experiment.

Write deterministic scheduler tests proving that (a) no response is applied before its availability time, (b) actions during the delay come from the old queue, (c) moving-object state evolves during that interval, (d) the response input contains no future observation, (e) the same registered simulated schedule is reproduced under different wall-clock inference durations, and (f) fast-query C4 shares its entire pre-trigger schedule with the standard comparator. Test action buffers crossing a query boundary, trigger/query events on the same tick, early release, and episode deadlines. A first-detachment fixture must show that motion freezes only at the second-tick confirmation, no further policy request/action is applied, passive settling lasts exactly 1.0 simulated second, and a wrong first placement cannot be rescued.

The controlled-clock curves measure behavior at the specified delay and query schedule. Actual GPU/network latency is a cost measurement only; these are **not** claims of native real-time performance or maximum hardware tracking speed. Native wall-time evaluation is outside this campaign. Do not accidentally let reference motion advance on wall time or let one model use a different time contract.

## 8. Cluster dispatch and isolation

Parallelize **independent reset groups**, not requests inside a stateful policy session. One lane owns one simulator process group and one policy-server session. A lane can execute a small batch of registered groups sequentially to amortize model loading, provided reset independence has passed. Do not multiplex unrelated episodes through the same stateful server.

For each group:

1. Acquire an exclusive durable lease on the group and record the lane/attempt owner.
2. Verify runtime and launch hashes against the freeze.
3. Execute the frozen randomized condition order serially, with complete environment and policy reset between independent starts.
4. Preserve every partial attempt and all valid outcomes.
5. Compile and validate the group from raw evidence.
6. Atomically publish the group-complete receipt and release its lease.

Where exact branching is used, serial continuations restore the same accepted prefix. If the server's hidden state cannot be reset, launch a fresh server or use the declared fallback; do not rely on an HTTP health check as proof of state reset.

Randomize condition order within blocks using a separate recorded scheduling seed. Balance family/condition assignment over qualified lanes or hardware strata so an entire condition is not confounded with one machine, time period, or cluster. Keeping a matched block on one qualified lane is preferred. GPU kernels need not have identical outputs across GPU architectures; compare repeat probes and keep materially different runtime behavior in separate strata.

For access and Kubernetes ownership, follow the repository handoff. Discover only explicitly Ali-owned contexts/namespaces/resources; do not enumerate other users' workloads. Use already supplied or normal authorized configuration instead of requesting permission again for each lane.

Use **fresh immutable Jobs** and detached commit-suffixed checkouts for real execution. The later lane-bundle contract supersedes old examples that launch experiments through reusable pods and `kubectl exec`. Read-only `kubectl logs/get` remains suitable for inspection.

Each immutable launch specification binds:

- context/namespace/PVC ownership, lane ID, attempt ID, launch hash;
- image digest and qualified simulator/policy GPU class;
- policy checkpoint and auxiliary asset hashes;
- exact executable paths and argv arrays;
- all launch-critical mounted files as path/bytes/SHA-256;
- private lane Service identity and a health probe that does not invoke inference;
- immutable output parent, separate process caches, and termination behavior.

Create objects; do not apply/patch an old Job into a new scientific attempt. The unique selector must include lane, attempt, and launch identity. The existing `/healthz` contract is HTTP 200 with `OK` after checkpoint load; use the adapter's audited metadata endpoint for other models rather than raw TCP readiness probes.

The following are real existing validation commands, to be run from the restored repository with actual paths:

```bash
python3 tools/validate_vla_wam_v3_protocol.py
python3 tools/validate_vla_wam_v2_protocol.py
python3 tools/render_v3_k8s_lane_bundle.py --help
python3 tools/validate_v3_k8s_lane_bundle.py --help
git diff --check
```

These preserve historical integrity and inspect the reusable machinery. They are not V4 scientific validators.

The delivered V4 planning/validation helper defaults to `docs/online_correction_v4/campaign.json`. Its supported commands are:

```bash
python3 tools/online_correction_v4.py validate
python3 tools/online_correction_v4.py manifest --out "$V4_MANIFEST"
python3 tools/online_correction_v4.py release-check --lock "$V4_RELEASE_LOCK" --manifest "$V4_MANIFEST"
python3 tools/online_correction_v4.py check-results --manifest "$V4_MANIFEST" --results "$V4_RESULTS"
```

Set `V4_MANIFEST`, `V4_RELEASE_LOCK`, and `V4_RESULTS` to the intended absolute output/input file paths first. These commands validate the plan, generate the assigned-cell manifest, check a completed release lock, and check result coverage. They do not implement simulator physics, qualify a policy server, or dispatch inference. `release-check` verifies local config/manifest hashes and the lock's structure, declared family coverage, and receipt identities; it does **not** independently fetch or validate remote receipt contents. The executing coordinator must retrieve and verify those evidence contents/hashes before launch. The implementation agent must add the runner and its exact invocation before release; a syntactically valid lock is not a substitute for the evidence it references.

## 9. Attempts, failures, and resumability

Separate **scientific episode identity** from **execution attempt identity**. A registered episode ID remains stable across a technical retry; each attempt receives a new ID, output directory, lane identity, logs, and immutable status receipt.

| Event | Classification | Action |
| --- | --- | --- |
| Policy produces a valid action sequence but misses, drops, places wrongly, reacts late, never grasps, or times out | Valid behavioral outcome | Keep in the registered denominator; never retry for a better result |
| Reference moves according to the frozen feasible schedule and the policy contacts it or collides because of its own actions | Valid behavioral outcome unless the protocol explicitly establishes a simulator defect | Preserve; do not exclude difficult interactions after seeing them |
| Checkpoint hash mismatch, renderer crash, corrupt transport, missing mandatory stream, wrong reset, or scheduler violates motion timing | Infrastructure/protocol-invalid | Preserve attempt; repair technical issue; retry exact registered episode under a new attempt ID |
| Cluster preempts an episode before a valid terminal record | Infrastructure-incomplete | Preserve partial trace; retry exact episode unless the frozen partial-data rule permits a limited time-series analysis |
| Mandatory logs fail but video appears successful | Not a valid full-scoring episode | Preserve; do not reconstruct primary scores from a favorable video impression |
| One optional auxiliary stream is absent while all registered primary streams are intact | Partial measurement coverage | Keep eligible primary outcomes; mark the absent auxiliary metric unavailable under its prespecified rule |
| Genuine no-trigger behavior | Valid end-to-end observation; not exposed to perturbation | Keep all-start outcomes and trigger counts; do not encode a fictitious zero-latency/zero-correction value |

Retries are for technical invalidity only. Fix the maximum retry count and lane-quarantine threshold in the execution configuration. Exhaustion creates a documented missing cell, not a model failure and not an invitation to replace its seed.

Never infer lease ownership merely from elapsed wall time: first verify the old process is dead or revoked, retain its evidence, then issue a new attempt. Storage needs compare-and-swap or proven atomic exclusive file creation/rename semantics. Test those semantics on the actual shared store. A local lock file on one node is not a cross-cluster lease.

Write logs incrementally to persistent storage and fsync at registered checkpoints. Finalization uses a write-once directory and a complete manifest, then an atomic completion receipt. The existing lane entrypoint deliberately rejects an already existing episode directory. Resume a campaign by dispatching its incomplete groups under fresh attempts, not by reusing an old episode directory.

If a paired group is interrupted, retain completed valid cells and run only its missing cells if the freeze permits the resulting runtime separation. Record the separation. If the exact shared prefix is unavailable, the group is incomplete for paired analysis; do not rerun successful or failed completed cells merely to make an attractive balanced table.

## 10. Required evidence for every episode

All pilot and main episodes require viewport video. Freeze its capture rate, codec, camera, and timestamp index before launch and provision storage from measured pilot sizes. Retain complete primary state/action/timing records for **every** main start, including failures and no-trigger outcomes. A lower-rate publication composite cannot replace source video or measurement logs. If infrastructure cannot sustain the archive, pause dispatch and fix capacity before losing evidence; do not quietly start sampling only successful videos.

Minimum retained raw evidence:

- exact episode/group/attempt IDs and queue-row hash;
- runtime, prompt, model, camera, scorer, and motion-manifest identities;
- settled reset and pre-trigger state, including actual rather than only requested reference positions;
- full time series for manipulated object, each reference/distractor, robot joints/end effector/gripper, contacts/support signals required by scoring;
- delivered observation frame IDs/hashes, crop/camera parameters, capture time, input history, and policy-input schema;
- actions before/after normalization, returned chunks, applied commands, clipping/saturation, controller step mapping;
- request IDs, RNG and session-reset receipts, timing records and dropped/replaced chunks;
- intervention trigger predicates, requested path, realized path, exposure events, and event-eligibility status;
- terminal flags, raw predicate components, detachment onset/detection, reference-freeze and passive-settling times, first-placement classification, failure classification and censoring reason;
- hashes, bytes, shape/dtype/units, compression format, and storage URI for all external arrays/video;
- model-exposed predictions only when the released interface provides them, clearly separated from actual rollout observations.

Do not save credentials, tokens, private environment dumps, or full unfiltered cluster configurations into evidence. Use the runtime allowlist already supported by the lane bundle.

Plan capacity before full dispatch. Use measured pilot bytes and seconds per episode, plus prefix-replay cost, inference request count, and headroom for bounded retries. The scheduler must enforce total authorized resource and storage budgets, not assume that many available clusters imply unbounded capacity. Keep raw outputs on persistent storage with at least one verified durable copy before removing compute resources.

## 11. Automatic health checks during the campaign

The controller can inspect technical integrity continuously without peeking to redesign the study. Release or pause work based on hash, timing, storage, schema, reset, and job-health checks. Do not change amplitudes, prompts, seeds, thresholds, or sample sizes because interim effect sizes are disappointing.

After every completed group, verify:

1. Queue identity and all mandatory artifact hashes match.
2. Prompt bytes stay constant through the episode.
3. Actual motion matches its registered schedule within tolerance.
4. No intervention reaches the policy before its permitted observation exposure.
5. Time ordering, chunk application, and control interval are consistent.
6. Reset/branch mode and eligibility record are present.
7. Every registered start has exactly one terminal scientific status or a preserved invalid attempt chain.
8. All-start, exposed, paired, and metric-specific denominators reconcile.

A systematic technical failure quarantines only the affected runtime/setup and stops further waste. Already completed valid episodes remain valid if the defect demonstrably does not affect them. Record the scope analysis; do not blanket-invalidate unfavorable results or silently retain favorable ones.

## 12. Finish with a paper-ready evidence package

The campaign is complete only when the main queue has no unexplained pending cells and the following are built from the frozen compiler:

| Deliverable | Completion criterion |
| --- | --- |
| Episode ledger | Every registered start, eligibility status, valid failure, invalid attempt, retry, and missing cell accounted for |
| Coverage report | Planned versus achieved counts by policy, setup, relation, intervention, wording, timing, and runtime stratum |
| Metric coverage | Every primary metric either computed from its required raw fields or unavailable with an explicit reason |
| Main tables and plots | Exact outputs specified by the measurement document; paired/reset-cluster uncertainty and named denominators |
| Static evidence bridge | Historical V2/V3 results reproduced from retained artifacts where supported, labeled historical; new V4 static controls labeled separately |
| Intervention validity | Realized motion, exposure latency, path feasibility, trigger-rate and contact/collision audits |
| Representative media | Preselected/random examples plus clearly labeled diagnostic failures; actual rollouts separated from predictions |
| Claim ledger | Each proposed paper claim mapped to a comparison, result file, uncertainty interval, and limitation; null and unsupported claims retained |
| `FINAL_REPORT.md` | Answers to the research questions; what worked/failed; exact remaining blockers; scope of defensible conclusions |
| Final evidence manifest | Transitive hashes for compact artifacts and durable raw storage inventories; validators pass |

The report must state whether each finding concerns an execution pipeline, a particular checkpoint, a selected event-eligible subset, or an end-to-end task. Do not promote a response-time pattern to an internal language-module explanation or pool arenas/hardware modes into a single model ranking.

## 13. GitHub handoff, ongoing commits, and cleanup

Commit code, protocol/configuration, queue manifests, compact compiled results, report markdown, plots/renderers, and bounded publication media consistent with repository policy. Keep checkpoints, raw per-frame arrays, full video collections, environments, and full private cluster records off ordinary Git. Commit hashes and durable storage references for those assets.

Use an isolated V4 branch/worktree. Do not force-push, clean unrelated worktrees, alter existing registrations, or commit other researchers' files. One coordinator integrates implementation-agent branches and publishes coherent reviewed commits; rollout workers write evidence to persistent storage rather than racing Git pushes.

Before each publication slice, run the actual V4 validator, historical integrity validators, compiler consistency checks, and `git diff --check`. Capture the current branch, parent commit, resulting commit, and pushed remote SHA. If the remote advances, integrate without overwriting unrelated work. A successful local commit is not a successful push.

Always retain Kubernetes Job/Pod JSON, image IDs, logs, and runtime receipts before deleting cluster objects. Shut down only the exact policy Job and associated simulator/Service/ConfigMap identified by the immutable launch record. Verify graceful-stop receipts where delivered; preserve any missing receipt as an operational finding. Do not delete shared storage or unrelated jobs.

Update the V4 continuation state with completed groups, remaining groups, active job/lease IDs, invalid attempts, storage inventories, blocker reasons, and the exact next supported command. A fresh agent should be able to resume using committed files and durable evidence without consulting chat history.

## 14. Coordinator acceptance checklist

- [ ] Historical artifacts remain immutable and their validators pass.
- [ ] V4 routing/authority amendment is explicit; simulator state never leaks into the policy.
- [ ] All required V4 modules exist and their real invocation is documented.
- [ ] Matrix, scorer, prompts, paths, seeds, runtime identities, and sample allocation are frozen with no launch-critical TODOs.
- [ ] The selected branch/replay/fallback mode is demonstrated, not assumed.
- [ ] Motion advances under the same declared clock contract for every compared condition.
- [ ] First-detachment detection freezes imposed motion causally, ends policy actions, and uses the same 1.0 s passive-settling scorer for correct and incorrect placements.
- [ ] Physical feasibility and reset/coordinate checks pass for every released setup.
- [ ] Pilot and miniature-campaign failures are retained and excluded from main evidence.
- [ ] Concurrent lanes have independent policy sessions, caches, RNG/reset state, services, output paths, and atomic leases.
- [ ] Technical retries cannot overwrite or replace valid behavioral failures.
- [ ] No-trigger, missing, censored, paired, and all-start denominators are independently verifiable.
- [ ] Raw logs are sufficient to recompute all registered metrics and figures without further policy inference.
- [ ] Final report and claim ledger include null/blocked outcomes and practical limits.
- [ ] GitHub contains the coherent compact evidence slice and the pushed SHA is verified.

Completing these checks is the operational meaning of “run once and write the paper.” It minimizes avoidable reruns while preserving honest outcomes when a model, setup, or hypothesis fails.
