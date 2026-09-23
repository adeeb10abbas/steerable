# SGW-01 status

**The new 0.50 m native engineering layout passed all six recorded scripted
checks: both goals, three resets each. It ran on an additional verified-idle
A40 without interrupting the original four workers. Those workers have since
completed their current candidates and acknowledged the separately authorized
administrative pause. The redesigned layout is now prospectively assigned to
the existing six N3/LAT P cells, but no learned-policy run is released.**

**LAT qualification is complete: 20 all-six passes and 20 physical rejections,
including candidate 032, plus 60 geometric rejections. The frozen 29-layout
behavioral gate cannot pass. Four repaired HEIGHT/DIST baseline captures now
have verified receipts, views and complete videos. Native smoke br has completed
24 full physical trials: two candidates pass all six trials and two are valid
physical rejections. All four raw evidence chains passed independent rechecks.
Including the first fifty-two subsequent candidates, 56 HEIGHT/DIST candidates
have completed 336 full trials: seventeen all-six passes and thirty-nine physical
rejections. That retained scientific prefix is unchanged. The final worker
accounting now contains 110 completed operational slots and seven unstarted
pending slots; BU is independently compiling the remaining raw/video evidence.
HEIGHT's fixture gate is now blocked: its right-side best-case capacity is
3, below the required 14, and its total capacity is at most 19 against 29.
The qualification queue is administratively paused, not rewritten;
DIST's pool is not yet ruled out by the retained prefix. Historical layout uniqueness is not a release
requirement; the learned-runtime release gates remain unresolved.
Learned-policy requests and episodes remain at zero.**

## User requirement clarification (23 September, 14:31 UTC)

[SGW-REQ-001](historical_layout_reuse_clarification.json) records the user's
explicit clarification: historical layout uniqueness is not required.
Historical geometry may be reused; missing historical reset/producer/asset
coverage does not block SGW release. Existing historical audit entries below
are retained as provenance and must not be treated as active release gates.
Do not continue historical-novelty audits on the critical path.

Within-study layout distinctness, frozen physical outcomes, candidate caps,
counterbalance and current asset/runtime/recording qualification remain intact.
This clarification alone releases no fixture or policy run, changes no prior
failure, and does not refill the blocked LAT/HEIGHT pools. The original frozen
specification, queue and V2/V3 records remain unchanged.

The supplied study is committed on
`sz5vjy-gme-spatial-grounding-experiments`. Four Terra/Luna child sessions
implemented separate fixture, policy, persistence and analysis components.
Implementation and synthetic tests do not establish scientific readiness.

## Prospective engineering execution

**Current resource authorization:** [SGW-OPS-002](operational_authorization_20260923_expanded.json)
records the user's explicit permission to use as much verified idle existing
capacity as needed. The four-GPU/two-model-worker authorization ceilings are
superseded. The bounded engineering attempt used one additional verified-idle
GPU and is now complete; do not launch another copy.
Other qualified work may scale through finite concrete allocations. Ownership,
fresh idle checks, no preemption of unowned work, storage and stage gates remain;
no new paid capacity or privilege escalation is authorized.

The user's subsequent execution direction is now separately disclosed as
[`SGW-ENG-008-LAT-057`](infrastructure/paper-engineering-20260923cj/registration.json).
Its native path captures the preserved near-robot example without controller
actions, measures the articulation root and cameras, realizes one translated
0.50 m cube-center arrangement, then records both goals across the standard
six scripted trials. Root-offset conversion, raw/video retention and independent
evidence verification are implemented; this is not another candidate campaign.
It has now completed using exact native source `4f04e31e`.
The [native evidence](infrastructure/extra-capacity-20260923cq/native-evidence.json)
and [deployment record](infrastructure/extra-capacity-20260923cq/deployment-evidence.json)
bind the sole owner, `sgw01-ali-paper-engineering-20260923ct-1` on node171,
to the actual `paper-engineering-20260923cs` PVC output root. The realized
cube-center/root distance is **0.4999999796 m**. Robot joints and positions,
and camera positions, were unchanged; maximum measured camera angular
difference was 0.02743 degrees, within the retained 2-degree tolerance.

All six trials retained 450 commands and 451 states. The independent verifier
checked 10,824 trial files and decoded all 2,706 trial-video frames plus
726 warmup-video frames. Both native captures also retain viewport videos.
Requested terminal margins span 0.1254-0.1307 m; maximum bowl drift is
0.00002973 m. Raw frames, actions, states and every video remain on the PVC.
The nine measured arm-link bounds had no recorded obstacle AABB overlap;
these bounds do not cover the gripper's collision geometry. Filtered gripper
forces were observed on the cube and zero on the bowl, banana and table.
Neither those bounds nor filtered contacts establish complete collision safety.

The native Job exited with an infrastructure error **after all six trials**:
its verifier compared in-memory cube/bowl row order against the producer's
serialized bowl/cube order. Every numeric reset comparison matched.
The original failure and process receipts remain unchanged. A separate
[CPU-only recheck](infrastructure/extra-capacity-20260923cq/recheck/qualification-verification.json)
used the unchanged source4f verifier with the exact serialized proposal and
confirmed **6/6 passes**, without rerunning physics or changing any threshold.
The caller now reads that same proposal, and an explicit CPU-only recheck mode
preserves the original evidence tree.

The previous
[deployment receipt](infrastructure/paper-engineering-20260923cn/deployment-receipt.json)
recorded **a client-independent natural-release handoff**. Under that superseded
allocation policy, native Job `cn` was created
suspended and pre-bound to rank1's existing node. CPU Job `cp` is now retired
and suspended, its logs preserved and its Pod confirmed absent. Do not restart
`cn` or `cp`. Ten device-admission failures and five rejected non-idle leases
remain administrative evidence, not scientific failures. Only CT-1 passed
both idle checks and acquired the persistent cross-wave claim. That claim
must never be removed to rerun this attempt. No preemption, paid provisioning
or unrelated workload modification occurred. The extra GPU was released when
the native container exited; the four original GPU workers and `bu` collector
remain active under the expanded authorization.

The rejected `ck`/`cl` scheduling routes and `cm`/`co` CPU dry-run attempts
remain infrastructure provenance; none executed the new scene or a model.
The next redesign milestone is a separately disclosed prospective campaign
based on these native results, not another historical audit or a duplicate
engineering attempt. No frozen pool is paused, refilled or relabeled, and no
campaign or learned run is released by this engineering demonstration.

## Priority reconciliation (23 September, 18:41 UTC)

[SGW-OPS-003](priority_pause_authorization_20260923.json) retains the user's
permission to pause our older queue **after each current candidate completes**
for a future concrete priority decision. The immediate rationale was corrected:
SGW-ENG-008 already completed. **No stop sentinel was written and no cluster
resource was modified.** The read-only snapshot found all four BT workers and
BU running, with 102 partition completion receipts and DIST070/079/072/083
in progress. These are operational counts, not a newly reverified scientific
prefix. Completed evidence stays intact; unfinished slots are never failures.

The next concrete redesign deliverable is a **separate post-result LAT/HEIGHT
qualification registration**: a hash-bound seeded candidate generator,
family-appropriate landing/support geometry derived from actual native
measurements, finite cap, deterministic selection and counterbalance.
Robot/cameras/controller, physical thresholds, prompts and model counts remain
unchanged. Native candidate checks must still establish 29 distinct qualifying
layouts per family; the successful one-layout demonstration does not refill
the old pools. BT/BU continue the original queue and its independent terminal
accounting, including DIST. The user's subsequent priority is to execute the
already registered **six nonbehavioral fixed-input requests per model**,
starting with N3 when its actual source-bound image/proprio/reset inputs and
idle allocation are ready. These learned requests are not behavioral episodes
and do not wait for a 29-layout pool or historical novelty. Full-pool behavioral
release remains separate. The user subsequently cancelled the separate
engineering pilot and explicitly confirmed the MAIN P -> D -> C progression.
Priority is now the six N3 requests registered in
`n3_fixed_input_20260923cw.json`, then the existing MAIN N3/LAT P-stage six
episodes under a distinct prospectively disclosed redesigned-scene assignment
and scoped stage-ordering amendment. This neither repairs the old blocked LAT
pool nor releases full confirmation. No learned request has completed at this
registration freeze.

The deployed stop hook can drain at candidate boundaries, but the old launcher
and collector label incomplete partitions as generic infrastructure errors.
If a future priority decision uses it, retain their original records and add
an explicit administrative-pause receipt and remaining-slot ledger; do not
invent infrastructure faults or relabel physical outcomes.

### 19:19 UTC: native N3 startup attempted; exact access blocker

The fixed-input registration and runner are committed. CW B200 admission
failed before container startup (zero available devices). CZ reclaimed the
previously released native A40: its actual idle receipt recorded zero used
memory, zero utilization and no compute occupants. It verified the source and
checkpoint, retained the actual camera/proprio inputs, then failed in the
native constructor before **any policy request** because OpenCV could not
resolve `libxcb.so.1`. Existing native libraries resolve that CPU import.

The next constructor prerequisite is now explicit: its unchanged auxiliary
safety assets require
`nvidia/Cosmos-Guardrail1@d6d4bfa899a71454a700907664f3e88f503950cf`,
including `face_blur_filter/Resnet50_Final.pth`. The official pinned download
returned **“Access denied. This repository requires approval.”** Authorized
Hugging Face access/authentication must be resolved; safety settings and access
controls were not bypassed. No policy responses or generated futures exist:
**0/6 N3 requests started, zero behavioral episodes**. Original receipts and
the owner claim are retained in the
[execution capsule](infrastructure/n3-fixed-input-20260923cw/execution-summary.json).

For the confirmed main-experiment priority, the administrative boundary pause
was activated at 19:00:41 UTC with 106 completed partition receipts and four
current candidates. Ranks 1 and 2 subsequently acknowledged
`stopped_before_next_slot` after completing DIST088 and DIST080; their original
launcher error labels remain untouched and are classified by the separate
[administrative receipt](infrastructure/administrative-pause-20260923cx/activation.json).
Ranks 0 and 3 were still completing their candidates at the latest read.
Unstarted slots stay pending; no candidate was killed and no new scientific
failure is inferred from the pause. BU's independent final collection remains
required.

### 20:49 UTC: boundary pause complete; MAIN P physical assignment frozen

All four workers acknowledged `stopped_before_next_slot` after completing the
four candidates active at the stop request. The last Pod terminated at
19:25:58 UTC. [Boundary accounting](infrastructure/administrative-pause-20260923cx/boundaries/summary.json)
reconciles all 117 assigned slots: **110 operationally complete, seven
unstarted pending, zero unfinished claims**. Pending DIST slots are
086/092/098 on rank 0, 094 on rank 1, 090/095 on rank 2, and 096 on rank 3.
Their unfinished status is not a scientific failure. Original generic launcher
errors remain preserved alongside the administrative cause. No GPU worker
remains active; any new allocation still requires a fresh ownership/idle check.
BU remains running and has not yet produced its final independent summary.

The [prospective MAIN P assignment](main_n3_lat_p_assignment_20260923.json)
binds the actual hash-verified redesigned capture and six-pass qualification
receipt to the **six existing N3/LAT P cells** in their original
I+/D+/C+/C-/D-/I- order. IDs, prompts, seeds and episode allocation are unchanged.
Selection after known scripted outcomes is disclosed; the layout is reserved
against later D/C duplicate-within-tolerance reuse. The rejected original
LAT057 and old blocked pools remain untouched.

This is a physical assignment, not a runtime release. N3 still has **zero
learned requests** and requires actual fixed-input, runtime and time-map
evidence. The user asked why the auxiliary content filter is needed; no runtime
configuration change was authorized or made. The current native startup
access blocker remains, and no new GPU or simulator was launched here.

### Authentication diagnosis after the user's 21:26 UTC status request

Direct `hf auth whoami` checks on both the laptop and the cluster's exact
`HF_HOME` returned **Not logged in**. The Hub reports
`nvidia/Cosmos-Guardrail1` as **automatically gated**, not manually approved.
The earlier unauthenticated download denial therefore did not establish that
the user's account lacked permission. Normal browser sign-in has now succeeded,
but the exact authenticated file dry-run still reports access required. The
official model page is open for the user to accept its repository access terms.
The [authentication record](infrastructure/n3-fixed-input-20260923cw/authentication-check-20260923.json)
preserves these observations without credentials.

The main Nano checkpoint is already downloaded and hash-verified. Only its
native auxiliary startup dependencies remain to be staged after legitimate
authentication. No runtime configuration change, new policy request or GPU
allocation occurred during this diagnosis.

## Latest retained evidence (23 September, 13:34 UTC)

The thirteenth prefix preserves HEIGHT084/086/088/089 with 6/6, 6/6, 5/6 and
4/6 physical passes. All 24 original raw trial/state/score/video projections
were checked. Including smoke, the retained total is 56 candidates, 336
complete trials and 151,200 scripted actions. The replayed capacity bound is
HEIGHT left16/right3; its fifteen left-side passes cannot repair the right-side
shortage. At 13:34 UTC, workers had 13/14/14/14 completions with active
HEIGHT090 and DIST006/007/008. HEIGHT091/097 and DIST002 were newly terminal
but are not classified in this retained prefix.
There was no shared stop, terminal launcher receipt, collector error or final
collector receipt. The original queue, controller and four-worker allocation
remain unchanged; the former aggregate resource ceiling is superseded above.

The [native metric capsule](infrastructure/native-metric-20260923cg/manifest.json)
now binds DIST000's actual actor roots and the current installed native
root/precision/metre API to the registered Isaac Sim5/IsaacLab2.2 contract.
The historically hash-bound V3-B002 producer supports an exact-arithmetic,
outward-rounded per-root error bound of 0.500 mm (rounded upward here).
All 108 named final-cell post-settle comparisons violate the necessary
distance-match condition, with a minimum remaining gap of 84.24 mm.
These are **conditional on the recorded native API contract**, not historical
native import-byte attestations, 108 independent layouts, global population
coverage or a fixture release. Constructor, settle-window, preflight and
infrastructure states remain outside this selection. Both proof and capacity
audit replay byte-identically on the PVC; all four native exports reproduce.

The [stochastic reset recovery](infrastructure/historical-pi05-stochastic-20260923ch/manifest.json)
retains all 864 named reset attestations from 432 final V3-D001 cells and
reconciles their original raw/capture streams. These 54 registered conditions
contain one numerical root pair, not 864 independent layouts. The inherited
runtime binds Phase-A sources; its nested stochastic attestation binds the
policy server, not the D001 simulator bridge. A current checkout or launch
directory cannot substitute for a historical producer hash. The audit and
all five raw exports replay byte-identically, but add **zero qualified
cross-frame nonmatches** and do not close global historical coverage.

The separate [Phase-A recovery](infrastructure/historical-pi05-phase-a-20260923ci/manifest.json)
does establish its **actual runtime patch's historical hash anchor**, through
the runtime identity, all original pair manifests and retained launch commands.
It preserves 54 behavioral initial states and 54 separately recorded setup
resets; all original raw/capture/partial streams reconcile. Their 108 necessary
DIST000 comparisons are nonmatches under the recorded native API, with an
83.62 mm minimum remaining separation difference after numerical error.
These records contain one numerical root pair, not 108 independent layouts.
The proof and 30 exports reproduce on the PVC. This bounded result does not
attest historical imported native bytes, cover all historical populations or
supply D001's different producer identity.

The user's offer of manual scene setup is treated as targeted engineering
help, not a human approval gate or authorization to rewrite frozen candidates.
An existing HEIGHT063 lower-goal reset0 recording and exact scene/controller
context were copied for inspection. Its bowl drifts 44.23 mm against the 5 mm
limit, first crossing the limit at action84 during transport (actions81-100).
The cube still finishes on the requested lower side with detached release.
Inspecting swept-path clearance around the bowl is more targeted than rebuilding
the whole scene; exact contacting-link attribution remains unmeasured.
Any future repair requires separate prospective disclosure and preserves every
existing outcome. No trial or policy request was added for this inspection.
The user explicitly selected **full benchmark qualification first**, not an
exploratory learned-policy pilot. Side chat separately supplied four existing
LAT/HEIGHT failure/control clips; that handoff is not duplicated or counted
as new evidence here.
The user also explicitly requires **RoboLab only**, with no LIBERO or other
benchmark switch. Side chat reports a completed LAT/HEIGHT/DIST manual-design
sheet and portable ZIP. Its proposed goal views are design aids, not new
qualification gates or successful trials. Central reach around 0.5 m is a
hypothesis, not a universal optimum or a causal explanation of these failures;
current documentation does not authorize upgrading pinned RoboLab or changing
the frozen 100-slot pools. This relay has not been independently reverified here.

The user subsequently approved the [paper-informed design direction](infrastructure/paper-informed-design-20260923ci/manifest.json).
Its approximate 0.50 m target is **cube geometric-center translation from a
measured robot root**, not authoring X or actor-root radius. Conversion must
use the measured root-local center offset and retained orientation, with
robot/cameras fixed. The supplied correction is retained: RoboLab Fig. 9 is a
success-conditioned posterior for three pi-family policies on two banana tasks,
and its pose metric includes rotation. It is not a universal pure-distance
success optimum. The supplied Fig. 5 discussion of arXiv:2602.17659 concerns
fine-tuned pi0.5 on real FR3/Robotiq with randomized positions, not matched-reset
N3/D1 evidence. Cube contact alone cannot establish SGW goal grounding.
At that proposal stage no scene had been materialized or released. The later
separately registered SGW-ENG-008 realization and six-trial result are recorded
above; full benchmark qualification remains a separate requirement.

## Current prospective freeze (23 September, 07:07 UTC)

CPU-only stage bm restored tested source `cdb7a55b` and froze exactly 100
proposals per family with unchanged seed `20260922`, exact bi captures, and
their sealed native-only reviews. HEIGHT has 57 capture-eligible proposals
(28 left, 29 right) and 43 geometric rejections; DIST has 64 (45 left, 19
right) and 36 geometric rejections. All rejected slots remain consumed.
The exact plans, campaigns and receipt are retained in
[`infrastructure/family-plan-freeze-20260923bm/`](infrastructure/family-plan-freeze-20260923bm/).
Geometric eligibility is not physical qualification.

The native canonical-root and early-rejection verification corrections are
integrated. The full source-backed suite passes 326 tests, with one
Linux-specific identity skip. Registered Job bn selects the first eligible
slot per family/side before physical outcomes: HEIGHT 000/001 and DIST
000/003. It is bounded to four A40 GPUs, zero model workers, no automatic
retries and six fixed trials per candidate unless a measured, independently
verified geometry rejection terminates the candidate. It retains all raw
frames, states, actions and trial/warmup videos on the PVC. The other 117
capture-eligible slots remain unlaunched until actual smoke integrity passes.
Historical coverage, learned runtime, pilot and behavioral release gates are
unchanged.

Job bn started at 07:12:51 UTC on four existing A40 nodes. All four
allocation probes measured zero MiB used before native simulator startup;
all four candidate captures and materializations completed. At 07:15:21 UTC,
all four workers were terminal infrastructure failures and released their GPUs.
Qualification could not read its required support-contact sensors before its
first controller action. All candidate capture and reset-warmup videos remain
on the PVC; compact evidence is in
[`infrastructure/native-smoke-20260923bn/`](infrastructure/native-smoke-20260923bn/).
No physical or model failure is inferred.

The cause is independently reproduced using the actual pinned native contact
factory: materialization alphabetically sorted the object inventory, reversing
support/cube sensor names relative to capture's original cube/support order.
The correction preserves the manifest's exact native import order and surfaces
native infrastructure receipts without obscuring them as candidate-hash errors.
All 35 targeted campaign/executor/proposal/native-boundary checks pass.
This is a disclosed engineering correction after bn, not a new scientific
freeze: the exact bm designs, seed, thresholds and calibration remain fixed.
Only the same four infrastructure-invalid slots may receive fresh technical
attempts; no valid physical trial is rerun and no rejected proposal is replaced.

CPU stage bo independently rematerialized all four real bn captures with the
corrected source `c3c976db`, preserving their recorded contact IDs. Its generated
repair manifest was byte/hash checked against the original template, changing
only source and attempt identities. Job bp started at 07:27:40 UTC at the same
four frozen slots, in new PVC roots, after four fresh zero-memory-use GPU
checks. The original bm CPU Job completed at 07:27:43 UTC. All bp captures
again passed, and contact lookup advanced beyond bn's error. At 07:30:06 UTC,
all four bp workers had failed before controller planning at the next native
boundary: `WorldState.get_bbox(env_id=0)` returns a list of `Gf.Vec3d`, not a
Tensor. Their immutable evidence is in
[`infrastructure/native-smoke-20260923bp/`](infrastructure/native-smoke-20260923bp/).
The snapshot now reuses capture's established per-corner vector conversion.
Seven native-boundary tests pass, including real `Gf.Vec3d` list and Tensor
corner representations through the actual snapshot method. This additive
engineering repair changes no geometry, threshold, seed, or valid outcome.
Job br started at 07:36:45 UTC on the same four fixed slots with tested source
`e43f5b40`. It has advanced into real controller execution: each slot recorded
its first complete 450-action/451-state trial and subsequent resets are
running. No all-six-trial physical outcome is yet claimed. The full
source-backed suite now passes 329 tests with one Linux-only identity skip.
The finite remaining-slot worker is being implemented independently, but its
launch remains conditional on completed native smoke integrity. All learned
gates stay closed.

Job br completed and released all four GPUs at 08:02:52 UTC. HEIGHT 000 and
DIST 000 each passed all six trials. HEIGHT 001 and DIST 003 each passed five
of six and remain valid physical rejections: HEIGHT's last negative-goal reset
disturbed the reference, while DIST's first positive-goal reset failed the
terminal physical predicate. Neither may be rerun or replaced. All 24 trials
contain 450 actions and their complete state/frame/video records. Compact
evidence is retained in
[`infrastructure/native-smoke-20260923br/`](infrastructure/native-smoke-20260923br/);
raw recordings remain on persistent cluster storage.

CPU-only stage bs restored exact tested worker source `68b9e100` and
independently reverified every raw smoke trial, reproducing all four existing
verification files byte-for-byte. Its sealed receipt and the registered
partition configuration are in
[`infrastructure/family-source-stage-20260923bs/`](infrastructure/family-source-stage-20260923bs/).
The partition/executor checks pass 23 tests. Registered Job bt assigns the
117 remaining slots to four disjoint ranks of 30/29/29/29. It has no automatic
retries, a shared stop sentinel for startup or slot infrastructure faults,
exclusive shared claims, fresh GPU checks before each slot, and a conservative
3,844 GiB free-space requirement before every claim. Already-running peers
finish their current slot after a stop; they cannot claim another one.

The observed complete smoke durations were below 27 minutes per candidate.
The new engineering bounds are 40 minutes per native child and 48 hours for
the finite Job, without changing six-trial/450-action scientific limits.
This model-blind qualification queue can run without the app or laptop; it
does not release any learned-policy queue or use a behavioral P95 estimate.
Historical comparison does not require a prior layout to have passed or been
released, but still requires complete frame/asset/layout lineage and population
coverage. The recovered preflight roots do not close that gate.

Job bt started at 08:16:48 UTC. All four allocated A40 devices measured zero
MiB used before startup, and each worker independently repeated all four smoke
verifications. The first four fixed slots, HEIGHT 003/004/005/006, have
completed native capture and are recording controller actions, states and
viewport video. Shared bindings and launch evidence are retained in
[`infrastructure/family-partition-20260923bt-startup/`](infrastructure/family-partition-20260923bt-startup/).
No remaining-slot terminal result is claimed from this progress snapshot.

CPU-only collector bu started at 08:23:10 UTC with separately staged source
`a995e014`. It independently checks an initial snapshot, then waits for the
four terminal launcher receipts and compiles the complete collection from raw
evidence. A missing or interrupted result remains explicitly incomplete.
Its maximum lifetime is 54 hours; neither it nor the GPU queue needs this app
to remain open. The compiler/worker/executor checks pass 34 tests. The collector
cannot release fixtures, launch a model, assign behavioral partitions, or change
the live worker source.

At 08:29:28 UTC, bu completed its real initial compilation and independently
accounted for all 200 proposals: 79 geometric rejections, two all-six physical
passes, two valid physical rejections, four in-progress slots and 113 unstarted
slots. This is explicitly an incomplete snapshot, not a completed campaign.
Its hash-verified output is retained in
[`infrastructure/family-collector-20260923bu/initial/summary.json`](infrastructure/family-collector-20260923bu/initial/summary.json).
The collector is now waiting for terminal worker receipts before the separate
final compilation.

By 08:53 UTC, each bt worker had completed its first remaining slot and
automatically claimed its next one. HEIGHT 006 passed all six trials;
HEIGHT 003 passed four, and HEIGHT 004/005 each passed five. These three
rejections remain consumed, with complete 450-action trials and videos.
The retained worker-verified prefix, including smoke, is eight candidates:
three all-six passes and five physical rejections across 48 complete trials.
The remaining 113 slots are not yet terminal. Compact verdicts, exact
qualification hashes and video bindings are in
[`infrastructure/family-partition-20260923bt-prefix-bv/`](infrastructure/family-partition-20260923bt-prefix-bv/).
This prefix is not bu's final independent raw compilation.

CPU-only bv completed at 08:44:42 UTC and released its resources. It rehashed
the same seven historical repair sources (1,338,347,469 bytes), retaining
663,728 bytes of selected fields and newly explicit source/frame bindings.
Every previous bl field is unchanged. The reproducible source audit matches
all 93 in-repository bindings against their exact recorded Git revisions.
All 100 named frame identities pass and record zero environment origins.
The producers explicitly read object actor roots from `root_pos_w`; their
separately named EEF fields do not redefine object-root coordinates.
The three distinct directly bound external files also match their retained
historical hashes. Evidence is in
[`infrastructure/historical-lineage-20260923bv/`](infrastructure/historical-lineage-20260923bv/).
These facts resolve selected provenance gaps, not missing asset dependencies,
root-local geometry or complete historical populations. Coverage stays closed.
The extraction/audit checks pass 19 tests. The finite bs accessor also completed;
bu remains the CPU-only PVC access path while waiting for the GPU workers.

At 09:18 UTC, bt had completed the next four slots as well. HEIGHT 009/010
passed all six trials; HEIGHT 007 passed two and HEIGHT 008 passed five,
both valid rejections. Including smoke and both retained prefixes, twelve
candidates have completed 72 full recorded trials: five all-six passes and
seven physical rejections. The remaining 109 slots are not yet terminal;
HEIGHT 012/013/014/015 are recording. The second prefix is retained in
[`infrastructure/family-partition-20260923bt-prefix-bw/`](infrastructure/family-partition-20260923bt-prefix-bw/).

The separate native-frame audit binds the exact RoboLab getter to commit
`0aef241f`: it returns actor world roots minus the environment origin, without
a robot-base rotation. Four retained HEIGHT captures and the 100 named
historical snapshots record zero origins in those same axes. All 400
cross-comparisons have a required actor root separated by more than the frozen
3 mm tolerance, proving non-duplication of those particular snapshots even
without their missing centroid offsets. Root agreement, missing required roots,
or uncertain frames remain unresolved. The primitive is not wired into a
release gate; complete historical populations remain unproved. The evidence
and reproducible proof are in
[`infrastructure/native-root-frames-20260923bw/`](infrastructure/native-root-frames-20260923bw/).
All 33 targeted root/lineage checks pass; no model or fixture branch is released.

At 09:34 UTC, HEIGHT 012 passed all six trials; HEIGHT 013/014 each passed
five and HEIGHT 015 passed four. The three valid rejections remain consumed.
The third worker-verified prefix is in
[`infrastructure/family-partition-20260923bt-prefix-bx/`](infrastructure/family-partition-20260923bt-prefix-bx/).
Including smoke, 16 candidates have completed 96 trials and 43,200 scripted
actions, with all videos retained. This remains worker-verified evidence,
not the pending independent final collector result. At 09:40 UTC, bt was
recording HEIGHT 016/020/021/024, its shared stop was absent, and bu had no
terminal output or error record. Geometrically rejected indices are not refilled.

The source trace found that bl/bv omitted the distinct
`materialization_environment/fresh_reset` path. CPU-only bx completed at
09:36:30 UTC, rehashing the same seven full sources and retaining 48 additional
materialization resets, their frame records, lifecycle bindings and scalar
outcomes. Its expanded selections preserve every old selected pointer/value.
A separately hash-bound 47,076-byte R012 geometry-preflight receipt supplies
one further reset omitted by both stream contracts. The reproducible audit
now accounts for 101 recorded environment lifecycles, 101 fresh-reset
snapshots and 48 candidate-state snapshots across these seven named
executions. All 149 frame identities pass and have zero environment origins.
Evidence and the audit are in
[`infrastructure/historical-population-20260923bx/`](infrastructure/historical-population-20260923bx/).
The 400 earlier nonmatch comparisons remain scoped to their original 100
snapshots; they are not silently extended to this larger set.

This is still **not complete historical coverage**. The committed R005
infrastructure ledger records an earlier `98f0234-a40r06-attempt01`: all
20 environments and four candidate pairs were computed before terminal
serialization failed, but ranks 1--3 lack retained materialization scientific
state payloads. It is outside the seven streamed executions. Neither its zero
accepted states nor its zero model requests permits excluding those layouts.
The exact ledger, source and raw failure-report bindings are retained in the
new audit. All 43 targeted extraction/lineage/root tests pass; learned runtime,
historical coverage and fixture release remain unresolved.

The follow-up read-only by inventory rehashed R005 attempt01's 25,614,537-byte
failure report and all eight bound native HDF5 files (2,943,214 bytes). They
contain finite float32 actor-root channels and two initial samples, but are
**IK-only**: no materialization HDF5 is present. Six source-defined snapshots
survive for rank 4, with passed zero-origin frame records. Neither those
snapshots nor IK telemetry recover the missing rank 1--3 materialization
states. The HDF5 coordinate frame remains unqualified. The compact inventory,
its exact inspector and source bindings are retained in
[`infrastructure/historical-r005-attempt01-20260923by/`](infrastructure/historical-r005-attempt01-20260923by/)
and on the PVC. At 09:49 UTC, file-based replay reproduced the inventory
byte-for-byte after correcting a recorded inspector-filename import collision.
No new pod, simulator, model request or behavioral episode was created.

At 10:01 UTC, the fourth remaining-slot wave was retained: HEIGHT 020/024
passed all six trials, while HEIGHT 016/021 each passed five and remain valid
physical rejections. Including smoke, HEIGHT/DIST now has 20 completed
candidates, eight all-six passes and twelve physical rejections across 120
full recorded trials / 54,000 scripted actions. All 24 new score projections
agree with the canonical worker-verification receipts. The fourth prefix is in
[`infrastructure/family-partition-20260923bt-prefix-bz/`](infrastructure/family-partition-20260923bt-prefix-bz/).
At 10:03 UTC, HEIGHT 026/027/032/033 were active, no shared infrastructure
stop existed, and no worker had emitted a terminal launcher receipt. The
independent final collector result remains pending.

The bounded R005 recorder audit resolves a misleading directory name:
the historical invocation used `RoboLab-11142d4` but explicitly pinned commit
`0aef241f`. That commit's initial-state and post-step recorders both request
`scene.get_state(is_relative=True)`. The currently installed IsaacLab 2.2.0
getter matches its distribution RECORD and subtracts environment origins
without rotating actor roots into the robot frame. Exact sources and
byte-identically reproduced audit output are in
[`infrastructure/historical-r005-recorder-20260923bz/`](infrastructure/historical-r005-recorder-20260923bz/).
This establishes the source contract, not an independent attestation of
historically imported getter bytes. It does not recover missing materialization
states or release a fixture/model. The pinned camera writer stores PNGs;
its numerical geometry is returned in memory, not a separately saved pose file.

At 10:22 UTC, the fifth remaining-slot wave was retained: HEIGHT 032 passed
all six trials, while HEIGHT 026/027/033 passed five/four/five and remain valid
physical rejections. Including smoke, the 24 completed candidates comprise
nine all-six passes and fifteen rejections, with 144 full recorded trials and
64,800 scripted actions. All 24 new score/video projections and canonical
worker digests agree. The fifth prefix is in
[`infrastructure/family-partition-20260923bt-prefix-ca/`](infrastructure/family-partition-20260923bt-prefix-ca/).
At 10:21 UTC, HEIGHT 034/035/036/037 were active, the shared stop was absent,
and no terminal launcher receipt existed. The CPU collector had no terminal
or error record at its 10:17 UTC check. These prefixes are not its pending
independent final raw compilation.

The source-bound ca proof recovers an implication rather than missing
coordinates. R005 attempt01's hash-bound launcher enforces the exact clean
study checkout; its producer sets a completed-reset marker only after the
frozen full-reset comparison passes. That comparison bounds every object
root within **5 mm Euclidean distance** of its reference. Outward-rounded
intervals for its 20 completed settled resets exclude each of the four
already frame-qualified HEIGHT 003--006 captures, yielding 80 scoped
nonmatches under SGW's unchanged **3 mm componentwise** tolerance.
The actual minimum root separations exceed 10 cm. Inputs, source checks and
the reproducible proof are in
[`infrastructure/historical-reset-bounds-20260923ca/`](infrastructure/historical-reset-bounds-20260923ca/)
and on the PVC; the finite receipt export reproduced byte-for-byte there.
All 70 targeted historical extraction, lineage, point/bound and source-proof
checks pass.

These intervals are not 20 recovered point observations, overlap the existing
rank4 reset evidence, and do not cover later constructed states or recover
rank1--3 materialization payloads. Historical coverage remains incomplete.
The newly retained controller verification also records the historical
pre-AppLauncher hash check for the entire `basic_recorders.py`, including
initial/post-step recorders. Historical imported IsaacLab getter bytes remain
separately unattested. No fixture or learned-policy gate changes.

At 10:42 UTC, the sixth remaining-slot wave was retained: HEIGHT 034/036
passed all six trials; HEIGHT 035/037 passed four/three and remain valid
physical rejections. Including smoke, 28 candidates now have 168 full recorded
trials and 75,600 scripted actions: eleven all-six passes and seventeen
rejections. All 24 new raw score/video projections and canonical worker digests
agree. The compact evidence is in
[`infrastructure/family-partition-20260923bt-prefix-cb/`](infrastructure/family-partition-20260923bt-prefix-cb/)
and on the PVC. At the 10:42 worker check, HEIGHT 039/040/041/044 were active,
with no shared stop, terminal launcher receipt, collector error or final receipt.
The pending independent final raw compilation is still required.

The cb historical recovery rehashed **all twenty** native `env_cfg.json`
files bound by R005 attempt01's failure report, one per recorded lifecycle.
Their 1,031,942 source bytes are preserved losslessly using one complete
51,615-byte configuration plus byte deltas; only recorder output directories
differ. The complete configurations, lifecycle matches, pinned serializer and
importer sources, runtime bindings and reproducible audit are in
[`infrastructure/historical-r005-inventory-20260923cb/`](infrastructure/historical-r005-inventory-20260923cb/)
and on the PVC. The persisted exporter reproduced both compact inputs
byte-identically.

Every configuration registers the cube, bowl, two bananas and table, with
the bowl as the success reference. Serialization occurs after native
environment construction and before the recorded construction-only horizon
extension; before/after termination contracts agree in all twenty lifecycles.
This recovers complete configurations, **not** missing measured materialization
poses. Registered-object absence alone is not a physical no-plate proof:
the native importer filters its registered objects while spawning the complete
USD scene, and complete historical transitive asset identity remains unproven.
DIST historical exclusion therefore remains unresolved. The exact historical
materialization finalizer also returns normally when its gates fail; its
completion cannot justify extending the fresh-reset bounds to later states.
No measured pose, global historical coverage or learned release is added.

At 11:03 UTC, the seventh wave was retained: HEIGHT 040 passed all six trials;
HEIGHT 039/041/044 each passed five and remain valid physical rejections.
Including smoke, 32 candidates now have **192 full recorded trials** and
86,400 scripted actions: twelve all-six passes and twenty rejections. All 24
new raw trial/video projections and canonical worker digests agree.
[`infrastructure/family-partition-20260923bt-prefix-cc/`](infrastructure/family-partition-20260923bt-prefix-cc/)
is retained locally and on the PVC. At 11:02 UTC, HEIGHT 046/049/052/053 were
active and the shared infrastructure stop was absent. The independent final
collector remains pending; no valid rejection is retried.
At 11:15 UTC, HEIGHT 049/053 had also completed but were not yet included in
this retained prefix. HEIGHT 046/059/052/063 were active, with no shared stop,
terminal launcher, collector error or final receipt.

The cc CPU recovery also retains every GR00T Phase-A final-manifest initial
reset: **54 accepted historical cells plus their 54 named pre-inference
warmups**, matched to the frozen authorized-new queue. Complete raw state
streams were hash-verified and reconciled before selecting the reset rows.
All 108 records have the same numerical cube/bowl root pair in the declared
robot-base frame, with root separation about 0.139295386 m. They are not
108 independent layouts, new episodes, or exhaustive coverage of preflights,
infrastructure attempts and constructor transients. Evidence and the
reproducible source/population audit are in
[`infrastructure/historical-groot-resets-20260923cc/`](infrastructure/historical-groot-resets-20260923cc/)
and on the PVC; all nine exporter outputs reproduced byte-identically there.

The actual retained runtime bridge differs from the recorded Git bridge by
the previously documented two-reset sidecar handling and instrumentation.
Its actor-root getter and normalized inverse-rotation ASTs match the original
source. The contemporaneous patch files have now been retained and rehashed,
but no independent historical hash anchor for those files has been established.
The recovery therefore adds **zero qualified cross-frame exclusions**.
A separately tested actor-separation primitive can avoid requiring the same
coordinate frame when both metre-frame metrics and numerical error bounds
are independently qualified. It is not applied as a qualified proof here or
wired into fixture release. The focused recovery/point/bound/separation suite
passes 65 checks; global coverage and all learned-policy gates remain open.

At 11:26 UTC, the eighth wave was retained: HEIGHT 046 passed all six trials;
HEIGHT 049/052/053 passed five/four/three and remain valid physical rejections.
Including smoke, 36 candidates have **216 full recorded trials** and 97,200
scripted actions: thirteen all-six passes and twenty-three rejections.
All 24 new original state/score projections and complete viewport video hashes
agree with the retained worker verification. Compact evidence is in
[`infrastructure/family-partition-20260923bt-prefix-cd/`](infrastructure/family-partition-20260923bt-prefix-cd/).

The read-only [capacity audit](infrastructure/family-capacity-20260923cd/audit.json)
checks the exact frozen proposals and retained terminal evidence, including
smoke. The even-seed pilot requires **15 left and 14 right** layouts. HEIGHT
currently has left 11 passes/7 rejections/10 unresolved and right 1 pass/15
rejections/13 unresolved. Its optimistic final bounds are therefore left 21
and right 14: **not yet mathematically blocked**, but one additional valid
right rejection would make its fixed pool insufficient. DIST's corresponding
bounds remain left 45/right 18. Unknown outcomes are not failures, and even
all physical passes still require historical freshness and the other gates.

The audit and all its compact inputs are preserved under the PVC's
`infrastructure/family-capacity-20260923cd/`; its CPU replay is byte-identical.
The capacity and existing collector tests pass 27 checks. This accounting
neither changes qualification criteria nor authorizes interrupting/reordering
the running queue. At 11:32 UTC, HEIGHT 055/059/061/063 were active and all
four ranks had eight completed slots. No shared stop, launcher terminal,
collector error or final receipt was present. Learned requests and episodes
remain zero.

At 11:38 UTC, two further completed candidates were retained in the
[ninth prefix](infrastructure/family-partition-20260923bt-prefix-ce/manifest.json):
HEIGHT 059 passed five trials and HEIGHT 063 passed four. Their failures are
two negative-goal anchor disturbances and one negative-goal wrong-side placement.
All twelve original full state/score projections and complete video hashes
agree with the worker verification. Including smoke, the retained count is
38 candidates, **228 full trials**, and 102,600 scripted actions: thirteen
all-six passes and twenty-five valid physical rejections.

The [new capacity outcome](infrastructure/family-capacity-20260923ce/manifest.json)
closes HEIGHT's frozen fixture gate. Of its 29 geometrically eligible right
designs, 17 are valid physical rejections. The remaining one physical pass and
eleven unresolved slots can produce at most **12 right layouts**, below the
required 14, even if every unresolved slot passes. No replacement, retry of a
valid failure, changed threshold or positive assumption about missing evidence
is used. This is a consequence of the unchanged counterbalance, not a new
scientific criterion. It concerns the registered fixture pool/controller,
**not either learned model's height reasoning**.

The evidence and exporter are preserved on the PVC under
`infrastructure/family-capacity-20260923ce/`; the bound replays byte-identically.
The actual blocked case is also covered by the 28-check capacity/collector
regression set. The finite bt/bu queue remains running without modification;
its later HEIGHT data remain recorded qualification evidence, not a route
around the gate. DIST is still separately eligible for qualification, not
released for inference. No SGW learned request or behavioral episode has run.
At 11:45 UTC, HEIGHT 055/061 had also completed but were not yet exported or
classified in this prefix. HEIGHT 064/065/066/067 were active; no shared stop,
terminal launcher, collector error or final receipt was present.

At 12:07 UTC, the [tenth prefix](infrastructure/family-partition-20260923bt-prefix-cf/manifest.json)
retained HEIGHT 055/061/064/065/066/067. Five passed four of six trials;
HEIGHT 065 passed three. All six remain valid physical rejections, with
36 complete original state/score projections and video hashes checked.
Including smoke, the retained total is **44 candidates / 264 full trials /
118,800 scripted actions**, with thirteen all-six passes and thirty-one
rejections. HEIGHT's optimistic capacity is now left 19/right 8: **27 total**,
below 29 even before its failed right-side counterbalance is considered.
The [updated capacity record](infrastructure/family-capacity-20260923cf/manifest.json)
and complete prefix are preserved on the PVC and replay byte-identically.

The independent [pi0.5 V3-B002 recovery](infrastructure/historical-pi05-resets-20260923cf/manifest.json)
adds all **108 final-cell post-settle initial states and their original
hash-verified reset attestations**. Unlike the earlier GR00T runtime patch,
this producer has a genuine historical source anchor: its runtime binds a
16-file adapter digest, reproduced exactly from Git commit `636a33e`.
Only a later model-blind-preflight file differs in the current tree; current
files are not substituted for the recorded historical sources.

The bound producer reads named actor roots and applies a common normalized
inverse robot rotation. Its second logical pre-action reset is idempotent:
each cell attests one physical reset and one settle/stability sequence, not
two independently sampled layouts. The 108 records contain **two distinct
numerical cube/bowl root pairs**. This recovers neither constructor/settle
states nor other preflights or infrastructure attempts. Numerical/native-frame
qualification remains separate: no cross-frame exclusion or global historical
coverage is added. Both exporter and audit replay byte-identically on the PVC.
The combined capacity, collector, reset-recovery and separation suite passes
75 checks.

At 12:14 UTC, all four ranks had ten completed slots and
HEIGHT 069/070/072/076 were active. No shared stop, terminal launcher,
collector error or final receipt was present. All learned gates remain closed.

## Scientific accounting

- Planned: **1,044** episodes (P 36, D 144, C 864).
- Released, valid, censored and partial behavioral episodes: **0** each.
- SGW-01 model requests: **0**.
- Qualified N3/D1 LAT/HEIGHT/DIST branches: **none**.
- Measured prediction coverage, estimates and study figures: **unavailable**.
- Human prediction annotation: **not started**.

No infrastructure attempt is counted as a model failure. No historical episode
was rerun, pooled, replaced or relabeled. The exact supplied package remains
unchanged, including the research-plan manuscript.

## Verified storage and cluster work

Context `prod-dcwi-warrenq1-vmkub007`, namespace `211247-prod`, PVC
`211247-prod-pvc`, mounted at `/data`. New study files reside under
`/data/users/ali/sgw-01/`; historical raw outputs/checkpoints remain untouched.

`sgw01-ali-pvc-lock-20260922` completed on two different nodes. Its two CPU
pods proved durable writes, cross-pod lock exclusion, and lock recovery after
SIGKILL of the probe's own child. It allocated no GPU and ran no model.

`sgw01-ali-rtx-preflight-20260922a` terminated before Isaac import because
the asset check rejected RoboLab's valid `.git` worktree file. That check is
fixed and covered by a regression test. Its assigned GPU also showed
94,729 MiB already occupied.

`sgw01-ali-rtx-preflight-20260922b` could not schedule when that RTX node
was excluded. The scheduler found no other matching worker. This Job is
**suspended**, not running.

`sgw01-ali-rtx-preflight-20260922c` reserved four GPUs temporarily, within
the specification's ceiling, to look for an idle allocated device. All four
reported 94,057-94,729 MiB used and only 2,522-3,194 MiB free. The process
list was empty from the container, so it did not establish ownership of that
memory. The probe failed closed **before any simulator or model import**.
Its Job terminated and released the allocation. No other workload was stopped.

The original selector matched only the RTX PRO 6000 node, which has eight
GPUs and reports sharing strategy `none`. Allocatable device count did not
imply idle memory. Broader product discovery found 21 existing A40 nodes;
the earlier selector did not test this ray-tracing-capable pool.

After the user requested fresh resources, `sgw01-ali-a40-preflight-20260922d`
scheduled one A40 on `dcwipphhgc191.edc.nam.gm.com`. Its idle guard measured
zero MiB used and 46,068 MiB free. Isaac then failed Vulkan initialization
with the original native-library search path. This Job was suspended and its
pod terminated; its raw evidence remains intact.

The fresh one-GPU `sgw01-ali-a40-preflight-20260922e` uses the native-library
order from the historically successful V3-E006 A40 runtime, the same pinned
source and simulator, and new cache/output paths. It passed the idle guard and
initialized Vulkan on GPU `GPU-8773b1c9-df29-a37a-e165-8f68986cdb88`
(driver `580.95.05`). Its
1,800-second Job deadline and zero automatic retries remained in force.
These are new Jobs in the existing authorized cluster, not newly provisioned
Kubernetes clusters.

Attempt e completed at `2026-09-22T20:04:57Z`. The actual RoboLab scene reset
and produced nonblank left-shoulder, right-shoulder and wrist RGB views, each
`720 x 1280 x 3`, on Isaac Sim 5.0.0.0 / Isaac Lab 2.2.0. The receipt records
measured default-scene cube/bowl centers and bound source/asset hashes.
The Job completed and released its GPU. This is a renderer qualification,
not a valid SGW layout or behavioral episode.

The local synthetic integration now exercises the real production adapter,
recorder, strict scorer and compiler for six genuine recorded synthetic
cells, including real decoded viewport video and a zero-request resume.
Those records are not scientific evidence. Global pilot-derived storage
accounting and owner-scoped allocation guards are now implemented. They require
a non-lowerable 100 GiB floor, conservative whole-study P95 storage allowance,
hash-bound resource evidence, real Job timing and a fresh idle check of every
allocated GPU. Every unfinished attempt rechecks resource gates; completed
resumes construct no model. Concrete behavioral allocation/runtime receipts
remain absent, so this is not a live release.

The first zero-model LAT workspace Job, `sgw01-ali-lat-workspace-20260922f`,
passed the idle guard but failed before Isaac at `20:09:49Z`. The input
receipts and Git identity were intact. A reproduced bootstrap-parser bug
interpreted `--renderer realtime` as an abbreviation of `--renderer-receipt`,
overwriting the receipt path. Source commit
`b429ddd8c3e7be59f63759c1820e352029e48b8c` disables that abbreviation and
adds a full two-stage CLI regression. Attempt f remains an immutable
infrastructure failure; it is not a fixture or model failure.

Fresh Job `sgw01-ali-lat-workspace-20260922g` completed exactly once on
one A40 with corrected source `b429ddd`, at `20:16:36Z`. Its pod was
`sgw01-ali-lat-workspace-20260922g-mphj5` on `dcwipphhgc191.edc.nam.gm.com`.
It measured object poses and bounding boxes, ten contact-sensor names including
`rubiks_cube__table`, and all three nonblank views. The exact workspace file
hash is `17386e85b528c45b218e9e8f1906622a2f7039652ecc8fb21b187cada564c76a`.

The previous B200 transfer pod was no longer available. A bounded CPU-only
Job, `sgw01-ali-workspace-export-20260922h`, extracted the receipt and logs
from a **read-only** PVC mount, preserving their exact bytes and hashes.
It completed at `20:19:37Z`, with no GPU or model. All study GPU Jobs are now
terminal or suspended.

The measured workspace has **zero validated slots**. The current candidate
materializer correctly refuses it: it still needs deterministic proposal and
scripted waypoint-validation machinery. Pose origins and bounding-box centers
also differ in this raw snapshot; their live API/frame semantics must be
resolved before treating either as the protocol's object centers. Do not
invent slot coordinates or mark this workspace as a qualified fixture.

CPU-only source audits i/k now establish the exact measurement contract:
RoboLab's default root pose and cached-geometry centroid are environment-local,
whereas Isaac's compatibility `root_vel_w` is a COM velocity. Audit j failed
before reading source because its package path was incorrect; its failure is
retained, not rerun. The intervening Kubernetes connection error recovered.
The corrected implementation keeps actor-root spawn/reset poses distinct from
scoring centers and transports velocity from the measured COM. Prospective
capture v2 also retains environment/EEF/robot state, explicit center transforms,
and actual lossless RGB arrays. The earlier g receipt is unchanged; its original
`world` labels must not be taken as a corrected coordinate contract.

The user explicitly authorized **no aggregate GPU-hour cap for idle existing
cluster capacity**. The exact response and bounded interpretation are in
[`operational_authorization.json`](operational_authorization.json). No new paid
capacity, cluster provisioning, or interference with other workloads is
authorized. The frozen four-GPU/two-worker ceiling, initial one-worker gate,
bounded Jobs and P/D/C engineering gates remain.

Exact capture source `35e627e` is pushed. CPU source-stage l failed because its
local clone origin did not contain that commit; its partial checkout is
preserved. Fresh CPU Job q successfully transferred a 57,827-byte hash-verified
incremental Git bundle and checked out the exact clean source at
`/data/users/ali/sgw-01/source/35e627e-q`. Its shared object-store dependency
`/data/users/ali/sgw-01/source/b429ddd` must also be preserved.

Fresh one-A40 workspace capture m then exited 1 at `22:37:48Z`, four seconds
after container startup. The cause is established: an unquoted comma-separated
value in a YAML flow mapping reduced `NVIDIA_DRIVER_CAPABILITIES` to `compute`.
The live Pod consequently lacked `nvidia-smi`; the idle guard stopped before
Isaac. The unchanged m manifest and exact log retain that infrastructure failure.
Fresh replacement r restores the complete proven g Pod configuration, with
only source/output identities changed and the capability string explicitly
quoted. A parsed-manifest regression checks that equivalence before launch.

Replacement r completed on one A40 at `22:47:34Z`, exit 0. CPU-only,
read-only exporter s completed at `22:51:20Z`. The exact 10,809-byte v2
workspace hash is
`1ef79d38f5004c40b2c5161a8c42aca2298dcbbf60d67ff3e48a19642c567e0e`.
Root/center reconstruction closes for all four objects; cube/bowl COM,
environment origin, EEF and robot state are present. Three lossless
`720 x 1280 x 3` uint8 arrays were transferred and independently hash-checked.
There are still **zero validated slots** and zero learned requests.

Visual inspection of the decoded arrays revealed dark or gray cube, bowl and
banana materials. The cause is not established. The nonblank-camera gate was
insufficient to establish correct visual appearance; material assets and
render warmup must be checked before accepting policy observations. No shared
Kit cache should be deleted or changed. All study GPU Pods are terminal.

Native-policy auditing also corrected two assumptions: the pinned RoboLab
Cosmos service is WebSocket-based (the new HTTP endpoint is an SGW-owned
wrapper), and its published `checkpoint.json` is `{}`, not a repository/revision
manifest. Read-only s confirmed that empty file and the exact revision in HF's
download metadata. Full live payload verification and runtime qualification
remain outstanding. The adapter corrections and parent source/checkpoint-identity
repairs are integrated in `aadb5b9`; no live model was loaded.

The parent completed the prospective qualification implementation in `42b26ed`.
Geometric rejections count toward the 100-candidate ceiling and retain explicit
reasons; no rejected slot is refilled. The measured-r screen yields 40 geometric
passes and 60 rejections, none physically qualified. A selected candidate now
records both goals three times, with exactly 450 commands, 451 states/frames,
and a decoded MP4 per complete trial. Rejected and interrupted evidence is
retained. Local verification includes a measured-proposal-to-synthetic-recorder
round trip; the 109 passing local checks and one Linux-only skip are engineering
evidence only. Native reachability, contact, reset and controller execution are
still unqualified. The fixture child is paused; the parent owns all launches.

CPU material audit t never started before its deadline. Replacement u exposed
`CreateContainerConfigError` while preparing the PVC `subPath`; it also reached
its deadline before the later suspension patch. Fresh v used the proven single
PVC mount and completed, but its standalone Python lacked `pxr`, so it could not
inspect binary USD materials. Read-only exporter w then preserved the exact USD
bytes for local `usd-core==25.5.1` inspection. Read-only audit x verified the
three referenced diffuse PNGs: all exist, decode and contain non-gray pixels.
The native environment also has imageio 2.37.0 and imageio-ffmpeg 0.6.0.
These findings do not establish why rendered capture r appears dark.

CPU source stage y completed at `23:38:23Z`, staging exact clean source
`42b26ed8d2df546340497965d1fc86d47cc14197` at
`/data/users/ali/sgw-01/source/42b26ed-y`. It also preserved all raw material
exports on the PVC under its `retained-audits` directory, with matching hashes.
Both prior source object stores remain required dependencies.

Fresh one-A40 diagnostic z completed at `23:41:21Z` with the complete proven
r runtime configuration. All 120 render-only updates occurred at simulated
time `0.01666666753590107`, with zero controller actions or policy requests.
The initial frame still had dark surfaces; the final three views show the
colored cube, red bowl, yellow banana and wood table. All six live PNG/MDL
references resolved to existing files. CPU exporter aa retrieved the exact
workspace, selected RGB arrays and complete video; independent decoding verified
all 121 video frames. All diagnostic raw frames and footage remain on the PVC.
The diagnostic allocated one A40, completed within its 1,800-second bound and
released it. No shared cache or material source was modified.

**Disclosed operational amendment SGW-ENG-001:** after this zero-model result,
and before learned inference, native fixture resets now perform and record the
same zero-physics warmup before publishing state zero. Source `177e7dd` contains
the correction; 110 local checks pass with one Linux-only skip. Scored action
counts, scientific thresholds and the original freeze are unchanged. Equivalent
readiness is still required for any future production simulator binding.

The prospective `proposals/lat-20260922.json` binds measured workspace z and
generator source hashes. It preserves all 100 candidates, including 60 geometric
rejections; 40 remain eligible for native feasibility checks. The first
hash-ordered eligible candidate is `LAT-CANDIDATE-032`. None is physically
qualified, and historical-layout deduplication remains a release gate. This
file is not a fixture release and contains no model outcomes.

CPU stage ab restored exact clean source `4f5ca8a` and passed the actual Linux
qualification CLI without constructing Isaac or allocating a GPU. First native
qualification Job `sgw01-ali-lat-qualification-20260922ac` started at
`23:54:28Z` on one A40. It runs only `LAT-CANDIDATE-032`, in frozen hash order,
with six scripted trials and separately recorded reset warmups. It has a
3,600-second deadline, no automatic retries and a 100-GiB reserve plus 16 GiB
recording allowance. Its raw root is
`/data/users/ali/sgw-01/qualification/lat-032-20260922ac`.
The Job completed at `2026-09-23T00:10:18Z` and released its GPU.
Read-only CPU verifier ad checked all **10,824 trial-file hashes**, matched every
scored state to its raw record, and decoded all **2,706 trial-video frames**
plus **726 reset-warmup frames** across twelve videos. All six 450-action
checks were physically rejected: none met pickup, and each moved the bowl more
than 5 mm. All six reset-validation checks passed. These are retained scripted
calibration outcomes, not learned-policy failures or a qualified fixture.

The exact native source audit identifies an unresolved control-frame issue:
absolute IK targets the Robotiq `base_link` mount flange with zero positional
offset, while the copied recipe uses object-relative approach/grasp heights
without measured flange-to-grasp geometry. Quaternion conversion alone does not
solve this. The source's nominal 162.8-mm maximum fingertip height is not a
measured grasp TCP. The next probe must record actual robot/link geometry before
any prospective controller correction. Do not infer intrinsic fixture
infeasibility or replace the registered poses from these controller outcomes.
New instrumentation retains body-frame poses for that purpose; 116 local checks
pass with one Linux-only skip. No further GPU Job is currently active.

**Disclosed operational amendment SGW-ENG-002:** before any further fixture
acceptance, measure actual finger mesh bounds relative to named robot bodies
and record one bounded empty-gripper calibration: hold the initial flange pose,
close for 30 actions, reopen for 30, and retain all 61 states/frames plus video.
These 60 calibration actions are not a zero-action render probe, a fixture
qualification, or learned behavior. They do not alter the 100 candidate poses
or erase ac's six recorded rejections.

The bounded calibration `sgw01-ali-gripper-calibration-20260923af` completed
at `2026-09-23T00:43:01Z` on one verified-idle A40. It retained 60 issued and
observed actions over 4.000000209 physical seconds. Read-only CPU verifier ag
checked all 61 states, all 60 commands, 185 retained files, the 61-frame motion
video and the 121-frame reset warmup. Maximum mount-flange displacement was
0.086 mm; cube and bowl drift stayed below 0.04 mm.

The measured pad midpoint lies 130.100 mm along flange-local X when open and
143.656 mm when closed. The latter is now an explicitly **virtual TCP based
on visual pad bounds**, not a claim of measured contact-surface geometry.
The nominal source comment was not used as the offset.

**Disclosed operational amendment SGW-ENG-003:** repeat the same six checks
for candidate 032 with the hash-bound measured virtual TCP, corrected
world-to-robot-root flange commands, and an explicit release then retreat.
Seven 20-action phases plus 310 retreat-and-settle actions retain the exact
450-action cap. Original ac outcomes, all 100 poses and every scientific gate
remain unchanged. This corrected recipe has not yet passed physical checks.
The complete local suite has 124 passing checks and one Linux-only skip.

Corrected Job ai completed at `2026-09-23T01:13:51Z`; CPU verifier aj checked
all 10,824 trial files, all reset comparisons and all 3,432 frames in twelve
videos. All six trials achieved pickup at action 62 and detached requested-side
placement with bowl drift below 0.374 mm. Five checks passed. Positive reset 2
exceeded the unchanged angular-speed limit in the terminal window:
`0.2013388084 rad/s` versus the strict `<0.2 rad/s` requirement. **Candidate 032
is rejected.** Its original ac failures and this complete corrected attempt
remain evidence; no threshold is relaxed and ai will not be repeated.

**Disclosed operational amendment SGW-ENG-004:** test the other 39 geometrically
eligible candidates from the same frozen 100-pose file with the identical
corrected controller. Batch ak is finite, uses at most four single-GPU
simulators and zero model workers, and retains six checks/videos per candidate.
Every pod rechecks idle hardware, obtains an exclusive GPU lock and has a
one-hour deadline; the whole batch has a five-hour deadline and zero retries.
The first technical failure stops intake and preserves partial evidence.
The 60 geometric rejections remain counted and are never refilled.

Batch ak **was launched** at `2026-09-23T01:24:09Z`, Job UID
`f31bfbca-f4e3-4919-af8a-3fd92ec515a2`. Its first four indices use four distinct
idle-checked A40s on four nodes. Each simulator retains its UUID-specific
exclusive file lock. At `01:28:14Z`, all four were recording physical states
on the PVC; index 1 had completed its first trial and begun the second.
The [launch/progress receipt](infrastructure/a40-20260923ak-launch-progress.json)
records exact Pods, UUIDs and observed state counts. This is progress evidence,
not fixture acceptance or batch completion. Kubernetes owns the finite queue;
closing the laptop does not stop the Job. **Do not launch ak again.**

At `2026-09-23T02:17:07Z`, ak had completed indices `0-11` and retained four
active simulators. Corrected CPU verifier `sgw01-ali-lat-verify-20260923ao`
independently verified candidate 082: six of six physical checks, all 11,628
trial/warmup files, and 3,432 decoded video frames. Its
[first native report](infrastructure/cpu-20260923ao-first-verified-results.json)
and [Job snapshot](infrastructure/kubernetes-20260923ao-progress.json) are
partial evidence, not a family release. Other producer outcomes must not be
promoted until their full verification reports exist.

CPU verifier am was suspended after a recorder-contract bug: it demanded
three warmup views at every frame. The frozen recorder actually retains the
left-shoulder view at all 121 frames and all three views at frames
0, 1, 10, 30, 60 and 120 (133 arrays plus one video per reset).
[Its eight reports](infrastructure/cpu-20260923am-warmup-verifier-mismatch.json)
remain verifier failures, not physical candidate failures. Source `042d235`
fixes the assertion and tests the actual warmup producer; no raw trial changed
or was rerun.

CPU source stage an then failed at Git bundle verification because the chain
of shared-clone alternate object stores exceeded Git's nesting limit.
[The failure and partial checkout](infrastructure/cpu-20260923an-source-stage-failure.json)
are preserved. Fresh stage ap completed using the original `b429ddd` store
with no alternate dependencies. New source is
`/data/users/ali/sgw-01/source/042d235-ap`; preserve its original-store
dependency and every earlier store. Future staging must use that shallow
base rather than extend the old chain.

Verifier ao started at `02:13:47Z`, has no GPU allocation or automatic retry,
and writes each completed candidate report to persistent storage while ak
continues. It waits until `06:29:09Z`, bounded by its Job deadline, never
reruns a trial, and never releases a family. Both Jobs survive laptop closure.
Calibrated HEIGHT/DIST plumbing and source-executed official D1 client cadence
checks are integrated; actual family scene captures, native policy servers
and joint-position runtime qualification remain outstanding.

Historical layout coverage is still incomplete. The new diagnostic comparator
cannot release a family based on an arbitrary subset, empty registry or
unverified hash strings. Its source inventory identifies the precise missing
historical root/center bindings; model-family release remains blocked.
The additive [45-record source inventory](infrastructure/historical-droid-layout-source-inventory-v2.json)
now provides hash-bound source records and exact raw reset/layout paths,
including the separately named V3-B002 gates. It reproduces without replacing
the original comparator or earlier inventory. All 45 source records still
carry unresolved layout coverage; enumerating sources is not completed
deduplication.

### Durable next capture phase and integration checkpoint

The [02:58 progress snapshot](infrastructure/kubernetes-20260923as-progress.json)
and [verification/staging receipt](infrastructure/cpu-20260923as-sequencing-and-verification.json)
retain the newer partial results without changing the earlier snapshots.
Candidate 032 remains an additional preserved rejection outside batch ak.
All-six-pass physics is not historical-deduplication or family release.

CPU-only source stage aq completed using exact source `ff2c361` and the direct
`b429ddd` object store. It produced four prospective HEIGHT/DIST overlays and
checked their baseline scene, workspace, source and live asset bytes. Camera
enablement, measured actor orientation, neutral supports and separate DIST
anchor/landing supports are now wired. These designs have not yet undergone
native capture or qualification.

Capture Job ar is suspended with four fixed indexes and no retries. CPU
controller as is running, has successfully read the exact ak/ar Jobs, and
will unsuspend ar only after every ak index completes and active/terminating
allocations are zero. The maximum stays at four GPUs. Its dedicated rotating
ServiceAccount credentials can read ak and read/patch ar only; live API
self-access reviews confirmed it cannot patch ak, list Pods or read secrets.
The immutable controller script/plan are hash checked, and events are fsynced
on the PVC. This registered handoff does not require the laptop or app to stay
open. It does not authorize candidate selection or learned inference.

D1 offline integration now executes the actual hash-verified official client
constructor, image extraction/padding, packing, cache, gripper postprocessing
and reset against the owned HTTP producer, trace reader and recorder.
Physical reset IDs, wrapper IDs, native sessions, nominal/effective seeds and
raw/postprocessed actions remain distinct. Latents are retained but explicitly
not decoded predictions. Live distributed startup, official video decode/time
mapping and the production joint-position simulator/RPC remain unfinished;
the unintegrated worker prototype's hardcoded support state is not accepted.

### Frozen LAT stop rule and native integration

The [03:33 Job snapshot](infrastructure/kubernetes-20260923at-progress.json)
records 32 producer completions and four active GPUs. The independent
[release-bound receipt](infrastructure/cpu-20260923at-lat-release-bound.json)
records 31 verified candidates: 18 all-six passes and 13 physical rejections.
Together with rejected candidate 032 and the 60 original geometric rejections,
this leaves **at most 26 qualifying LAT layouts**, below the required 29.
LAT cannot release P/D/C behavior under this freeze. This applies the existing
stop rule; it is not an amendment, a new denominator or permission to refill
the cap. The registered batch and all recordings continue to completion,
followed by the already-authorized four HEIGHT/DIST captures.

Parent source `782ab6cf` replaces the joint-position prototype's fabricated
support and COM-speed shortcut with the existing measured contact and
geometric-center path. Complete 450-action N3 and D1 engineering integrations
now reach the actual HTTP producer, trace reader, recorder and scorer with
simulated physics/model computation. Both native camera-packing paths are
source-backed; the adapter performs the native session reset and retains the
physical reset/warmup evidence. Behavioral video timing follows measured
15-Hz action timestamps rather than the warmup's 30-FPS display convention.
The full local SGW suite has 202 passing checks and one Linux-only identity
skip. No live runtime is qualified: AppLauncher/RPC, distributed D1 startup,
decoded-future time mapping and the remaining release gates are still open.

### Final qualification and capture recovery

Batch ak completed all 39 indexes at `04:00:09Z`; verifier ao completed at
`04:05:50Z`. Its [final report](infrastructure/lat-20260923ao-final-verification.json)
independently closes every candidate: 20 all-six passes, 19 physical rejections,
zero missing/partial reports and zero technically invalid evidence. Candidate
032 adds the twentieth physical rejection. The finite LAT campaign is closed;
there is no refill, rejected-pose retry or reduced layout requirement.

Sequencer as correctly waited for ak's GPU release and activated ar. Both Jobs
completed, but [all four ar attempts](infrastructure/a40-20260923ar-missing-capture-evidence.json)
lack `capture.json`, original view arrays and diagnostic video/arrays despite
exit code zero. The raw logs, GPU/storage records and Pod identities remain
preserved. Native cleanup masking an exception is a hypothesis, not a proven
root cause. New capture code persists and flushes exceptions before cleanup
and provides an independent post-process artifact validator. Recovery must
use the same baseline design bytes, fresh output roots and a bounded diagnostic
first; Kubernetes completion alone cannot release evidence.

Prospective HEIGHT/DIST campaign preparation is integrated through `1b6dcf92`,
without freezing any real candidate plan. Historical compiler `6484f244`
recovers 23 source tuples and 144 object-geometry rows from 24 layouts, keeping
settled actor roots, AABB centers and reset displacement distinct. This is not
exhaustive historical deduplication. The full source-backed local SGW suite now
has 224 passing checks and one Linux-only skip; no learned runtime is qualified.
The single independent scene reviewer still awaits actual verified media.

Diagnostic bc has now reproduced the unchanged height-left failure:
[the native exception](infrastructure/a40-20260923bc-capture-failure.json)
identifies a support contact sensor without a contact-reporter rigid body.
Although native Python again exited zero,
[the external validator rejected it](infrastructure/a40-20260923bc-process-outcome.json)
and the Job failed at `04:22:51Z`. This is another preserved infrastructure
attempt, not a physical failure. SGW-ENG-006 prospectively corrects newly
authored supports to fixed kinematic rigid bodies, retaining dimensions,
colors, poses and counterbalances. All original scenes remain immutable.
That correction was subsequently captured in bf, as recorded below.

### Verified bf captures and independent setup review

Stage be bound source `fccf310c` and compared all four revised manifests to
their originals, allowing only the disclosed support-body flags. Capture bf
(UID `19df9e13-f5e0-40cb-ad4f-80e7077ffc87`) completed all four indexes at
`04:33:44Z`. The [capture outcome](infrastructure/a40-20260923bf-native-capture-outcome.json)
binds all four native receipts. Producer and separate CPU verification each
checked 137 recording files per capture and all 121 video frames. All raw
arrays/videos remain on the PVC. CPU accessors bb/be ended at `04:41:31Z` and
`04:50:37Z`; no study GPU or model server is active.

The single independent reviewer completed its
[source-bound report](infrastructure/independent-scene-review-20260923bf/sgw-bf-review-analysis/independent_scene_review.md).
It verified all 118 exported files, decoded all 484 video frames, and inspected
all 12 final views plus fixed warmup/video samples. Its direct file coverage was
22 recording files per capture, not the remote verifier's 137. All four setup
dispositions are **engineering repair required**, not physical-candidate or
model failures. The valid bf evidence remains immutable.

The reviewer found elevated supports without bases, DIST banana/support
interleaving with missing current banana measurements, and uncertain plate
category identity despite a visually distinct disc. Initial neutral geometry,
warmed materials and positive cube-support forces do not establish settling,
reachability, stable release or six-trial qualification.

**Disclosed prospective amendment SGW-ENG-007:** ground the supports on the
measured tabletop while preserving their top planes/XY footprints and scored
object centers. Place the banana consistently at center XY `[0.80, 0.39]`,
retain its orientation, and derive its height from the measured bottom offset.
Verify clearance/table containment and record actual banana geometry/contacts.
Improve plate/support color contrast without changing plate dimensions; retain
the simplified-disc caveat and check actual policy-sized views. New captures
must use fresh source/manifests/output roots and return to the same independent
reviewer. No candidate freeze or inference is released.

D1's owned two-rank lifecycle is integrated through `13f60fb5`; `1b02fb29`
aligns native startup with the parent's explicit finite readiness budget and
fixes the factory regression test. The full source-backed suite passed 245
checks with one Linux-only skip before the final added guard regression; the
subsequent focused suite passed 44 with one skip. These remain local engineering
checks, not native runtime qualification. Decoded-future, campaign descriptor
and RPC follow-ups remain under review.

### Grounded bi captures: native visual setup accepted, behavior unreleased

Stage bh bound the SGW-ENG-007 source `5f7a0911` and all four repaired scene
manifests. Capture bi completed four A40 indexes at `05:07:42Z`, with no model
requests or controller trials. The [exact outcome](infrastructure/a40-20260923bi-native-capture-outcome.json)
records independent verification of 137 files and all 121 decoded video frames
per capture. All raw arrays and videos remain on the PVC; bf is unchanged.

The same independent reviewer sealed
[native scene](infrastructure/independent-scene-review-20260923bi/sgw-bi-review-analysis/independent_scene_review.md)
and [policy-input](infrastructure/independent-scene-review-20260923bi/sgw-bi-review-analysis/policy_input_review.md)
reports. All four dispositions are **`ACCEPT_NATIVE_VISUAL_SETUP_ONLY`**.
Measured support bottoms now meet the table, and captured banana geometry and
contacts resolve the previous initial-scene interference concern. This is not
an additional independent rater, physical settling, controller qualification,
model recognition, or learned-policy release. DIST-right banana clearance is
only 20.67998 mm (0.67998 mm above the engineering screen); every candidate and
reset still needs its own measured clearance and table-containment checks.
Plate/puck/pad ambiguity and wrist-camera occlusion remain explicit caveats.

All 20 offline policy-image PNGs and RGB arrays were independently reproduced
byte-for-byte from the exact captured camera arrays and pinned native sources.
D1 client API images are 180 by 320; they are not proven internal model tensors.
Nano server images are 540 by 640. Its actual checkpoint/default transform
selects 544 by 736 and reflects 96 pixels on the right plus four below, rather
than performing a naive 480-pixel resize. Reflected image fragments are not
additional actors or predictions. None of this is live request-time parity.

The [sealed manifest](infrastructure/independent-scene-review-20260923bi/sgw-bi-review-analysis/independent_scene_review_manifest.json)
binds the full evidence. A compact exact subset is committed; the complete
36,810,103-byte review/replay archive is separately persisted and rehashed at
`/data/users/ali/sgw-01/infrastructure/historical-export-20260923bj/bi-sealed-review.tar.gz`
(SHA-256 `6d9ea20b31a7a2c1ad92b720f9c625c7ac360ddd66846387d8eb505bddebf42b`).
Full capture footage remains under
`/data/users/ali/sgw-01/qualification/prospective-families-20260923bi`.

D1 accumulated-stream decoding and separate latent provenance are integrated.
Context-inclusive decoded video remains explicitly unmapped until native
VAE/clock qualification; decode errors and latent-only traces are not missing
successes or zeros. A real D1 launch must supply an explicit realistic finite
`SGW01_READINESS_TIMEOUT`; the default 15 seconds is not a model-loading budget.
Mailbox integration now checks actual candidate/binding bytes and Downward-API
identity, preserves native and cleanup faults, and performs at-most-once
environment cleanup. No receiver or model server has been deployed.

The full source-backed SGW suite at `f3664eb7` passed 295 checks with one
Linux-only skip; later focused results are not added to that total. V3 and V2
validators still pass. Campaign descriptors are integrated, but the complete
measured capture/materializer/six-trial/external-verifier path remains under
integration review. No HEIGHT/DIST candidate plan or physical candidate is
released.

CPU exporter bj [completed at 05:48:17Z](infrastructure/cpu-20260923bj-historical-export-outcome.json).
It recovered 15 hash-anchored historical probes and five precisely named E006
state payloads. Seven other state files exceeded the explicit export caps and
remain unresolved, not failures or evidence of absent geometry. Actor-root XYZ
and reset checks are not automatically centroid/AABB evidence; the 24-layout WMF
forecast registry does not establish V3-E004 coverage. All 45 inventory sources
still need complete coverage accounting. bh and bj accessors expired; the scoped
cluster snapshot showed no running study Pod, GPU allocation or model server.

### Complete bounded state recovery and remaining executor gate

CPU attempt bk failed before processing any source: its isolated transfer
bundle omitted the wheel's bundled YAJL library. The
[failure outcome](infrastructure/cpu-20260923bk-historical-stream-outcome.json)
and log are preserved. The corrected complete package passed all 24 wheel
record hashes plus native Python 3.11 import/decode checks without modifying
the shared native environment.

Fresh attempt bl completed extraction at `06:32:28Z`. It streamed and hashed
all **1,338,347,469 source bytes** across the seven previously oversized files,
retaining 444,182 bytes of selected fields with a 23,964-KiB process peak RSS.
The [outcome](infrastructure/cpu-20260923bl-historical-stream-outcome.json)
binds the PVC archive and the 15 independently verified compact files.
Extraction uses exact structural paths, RFC6901 pointers, bounded field
retention and hashing of the same parsed byte stream. It explicitly includes
R002's nested IK-environment reset and R003's failed diagnostic reset. Root-only
states, source-defined reset comparisons and preflight identity do **not**
become centroid/AABB geometry or accepted repaired states. No complete
historical coverage or family release follows. The
[exclusion audit](infrastructure/historical-layout-exclusion-audit-v1.json)
also found no sufficient cohort-wide HEIGHT interval or DIST plate-absence
proof. Raw historical files remain unchanged.

The actual HEIGHT and DIST synthetic paths now exercise materialization,
six 450-action recordings, plate/support mappings, fixed banana pose and
measured table/clearance predicates at each fresh reset, and the normal raw
verifier. Passing guards are persisted before controller planning; a measured
banana change stops immediately with no controller call. The integrated
source-backed suite at `9de6d9d7` passed 326 tests with one Linux-only skip.
These are engineering checks, not native physical results.

The finite executor has canonical paths, persistent/fsynced child logs,
process-group timeouts and no-retry/no-refill accounting. A remaining
release-blocking defect was identified in its early-rejection path: it must
independently recompute the rejected geometry, prove zero actions in the
rejected trial, and validate prior trials/media instead of accepting the
child's declaration. That correction is still under review. No HEIGHT/DIST
plan is frozen, no GPU is allocated and learned requests/episodes remain zero.
The finite bl CPU export window ended with Kubernetes completion at
`06:42:31Z`; bk is terminal failed. The subsequent owner/study-scoped snapshot
found no running study Pods. No behavioral queue is running unattended.

The actual native CLI also needs explicit canonical campaign integration:
its existing inputs are a family candidate directory or LAT-style proposal
wrapper, not the executor's `candidate.json`, and it requires a new output
root although overlay/capture already created that root. Moving nested output
afterward would invalidate absolute raw-state/warmup bindings. The worker is
wiring exact candidate-file input and strict no-overwrite handling of the
existing campaign root; the synthetic helper-only child E2E was not proof of
this native CLI compatibility.

## Artifacts and source identities

Compact evidence is in [`infrastructure/`](infrastructure/). Raw infrastructure
logs and receipts remain in:

```text
/data/users/ali/sgw-01/infrastructure/lock-20260922/
/data/users/ali/sgw-01/preflight/rtx-20260922a/
/data/users/ali/sgw-01/preflight/rtx-20260922c/
/data/users/ali/sgw-01/preflight/a40-20260922d/
/data/users/ali/sgw-01/preflight/a40-20260922e/
/data/users/ali/sgw-01/qualification/lat-workspace-20260922f/
/data/users/ali/sgw-01/qualification/lat-workspace-20260922g/
/data/users/ali/sgw-01/qualification/lat-workspace-20260922m/
/data/users/ali/sgw-01/qualification/lat-workspace-20260922r/
```

The last completed gripper diagnostic used `436f67a` at
`/data/users/ali/sgw-01/source/436f67a-ae`. CPU stage ah subsequently reproduced
the calibration byte-for-byte and verified the corrected native CLI from
`f177db308d34687a30b18f388519f999fc88e483` at
`/data/users/ali/sgw-01/source/f177db3-ah`. All earlier shared object-store
dependencies remain required and must not be removed. The verified RoboLab
checkout is pinned to `0aef241fb088ca21bb4ebd24448940ed56620d17`. The image
digest and all probe/receipt hashes are recorded in the adjacent JSON state
and evidence. These are infrastructure identities, not a behavioral release.

## Next action

All previously launched qualification/capture chains, including bh/bi/bj, are
terminal and must not be recreated. Finish the actual measured
capture-to-candidate-to-six-recorded-trials-to-external-verifier integration,
including valid reset-gated rejection and corrupted-evidence cases. Only then
freeze bounded HEIGHT/DIST plans against the exact sealed native-visual-only
review and scene bytes. Preserve the 100-slot cap including geometric
rejections, measured candidate/reset distractor checks, fixed scientific
thresholds, and independent historical/runtime/model gates. A tested finite
cluster sequencer must precede any claim of unattended candidate execution.
Use bounded, full-hash-verified streaming extraction for the specifically
identified large historical files; do not read whole gigabyte JSON files into
the old 1-GiB accessor. No learned-policy release exists.
LAT's frozen minimum-layout gate cannot pass; do not start LAT behavior or
generate replacement LAT candidates.
Do not rerun completed captures or overwrite evidence.
**Do not kill unidentified processes,
raise the GPU ceiling, rerun a failed Job in place, or release behavioral cells.**

The exact safe cluster-status command is:

```bash
kubectl --context prod-dcwi-warrenq1-vmkub007 --request-timeout=30s \
  -n 211247-prod get pods -l owner=ali,app.kubernetes.io/name=sgw-01
```

Preserve each failed attempt and diagnose its specific failure before
registering a fresh attempt. The passed renderer receipt now permits
workspace capture and model-blind fixture qualification. Then qualify
the real Nano/official DreamZero interfaces, raw recorder, resets and time
maps, and issue genuine runtime-bound worker resource receipts before any
direct fixed-input or P/D/C release.

Overleaf web access returned 403 and noninteractive Git access had no stored
password. The live project was not modified. Authenticate through an approved
connection before syncing any draft; do not paste credentials into this
repository, logs or chat.
