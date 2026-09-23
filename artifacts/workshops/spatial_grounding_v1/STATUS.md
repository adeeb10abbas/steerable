# SGW-01 status

**LAT qualification is complete: 20 all-six passes and 20 physical rejections,
including candidate 032, plus 60 geometric rejections. The frozen 29-layout
behavioral gate cannot pass. Four repaired HEIGHT/DIST baseline captures now
have verified receipts, views and complete videos. The new bi captures passed
independent native-visual-setup review after support/distractor repair; physical
candidate qualification and all learned-runtime release gates remain open.
Learned-policy requests and episodes remain at zero.**

The supplied study is committed on
`sz5vjy-gme-spatial-grounding-experiments`. Four Terra/Luna child sessions
implemented separate fixture, policy, persistence and analysis components.
Implementation and synthetic tests do not establish scientific readiness.

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
