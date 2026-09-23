# SGW-01 status

**Gripper geometry is measured and a corrected static controller is bound. Its physical qualification is pending; learned-policy episodes remain at zero.**

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

The source used by the last completed diagnostic was
`42b26ed8d2df546340497965d1fc86d47cc14197`, staged as a clean checkout at
`/data/users/ali/sgw-01/source/42b26ed-y` using the preserved 35e627e-q and
b429ddd object stores. The verified RoboLab
checkout is pinned to `0aef241fb088ca21bb4ebd24448940ed56620d17`. The image
digest and all probe/receipt hashes are recorded in the adjacent JSON state
and evidence. These are infrastructure identities, not a behavioral release.

## Next action

Stage the measured SGW-ENG-003 controller and run the same candidate's six
physical checks in a fresh bounded Job before
correcting the scripted controller. Preserve ac's six rejections and all 100
candidate poses; no learned-policy release exists.
Do not rerun completed captures or overwrite evidence.
**Do not kill unidentified processes,
raise the GPU ceiling, rerun a failed Job in place, or release behavioral cells.**

The exact safe cluster-status command is:

```bash
kubectl --context prod-dcwi-warrenq1-vmkub007 --request-timeout=30s \
  -n 211247-prod get jobs -l app.kubernetes.io/name=sgw-01
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
