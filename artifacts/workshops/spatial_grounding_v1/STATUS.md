# SGW-01 status

**A40 rendering and zero-model workspace measurement passed. No fixture or learned-policy episode is qualified yet.**

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
Native-policy source auditing and local runtime fixes are proceeding
separately without a model server. Consult the live study-owned Job status
before continuing; pending work is not recorded as complete.

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
```

The source used by the last probe was
`b429ddd8c3e7be59f63759c1820e352029e48b8c`, staged as a clean independent
checkout at `/data/users/ali/sgw-01/source/b429ddd`. The verified RoboLab
checkout is pinned to `0aef241fb088ca21bb4ebd24448940ed56620d17`. The image
digest and all probe/receipt hashes are recorded in the adjacent JSON state
and evidence. These are infrastructure identities, not a behavioral release.

## Next action

Collect the source-stage diagnosis and prospective workspace v2 evidence.
Then complete the missing deterministic model-blind workspace-to-slot/waypoint
stage and qualify physical fixtures from measured evidence. Do not rerun f/g
or overwrite their evidence.
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
