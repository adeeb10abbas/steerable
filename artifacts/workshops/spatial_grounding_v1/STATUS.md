# SGW-01 status

**A40 renderer passed; fresh zero-model workspace capture is running. Fixture qualification and model inference remain unreleased.**

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
accounting and GPU-hour budget enforcement are still missing from the worker;
the configurable disk floor alone does not satisfy the runbook.

The first zero-model LAT workspace Job, `sgw01-ali-lat-workspace-20260922f`,
passed the idle guard but failed before Isaac at `20:09:49Z`. The input
receipts and Git identity were intact. A reproduced bootstrap-parser bug
interpreted `--renderer realtime` as an abbreviation of `--renderer-receipt`,
overwriting the receipt path. Source commit
`b429ddd8c3e7be59f63759c1820e352029e48b8c` disables that abbreviation and
adds a full two-stage CLI regression. Attempt f remains an immutable
infrastructure failure; it is not a fixture or model failure.

Fresh Job `sgw01-ali-lat-workspace-20260922g` is running exactly once on
one A40 with corrected source `b429ddd`. Its pod is
`sgw01-ali-lat-workspace-20260922g-mphj5` on `dcwipphhgc191.edc.nam.gm.com`.
It passed bootstrap parsing and initialized Vulkan. Its workspace measurement
receipt remains pending; the fixture-qualification session owns collection.
The Job has a 1,800-second deadline and no automatic retry.

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

Collect the already-running workspace attempt g, then qualify the physical
fixtures only after its receipt passes. Do not launch a duplicate, rerun f,
or overwrite either attempt's evidence.
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
maps, and finish worker storage/budget guards before any direct fixed-input
or P/D/C release.

Overleaf web access returned 403 and noninteractive Git access had no stored
password. The live project was not modified. Authenticate through an approved
connection before syncing any draft; do not paste credentials into this
repository, logs or chat.
