Plan: workshops/corl2026_world_models/docs/ABLATION_SPEC.md

User authorizes new core study on GM, with parallel execution. New namespace: wmf_ablation_001_20260912.

- Located source/spec commit e67e6c4; created isolated codex/forecast-layout-gm-20260912 worktree.
- Cluster and runtime discovery running independently. IPv4 work Mac route works; GM API not yet reachable.
- Historical stopping sensitivity implementation delegated; no new model execution.
- Main batch depends on synchronized fixed-450 recording, model qualification, model-blind scenes, development and annotation freeze.
- Spec overrides earlier plan ranges: exactly 24 confirmation layouts; optional D2 excluded.
- Shared interfaces checked: recorder emits timestamped observation/request/action inventories for alignment and sampling; analysis consumes qualified mappings, never generated frame index as action index.
- No physical coordinates, horizons, capacity or qualification passes are invented.

- Completed schedule/seed preparation in defa113: 58 unreleased jobs, 232 cells, full 24-permutation confirmation assignment; bounded pinned-structured-artifact seed audit only.
- Completed historical stopping script in b6d022d: 27/27 ordered command pairs per fixed layout, retrospective outcome-dependent time explicitly retained.
- Parent verification: 43 workshop unittest tests passed; no new model episodes or generation requests.
- Exact source bundle restored and hash-verified on workstation in a new isolated checkout; original checkouts untouched.
- Final forced-IPv4 work-Mac SSH retry timed out. Required external state: reliable work-Mac access and GM network/VPN route.

- Independent review complete: both preparation scripts satisfy their bounded specifications; no actionable defects. Verified generated artifact counts/hash links, all shared strict-before-end stopping times, full condition ordering, and unreleased status. No model/runtime calls were made by the reviewer.

## 2026-09-13 cluster-independent execution milestone

- The workstation-to-GitHub-to-GM-to-GitHub-to-workstation queue route passed. Four distinct one-B200 workers completed infrastructure diagnostics; result commit `5c7c0ef8de093f10185b31d8b7a869fdb2ba7a83` was fetched and its published artifact hashes were verified. Another 28 workers remain scheduler-pending and are not available capacity.
- The cluster coordinator and workers use the PVC-backed queue and repo-scoped deploy key. The work Mac is no longer in the dispatch or result-return path. These diagnostics launched zero model requests and zero behavioral episodes.
- H01 and H02 declared archive targets were recovered on the GM PVC with 22/22 paths present and 21/21 supplied hashes matching. The selected H01 closure is 17/17 files; H02 is 123/123 files plus 115/115 embedded future pointers. H01 remains success-terminated historical evidence and H02 remains custom DreamZero s2, not D1.
- Exact clean N3, D1 and RoboLab source identities were verified. N3 checkpoint payloads passed 43/43 byte checks; D1 passed 25/25 and its tokenizer 4/4. The unchanged official D1 wrapper requires two B200 ranks, which is a live scheduling prerequisite rather than a qualification result.
- Implemented the fixed-duration recorder and timeout-only task hook. Its focused tests exercise 450 actual actions after early success, 451 original observation snapshots, exact request/action/future identities, reset evidence, censoring, hash-chained durable journals, and final two-action chunk truncation. This is software qualification only; no real model request or Isaac episode has yet run.
- Implemented deterministic model-blind layout generation, timeout-only fixture tasks, an append-only live rejection/acceptance ledger and a frozen-manifest builder. Numeric candidates remain unreleased until real simulator gates accept them.
- Current verified workshop suite: 106/106 passing. Scientific counts remain generation 0, pilot 0/8, development 0/32 and confirmation 0/192.
- Next concrete action: push the recorder/archive/layout source slice, then release real fixed-observation and model-blind simulator qualification jobs through the independent queue. Do not release behavioral development or confirmation until the corresponding live gates pass.

## 2026-09-13 live qualification milestone

- Corrected the owned worker manifests by removing `NVIDIA_VISIBLE_DEVICES=all`. Live checks show workers 00, 01 and 04 each see exactly one allocated B200; the dedicated D1 worker sees exactly two distinct allocated B200s. Six pre-correction one-GPU diagnostics failed with eight visible GPUs and two were interrupted during the controlled replacement; all receipts remain preserved. They contain zero scientific requests/actions.
- N3 attempt `n3-first-live-001` loaded the official model but failed in workshop metadata serialization before a generation completed. Attempt `n3-first-live-002` repaired only that recorder seam and completed 6/6 real generation requests. Fixed-input repeat, decode/no-decode action+latent equality and prompt sensitivity passed. This qualifies the pinned N3 generation runtime on one hash-bound historical packed observation only; robot episodes remain 0 and physical forecast-time alignment remains unqualified.
- P00 fixture attempts `fixture-p00-candidate-00` and `-r2` failed before starting Isaac with `robolab_checkout_missing` and `pod_uid_environment_missing`. Both are retained technical-invalid attempts with zero model requests and zero behavioral actions. Their objective launcher defects were corrected; `fixture-p00-candidate-00-r3` is staged on corrected worker01.
- Current counts: N3 generation 6/6; D1 generation 0/6; pilot 0/8; development 0/32; confirmation 0/192. No software or infrastructure receipt is counted as a behavioral episode.
- Next concrete action: complete the P00 live gate, freeze its accepted manifest, capture a current hash-bound observation, execute official D1 qualification on the exact two-B200 worker, and run the 450-action recorder-only simulator qualification before model pilots.

## 2026-09-13 P00 live renderer and spare-candidate milestone

- P00 attempts 3 and 4 reached corrected worker01 but remained zero-action technical-invalid setup evidence: the child first lacked the bundled IsaacLab source roots, then followed the RoboLab virtualenv Python symlink to base CPython and lost the virtualenv `toml` package. Both defects are fixed prospectively; all attempts and log hashes remain retained.
- Attempt 5 (`fixture-p00-candidate-00-r5`) completed all 8/8 model-blind reset captures and produced authoritative inner gate record `2e0d2a8c56feb8e975f2babb714c3ca7e0f09fed7eedaa55a79b590d8fddfe5b`. Candidate 00 was physically rejected under the original exact-RTX-byte criterion. No model request or behavioral action occurred.
- The retained evidence showed exact LEFT/RIGHT reset-state equality in 4/4 pairs and exact camera pose/intrinsic/image-size identity in 16/16 paired cameras. Lossless raster hashes differed, with mean absolute channel deltas 0.276–0.796 on uint8 images, overlapping same-command reset jitter. `p00_renderer_nondeterminism.json` records the raw path, hashes and bounded measurements.
- The rejected candidate is not retroactively accepted or rerun. Beginning with candidate 01, matched command qualification uses exact state/proprioception and exact camera-configuration identities as hard gates; lossless observation/RGB hashes remain diagnostics rather than an invented post-hoc pixel threshold. All existing collision, visibility, pose, stability, nonblank and success-false gates remain hard.
- `fixture-p00-candidate-01` is released through the independent queue. The current source also contains the official two-rank D1 launcher and the durable 450-action recorder-only launcher. Current counts remain N3 generation 6/6, D1 generation 0/6, behavioral pilot 0/8, development 0/32, confirmation 0/192.
