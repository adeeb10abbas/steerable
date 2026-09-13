# GM forecast ablation execution — 12 September 2026

**New valid model episodes: 4 / 232, all from the completed N3 P00 pilot. Generation qualification requests completed: 12 / 12 (N3 6/6, D1 6/6). N3 pilot attempt 004 passed 4/4 cells with 1,800 actions, 1,804 observations and 60 behavioral model requests. D1 behavioral pilot, all 32 development cells and all 192 confirmation cells remain unrun.**

## Current durable execution state

- The cluster-independent queue is operational and handed to this coordinator. `autonomy/independence.json` records the complete workstation -> GitHub -> GM worker -> GitHub -> workstation proof; `autonomy/deployment_receipt.json` records the durable controller and worker deployment.
- The active queue topology has four Running one-B200 workers, one isolated two-B200 N3 worker and one isolated two-B200 D1 worker. Worker00 and both model workers have exact prior identity receipts. Newly admitted workers05, 06 and 09 must pass the fresh hash-bound pod/GPU diagnostic before scientific work; a Running pod alone is not accepted capacity. Worker00 is reserved for the D1 simulator lane.
- The work Mac is not in the ongoing dispatch or result-return path. Continue through normal commits on `codex/forecast-layout-gm-20260912` and fetch compact receipts from `codex/forecast-layout-gm-20260912-results`.
- H01/H02 archive pointer closures and the exact N3/D1/RoboLab sources and model payloads are hash-verified in `archive_source_recovery.json`. This is identity/recovery evidence, not a new policy run or frame-time qualification.
- The live fixed-duration recorder passed in `recorder-qualification-p00-003`: 450 actions, 451 observations, zero model requests and zero behavioral episodes. Receipt SHA-256: `0e3f02f37a2548e36ae3a45a38a1fac63c56cd8798732056f24b03d103991fde`. Native camera, physics and control counters were recorded; this remains recorder-only evidence.
- Both primary generation runtimes are qualified. N3 completed 6/6 requests in `n3-first-live-002`; D1 attempt 005 completed 6/6 official conditional-path requests on two B200s after four preserved pre-request technical-invalid attempts. D1 receipt SHA-256: `3c856549999b9145dc07c30853a4c6d2968d09eb31c2db883f5eb1655d31627b`. Neither qualification is a robot episode or forecast-time mapping result.
- Current full verified workshop suite remains 175/175; the newly added N3 development plus stable-pilot focused suite passes 30/30. N3 pilot attempts 001 and 002 remain technical-invalid at zero actions and zero requests. Attempt 003 reached one real behavioral generation but the no-replay guard rejected a keepalive-induced duplicate; it executed zero actions. Attempt 004 passed all four cells with 450 actions, 451 observations and 15 requests per cell. Its aggregate receipt SHA-256 is `5a7a861acc82615b4eb7aa92308f40f7a0c965e65b6edaf12f776cb64b18b0e0` on results commit `06b8ab02b4a382a23372dfef327915121f1b2069`. This is four valid learned-policy episodes, not a task-success or forecast-accuracy claim.
- P00 attempts 1–4 are preserved as zero-action technical-invalid setup failures. Attempt 5 completed all eight model-blind reset captures; candidate 00 remains rejected under the original exact-RTX-byte rule. Candidate 01 attempt 000 is preserved technical-invalid due a deterministic producer/validator JSON-hash mismatch, not physical drift. The fresh r2 attempt passed all eight captures under the corrected shared hash contract with no tolerance change. Accepted gate receipt SHA-256: `9611b5bd1fee98ebc7f480e7044a871e54894bbb63ce2e91966ca46d96a6adbf`; frozen pose SHA-256: `c597bbafdaab3155b35945f8e6130e599fa3c8dc1c9d83f247611aee2bbd0954`.
- D01–D04 candidate 00 each passed the same model-blind live fixture gate. Their accepted receipts are retained on the results branch and PVC; their single-layout frozen pose/capture manifests are the next prerequisite before N3 development inference. C01–C07 candidate-00 attempts are preserved as zero-action, zero-request technical-invalid evidence because worker09 was not represented in the earlier static deployment receipt. C08–C24 were deactivated before execution. All confirmation fixture attempts will use new IDs only after a current dynamic worker identity receipt passes; no confirmation inference has been released.

The user authorized the attached core specification on GM, with aggressive parallel execution. Optional D2 is not selected. The scientific requirements in `../../docs/ABLATION_SPEC.md` remain binding; machine access does not itself qualify a model, recorder, fixture, or confirmation release.

## Source and access

- Prepared branch: `codex/forecast-layout-gm-20260912`.
- Local isolated source: `/Users/ali-adeeb/Downloads/astra_creative_director/world_models_forecast_run`.
- Workstation isolated source: `/home/ali/projects/steerable-forecast-layout-20260912`.
- Original specification source: `e67e6c4f990030fa8dc9b40a7e5dc713c99457e5`.
- Historical evidence source: `ce561e66f82e95055e39d3d7711691982f6b2086`.
- Both commits were restored to the workstation from a verified full-history bundle. Bundle SHA-256: `a8cc6b8d393902bd4ea8561ec1236097d02c74c88c500b5477da72df576fde69`.
- Direct work-Mac/Kubernetes access remains an initial setup and cleanup route only. Ongoing source dispatch and compact result return use the verified Git/PVC queue without the Mac.
- GM context `prod-dcwi-warrenq1-vmkub007`, namespace `211247-prod`, persistent root `/data/users/ali/vla_wam`; never use the workstation's unrelated default Kubernetes context or copy GM credentials.

The earlier network failure in `cluster_preflight.json` is historical. The subsequent independent queue proof supersedes it; preserve that failed attempt rather than rewriting it as a success.

## Resume sequence

1. Finish and release the D1 P00 behavioral pilot using the dedicated two-B200 official server and separate worker00 one-B200 simulator client. Never interleave contexts on the global-state DreamZero server; preserve all four conditions as one ordered block and use no-ping/no-replay transport semantics.
2. Publish current hash-bound diagnostics for newly admitted one-B200 workers. Retry C01–C24 fixture gates only under fresh immutable IDs after the selected worker diagnostic passes. Preserve C01–C07 as objective launcher-provenance failures.
3. Run the generalized fixed-observation freezer for D01–D04, publish each accepted pose/camera/input manifest, and release the four N3 development blocks from source `c46bb8b3dcfbc95bbf14684e5d87db2730fd077a` using the passed N3 pilot receipt. Implement/release the corresponding D1 development runner only after the D1 pilot passes.
4. From pilot/development native clocks, freeze the generated-frame to physical-time mapping, eligible horizons, camera tolerance, actual resource budget, request-selection algorithm/seed and blind two-rater rubric. Never infer time from generated frame number or concatenated presentation video.
5. Build the confirmation C01–C24 model-blind candidates and accepted pose manifests without inspecting target-model outcomes. Release all 24 base-layout pairs only after the development/annotation freeze; use the same assigned four-condition order for both models.
6. Retain partial and invalid attempts separately, count completed scientific cells once, preserve all valid failures, and obtain the specified independent image labels before reporting forecast accuracy. The phase ceilings are two pilot workers, eight development workers and 48 confirmation workers; worker count is not sample size.

No existing V2/V3 registry was activated, rewritten or rerun. No model was replaced. The fixed scientific ceiling is 232 episodes, plus the separately counted qualification requests. The original notebook-like specification is not an executable cluster launcher; recorder and measured qualification work remain.
