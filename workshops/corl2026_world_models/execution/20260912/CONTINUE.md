# GM forecast ablation execution — 12 September 2026

**New model episodes: 0 / 232. Generation qualification requests: 0 / 12. Four infrastructure-only GPU diagnostics passed; no scientific GPU job has yet been released.**

## Current durable execution state

- The cluster-independent queue is operational and handed to this coordinator. `autonomy/independence.json` records the complete workstation -> GitHub -> GM worker -> GitHub -> workstation proof; `autonomy/deployment_receipt.json` records the durable controller and worker deployment.
- Four one-B200 workers are verified running. Another 28 pre-created workers are Pending because the scheduler currently reports insufficient B200 capacity; they are not counted as available workers.
- The work Mac is not in the ongoing dispatch or result-return path. Continue through normal commits on `codex/forecast-layout-gm-20260912` and fetch compact receipts from `codex/forecast-layout-gm-20260912-results`.
- H01/H02 archive pointer closures and the exact N3/D1/RoboLab sources and model payloads are hash-verified in `archive_source_recovery.json`. This is identity/recovery evidence, not a new policy run or frame-time qualification.
- The fixed-duration recorder, timeout-only task hook and model-blind fixture workflow are implemented and locally tested. Their remaining gates are live Isaac/model execution, measured forecast/action/camera mapping, actual resource cost and accepted physical layouts.
- Current verified workshop suite: 106/106. Generation, pilot, development and confirmation scientific counts all remain zero.

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

1. Push the implemented recorder/archive/layout source slice and use that immutable source commit for queue jobs. Preserve exact attempt IDs and never mutate an existing descriptor.
2. Run the six fixed-observation qualification requests per primary model, using the official conditional DreamZero route. Preserve every attempt and record actual cost. Verify the generated-frame to physical-time mapping. Never assume generated frame number equals executed action number.
3. Run real model-blind fixture gates and freeze accepted/rejected candidates. Connect the task's settled reset, native clock/state/success callbacks and original observations to the recorder.
4. Provision the exact policy/simulator GPU separation required by the qualified integrations before the Mac cutoff, then demonstrate real scientific qualification dispatch and result return over the independent queue.
5. Qualify the historical P00 fixture and run up to four pilot episodes per qualified configuration. Then qualify the new layouts without observing target-model outcomes, run the four development blocks per model, and freeze the measured horizons, annotation rubric/threshold, scene poses, seed behavior and resource budget.
6. Release confirmation only once its prerequisites are supported by evidence. Run whole four-condition model/layout blocks on isolated workers. Use the prepared schedule's shared ordering across models. The phase ceilings are two pilot workers, eight development workers and 48 confirmation workers; these are independent-job counts, not a verified GPU allocation.
7. Retain partial and invalid attempts separately, count completed scientific cells once, preserve all valid failures, and obtain the specified independent image labels before reporting forecast accuracy.

No existing V2/V3 registry was activated, rewritten or rerun. No model was replaced. The fixed scientific ceiling is 232 episodes, plus the separately counted qualification requests. The original notebook-like specification is not an executable cluster launcher; recorder and measured qualification work remain.
