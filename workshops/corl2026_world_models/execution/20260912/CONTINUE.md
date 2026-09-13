# GM forecast ablation execution — 12 September 2026

**New model episodes: 0 / 232. Generation qualification requests completed: 6 / 12 (N3 6/6, D1 0/6). N3 runtime qualification passed on a historical fixed observation; physical-time mapping and every behavioral stage remain pending.**

## Current durable execution state

- The cluster-independent queue is operational and handed to this coordinator. `autonomy/independence.json` records the complete workstation -> GitHub -> GM worker -> GitHub -> workstation proof; `autonomy/deployment_receipt.json` records the durable controller and worker deployment.
- Three corrected one-B200 workers and one corrected two-B200 D1 worker are verified running. Another 27 one-B200 workers are Pending because the scheduler currently reports insufficient B200 capacity; they are not counted as available workers. See `autonomy/worker_pool_transition.json` for the preserved isolation failures and corrected identities.
- The work Mac is not in the ongoing dispatch or result-return path. Continue through normal commits on `codex/forecast-layout-gm-20260912` and fetch compact receipts from `codex/forecast-layout-gm-20260912-results`.
- H01/H02 archive pointer closures and the exact N3/D1/RoboLab sources and model payloads are hash-verified in `archive_source_recovery.json`. This is identity/recovery evidence, not a new policy run or frame-time qualification.
- The fixed-duration recorder, timeout-only task hook and model-blind fixture workflow are implemented and locally tested. Their remaining gates are live Isaac/model execution, measured forecast/action/camera mapping, actual resource cost and accepted physical layouts.
- Current verified workshop suite before the active runtime edits: 128/128. N3 completed 6/6 generation qualification requests in `n3-first-live-002`; its first serializer-invalid attempt is preserved. Pilot, development and confirmation scientific counts remain zero.
- P00 live fixture attempts 1 and 2 are preserved as zero-action technical-invalid setup failures. Retry 3 is staged on worker01 after correcting the exact RoboLab checkout and immutable replacement-pod identity handling.

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

1. Reconcile `fixture-p00-candidate-00-r3`. If it is accepted, freeze the P00-only pose manifest and run the current fixed-observation capture; if it is technically invalid or physically rejected, retain it and continue with the next eligible immutable attempt/candidate.
2. Run the six official conditional D1 qualification requests on the exact two-B200 worker after the live capture is hash-bound. Preserve every attempt and record actual cost. N3 already passed 6/6 on the historical fixed input; do not duplicate that sequence.
3. Run the 450-action recorder-only simulator qualification and then connect the qualified model clients. Verify the generated-frame to physical-time mapping from native evidence; never assume generated frame number equals executed action number.
4. Provision the exact policy/simulator GPU separation required by the qualified integrations before the Mac cutoff, then demonstrate real scientific qualification dispatch and result return over the independent queue.
5. Qualify the historical P00 fixture and run up to four pilot episodes per qualified configuration. Then qualify the new layouts without observing target-model outcomes, run the four development blocks per model, and freeze the measured horizons, annotation rubric/threshold, scene poses, seed behavior and resource budget.
6. Release confirmation only once its prerequisites are supported by evidence. Run whole four-condition model/layout blocks on isolated workers. Use the prepared schedule's shared ordering across models. The phase ceilings are two pilot workers, eight development workers and 48 confirmation workers; these are independent-job counts, not a verified GPU allocation.
7. Retain partial and invalid attempts separately, count completed scientific cells once, preserve all valid failures, and obtain the specified independent image labels before reporting forecast accuracy.

No existing V2/V3 registry was activated, rewritten or rerun. No model was replaced. The fixed scientific ceiling is 232 episodes, plus the separately counted qualification requests. The original notebook-like specification is not an executable cluster launcher; recorder and measured qualification work remain.
