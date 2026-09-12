# GM forecast ablation execution — 12 September 2026

**New model episodes: 0 / 232. Generation qualification requests: 0. No GPU job is queued or running from this task.**

The user authorized the attached core specification on GM, with aggressive parallel execution. Optional D2 is not selected. The scientific requirements in `../../docs/ABLATION_SPEC.md` remain binding; machine access does not itself qualify a model, recorder, fixture, or confirmation release.

## Source and access

- Prepared branch: `codex/forecast-layout-gm-20260912`.
- Local isolated source: `/Users/ali-adeeb/Downloads/astra_creative_director/world_models_forecast_run`.
- Workstation isolated source: `/home/ali/projects/steerable-forecast-layout-20260912`.
- Original specification source: `e67e6c4f990030fa8dc9b40a7e5dc713c99457e5`.
- Historical evidence source: `ce561e66f82e95055e39d3d7711691982f6b2086`.
- Both commits were restored to the workstation from a verified full-history bundle. Bundle SHA-256: `a8cc6b8d393902bd4ea8561ec1236097d02c74c88c500b5477da72df576fde69`.
- Workstation is reachable through SSH alias `workstation`; the usual GM route continues through `SZ5VJY@ZTMACMP49KLV4LX.local` and the work Mac's `/usr/local/bin/kubectl`.
- GM context `prod-dcwi-warrenq1-vmkub007`, namespace `211247-prod`, known owned pod `211247-ali-b200-1gpu`, persistent root `/data/users/ali/vla_wam` require fresh live verification.

The work Mac was reached once with forced IPv4. Its GM API request failed with `connect: can't assign requested address`; subsequent SSH attempts timed out. VPN state could not be inspected. See `cluster_preflight.json`. Do not copy GM credentials to another machine or use the workstation's unrelated default Kubernetes context.

## Resume sequence

1. Restore reliable work Mac access and its GM network/VPN connection. Query only the confirmed namespace and ali-owned resources. Verify GPU ownership, free memory, existing process groups and persistent storage before selecting worker count. Historical capacity is not current capacity.
2. Recover an H01 Nano and H02 DreamZero archive episode using the pinned manifests. Validate content hashes. If unavailable, record the exact searched host/path and unresolved asset rather than inventing timing from a comparison video.
3. Restore the exact external source and checkpoint identities into isolated locations. Inspect the runtime source map in `runtime_readiness.md`. Leave dirty existing external checkouts untouched.
4. Implement and qualify the workshop recorder: disable success termination in the task configuration before environment construction; retain success events separately; save original camera RGB, request inputs, returned and executed actions, predictions/latents, physical timestamps, camera preprocessing and context-reset receipts. RoboLab internally freezes terminated environments, so ignoring a returned done flag is insufficient.
5. Run the six fixed-observation qualification requests per primary model, using the official conditional DreamZero route. Preserve every attempt and record actual cost. Verify the generated-frame to physical-time mapping. Never assume generated frame number equals executed action number.
6. Qualify the historical P00 fixture and run up to four pilot episodes per qualified configuration. Then qualify the new layouts without observing target-model outcomes, run the four development blocks per model, and freeze the measured horizons, annotation rubric/threshold, scene poses, seed behavior and resource budget.
7. Release confirmation only once its prerequisites are supported by evidence. Run whole four-condition model/layout blocks on isolated workers. Use the prepared schedule's shared ordering across models. The phase ceilings are two pilot workers, eight development workers and 48 confirmation workers; these are independent-job counts, not a verified GPU allocation.
8. Retain partial and invalid attempts separately, count completed scientific cells once, preserve all valid failures, and obtain the specified independent image labels before reporting forecast accuracy.

No existing V2/V3 registry was activated, rewritten or rerun. No model was replaced. The fixed scientific ceiling is 232 episodes, plus the separately counted qualification requests. The original notebook-like specification is not an executable cluster launcher; recorder and measured qualification work remain.
