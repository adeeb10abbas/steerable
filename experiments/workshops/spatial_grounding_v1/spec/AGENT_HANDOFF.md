# Start here: SGW-01 spatial grounding study

You are implementing and running the experiment in `EXPERIMENT_SPEC.md` on Ali's authorized Kubernetes resources. Read that document, `WORKSHOP_FIT.md`, and `CLUSTER_RUNBOOK.md` fully. The package contains a planned queue, not a working inference runner or proof that fixtures have qualified.

## Research objective

Measure whether world–action models preserve a physical goal across direct wording, a longer subject-first clause, and an equivalent reference-inverted clause. Test lateral position, relative height, and relative distance. Center the paper on prediction reliability: compare execution with aligned generated futures and persistence, while separating observable failure patterns from claims about internal understanding. Workshop fit: primarily Motion 6; a limited connection to Motion 4.

## Existing facts to retain

- Historical π0.5 inversion is background only: keep it out of the main WAM results. A fresh matched π0.5 LAT baseline is optional, not queued, and not a causal world-model ablation.
- Nano/Edge already show exploratory wording effects. Do not call these reference-inversion replications.
- DreamZero historical s2 guidance is custom. The new D1 branch uses the qualified official conditional path.
- The previous stable-grasp construction failed; zero stage-localization model episodes exist.
- The September 12 232-episode forecast-only matrix is historical planning. Do not combine it with SGW-01 or terminate a different agent's running job.

## Work allocation for a team

1. **Coordinator:** verify current repository and cluster ownership, freeze the new namespace and resource binding, track receipts/queue/status. Own released manifests and avoid overlapping writer ownership.
2. **Fixture implementer:** create model-blind LAT/DIST/HEIGHT candidates and acceptance receipts. Exact poses must be tested, not invented. Never look at confirmation model outcomes to choose scenes.
3. **Runtime/recorder implementer:** adapt pinned Nano and official DreamZero, implement full resets, physical-time maps, persistent queue execution, retry and recovery behavior. Own the new worker package and its tests.
4. **Analysis/reviewer:** validate goal semantics, scorer cases, synthetic recovery/missingness cases and bootstrap units. Freeze analysis before confirmation; validate independent prediction annotations and final regenerated tables.

These are roles for the receiving team, not agents already dispatched by this handoff. If only one agent is available, perform them sequentially. Use isolated branches/worktrees for concurrent code edits; only the coordinator publishes the integrated release.

## Execute in order

- [ ] Read current repository instructions and historical continuation state; preserve existing workloads and all V2/V3 evidence.
- [ ] Reproduce `build_registry.py`; require 18 prompts, 1,044 unique cells, 174 six-cell blocks, 36/144/864 stage counts, and a valid package receipt.
- [ ] Implement the package and CLI specified in the runbook. Commands naming that package are not executable until it exists and passes its checks.
- [ ] Resolve cluster/PVC/image/runtime/budget from live user-owned resources; write `runtime_binding.json`.
- [ ] Qualify fixtures without model outcomes, then recording pilot P. If one family is blocked, continue independent qualified families.
- [ ] Complete development D; freeze fixtures, cameras, time mappings, deadlines, scoring, release hashes and annotation rules.
- [ ] Render and verify concrete Kubernetes Jobs. Test crash/restart, duplicate worker, missing artifact and exhausted retry behavior.
- [ ] Release confirmation C and keep consuming the finite queue. Do not ask for permission for every next valid cell. Stop only at a documented blocking condition or scope/budget boundary.
- [ ] Record failures as results. Keep infrastructure invalidity separate and retry at most twice after the original attempt.
- [ ] Finish with validated raw manifests, compact results, paper tables/figures, media selection rules, and `STATUS.md`. Clearly identify branches that did not run.

## Acceptance at handoff back to Ali

Report exactly: which model/family branches qualified; valid episodes versus planned; any partial/invalid attempts and censored trials; verified PVC location; Job/pod IDs; source/runtime/release hashes; prediction coverage; figures and estimates; and the exact next action if blocked. Never say “running” merely because a pod exists or a background process was started. Verify recent request progress and durable artifacts.

Do not publish, submit, contact collaborators, change project sharing, add training jobs, or add models/conditions. Updates to the designated Overleaf draft should distinguish existing evidence, planned analyses, and new validated findings.
