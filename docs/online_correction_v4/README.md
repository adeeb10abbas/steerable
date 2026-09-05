# Online spatial correction: V4 agent handoff

**Start here.** This is a prospective experiment and analysis package for extending the completed static spatial-language study. No V4 robot experiments are reported as complete. The allocation is **17,664 new policy episodes**, **240 excluded engineering policy pilots**, and separately logged model-blind feasibility checks. Reused C1/C3 controls are not rerun or counted twice.

The main question is whether a policy corrects its ongoing manipulation according to the **requested relation and named reference** when a scene object moves. We keep a small historical static foundation and add controlled online interventions. We do not change model weights or switch instructions during an episode.

## Read in this order

| File | Purpose |
| --- | --- |
| [00_EXISTING_EVIDENCE_AND_SCOPE.md](00_EXISTING_EVIDENCE_AND_SCOPE.md) | What is already established, reusable, confounded, blocked, or unavailable |
| [01_EXPERIMENTS.md](01_EXPERIMENTS.md) | Three research questions, exact C1–C8 conditions, prompts, physical interventions, timing, seeds, gates, and limits |
| [02_METRICS_AND_ANALYSIS.md](02_METRICS_AND_ANALYSIS.md) | Goal-region geometry, success, reference selectivity, raw logs, denominator rules, uncertainty, and fixed primary contrasts |
| [03_AGENT_EXECUTION_RUNBOOK.md](03_AGENT_EXECUTION_RUNBOOK.md) | Repository integration, implementation tasks, cluster ownership, release, retries, storage, and closeout |
| [04_PAPER_AND_RELEASE.md](04_PAPER_AND_RELEASE.md) | Required tables/figures, paper claims, reviewer-feedback mapping, and final evidence release |
| [05_DESIGN_REVIEW.md](05_DESIGN_REVIEW.md) | Independent design review and resolved risks |
| [campaign.json](campaign.json) | Machine-readable allocation, defaults, policies, counterbalances, and analysis registry |
| [runtime_lock.template.json](runtime_lock.template.json) | Intentionally unreleased template for the real runner, checkpoint, geometry, scoring, and qualification receipts |

The helper [../../tools/online_correction_v4.py](../../tools/online_correction_v4.py) validates the allocation and creates a planning manifest. **It is not a simulator runner, launcher, scorer, or completed statistical analysis implementation.** Agents must implement and qualify those pieces from the specification before releasing policy inference.

## First commands, from the repository root

```bash
python tools/online_correction_v4.py validate
python tools/online_correction_v4.py manifest --out /persistent/v4/planned_episodes.jsonl
python tools/online_correction_v4.py release-check --lock /persistent/v4/runtime_lock.json --manifest /persistent/v4/planned_episodes.jsonl
```

The template is deliberately expected to fail `release-check`. Complete the implementation and evidence receipts, not just the JSON strings. The helper checks structure, hashes, allocation, and dependencies; it cannot inspect remote videos, establish physical feasibility, or prove a receipt's scientific correctness. The qualification agent must review the underlying evidence.

After execution:

```bash
python tools/online_correction_v4.py check-results --manifest /persistent/v4/planned_episodes.jsonl --results /persistent/v4/results.jsonl
python -m unittest discover -s tests -p 'test_online_correction_v4.py'
```

`/persistent/v4` is an illustrative persistent-storage mount. Bind the actual mounted path in the runtime lock; do not write large robot traces to an ephemeral pod filesystem or ordinary Git.

## Give this instruction to the coordinating agent

> Read the V4 README and all six numbered documents, then the repository's historical source protocols and current cluster handoff. Implement the missing V4 runner, scheduler, fixtures, scorer, and analysis without changing the frozen V2/V3 evidence. Restore the exact selected checkpoints. Generate the complete V4 planning manifest; run the model-blind feasibility checks and excluded technical pilots; bind all runtime parameters and receipts; then release qualified families automatically and run every registered cell on isolated cluster lanes. Preserve valid failures and no-event outcomes, retry only infrastructure-invalid attempts, reuse the specified controls, and retain complete video/state/action/timing records. Run the fixed analysis and export every required paper table/figure, including null results and missing scope. Commit compact code, configuration, manifests, hashes, and reports; keep raw arrays and checkpoints on persistent experiment storage. Do not add conditions, choose checkpoints based on performance, tune prompts to pilot success, or launch unqualified families to fill an allocation.

## What is fixed, and what agents must still resolve

**Fixed now:** the questions, checkpoint roles, eight family allocations, relation and wording inventory, counterbalance recipe, motion-profile definitions and nominal scales, controlled clock, first-placement endpoint, primary outcomes and contrasts, fixed sample size, exclusion rules, and paper outputs.

**Resolve before launch:** exact artifact access/identity, actual simulator geometry and object names, native tick quantization, scripted reachability, collision and visibility checks, trigger/release detector validation, supported goal predicates and tolerances, raw-trace integrity, deterministic replay or a declared alternative, cluster/runtime qualification, and executable analysis implementation. These are explicit engineering gates. The package does not claim the current repository already supplies a valid shelf task, Bridge adapter, or V4 online runner.

The coordinator can release an independently qualified subset while preserving blocked families in the original inventory. It must disclose the lost scope; C3/C4 cannot ignore their control dependencies. No success-rate threshold or positive language effect is a technical release requirement. Poor behavior is data when the implementation is valid.

## Verification boundary

Package-level checks cover allocation arithmetic, deterministic identifiers/seeds, control reuse, release-lock rejection, and accepted-result ledger integrity. Physical experiments, robot feasibility, checkpoint access, cluster throughput, and empirical findings remain **TODO** until the execution agents produce their evidence. See the design review for the distinction between a reviewed plan and a validated experimental system.

Package verification is recorded in [design_validation.json](design_validation.json): 13 helper tests passed, the complete 17,664-row inventory was generated, control reuse was verified, and the unreleased runtime template was rejected as intended. These are design/software checks; no policy experiments were launched.
