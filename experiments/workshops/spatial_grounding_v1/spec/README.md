# SGW-01: Equivalent spatial instructions in world–action models

**Status: experiment specification, 22 September 2026. No SGW-01 cluster jobs or model experiments have been launched by this task.**

The question is whether a world–action model's generated futures remain a reliable account of its executed motion when spatial instructions change. The study compares direct wording, a longer subject-first clause, and an equivalent clause with the relational arguments reversed.

See [WORKSHOP_FIT.md](WORKSHOP_FIT.md) for the workshop framing. The core stays WAM-only; historical π0.5 results are background and a fresh matched baseline is optional, not queued.

## Give agents this package

Start with [AGENT_HANDOFF.md](AGENT_HANDOFF.md), then read the [experiment specification](EXPERIMENT_SPEC.md) and [Kubernetes runbook](CLUSTER_RUNBOOK.md). The worker still needs implementation and qualification. The Kubernetes file is a template with unresolved fields, not a runnable deployment.

| Planned experiment | Scope |
| --- | --- |
| Models | Cosmos3 Nano g3 and official conditional-action DreamZero |
| Relations | Left/right; higher/lower; closer/farther relative to two anchors |
| Wording | Three forms × two opposite physical goals |
| Layouts per relation | 1 pilot + 4 development + 24 confirmation |
| Episodes | 36 pilot + 144 development + 864 confirmation = **1,044** |
| New results currently available | **None** |

The independent sample is the layout, not a frame, request, or repeated condition. Each relation family must have physically valid fixtures. Completed model failures remain data and are never retried to improve the result.

## Contents

- `WORKSHOP_FIT.md`: target contribution, π0.5 decision, optional baseline and submission shape.
- `EXPERIMENT_SPEC.md`: existing evidence, exact ablations, physical scoring, forecast alignment, analyses, and interpretation.
- `CLUSTER_RUNBOOK.md`: storage, ownership, worker interfaces, bounded retries, restart recovery, and acceptance checks.
- `AGENT_HANDOFF.md`: receiving-team instructions and completion checklist.
- `prompts.json`: 18 exact instructions and hashes.
- `protocol.json`: fixed scientific settings and unresolved runtime prerequisites.
- `planned_cells.csv`: 1,044 uniquely identified planned cells; all remain unreleased.
- `registry_validation.json`: planning consistency checks; not proof of runtime qualification.
- `build_registry.py`: rebuilds the planning registry using Python's standard library. It never contacts a cluster or model.
- `kubernetes/worker-job.yaml.in`: partition Job template, to be bound to verified existing resources.
- `paper/main.tex` and `paper/research_plan.pdf`: editable and compiled research plan, with the exact prompt appendix.
- `PACKAGE_MANIFEST.json`: hashes for the packaged handoff files.

Run `python3 build_registry.py` from this directory to regenerate the planning files. Implementing agents must separately run the acceptance checks in the runbook before releasing confirmation. Do not treat a regenerated planning receipt as a launch receipt.

The working document is in [Overleaf](https://www.overleaf.com/project/6ab2bdb39b30df4bbc98a15e). This detailed research plan is longer than the final workshop submission. It contains no invented new results.

The September 12 forecast-only specification remains historical planning. This package does not modify its evidence, stop another agent's workload, or combine its proposed sample with SGW-01.
