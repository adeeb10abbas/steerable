# V3-E007 zero-model reachability analysis

This post-result, CPU-only analysis tests whether strict joint-limit pose-IK
volume explains the directional behavioral differences reported in the paper.
It makes no policy/model requests and adds no behavioral episodes.

## Frozen calculation

- 14 paper layouts: reflection control/mirrored, seven reference-sweep levels,
  and five symmetry levels.
- 160 symmetric pose voxels per side and layout (4,480 targets total).
- Exact robot USD kinematic chain, 17 deterministic starts per target, and the
  frozen E002 45-degree relation cone.
- Feasibility threshold: position error at most 1 mm and orientation error at
  most 1 degree, within joint limits.
- Collision, contact, dynamics, and policy state visitation are intentionally
  outside the test.

## Result

All 2,240 targets on each side were feasible: 160/160 per side in every one of
the 14 layouts. The right-minus-left feasible-volume contrast is therefore
exactly zero throughout, even though the observed policy contrast changes with
reflection, reference displacement, and symmetrization.

Basic IK-feasible placement volume does **not** explain the policy-favored
direction under this domain. The paper should claim dependence on scene
configuration without identifying plain reachability as the mediator.

## Evidence

- `registration.json`: frozen inputs, grid, solver, and decision rule.
- `raw/workspace_points.jsonl`: all 4,480 target-level solver records.
- `raw/workspace_summary.json`: layout-level volumes and raw-stream binding.
- `results/results.json`: comparison against the completed policy cohorts.
- `results/v3e007_reachability_mechanism.pdf`: publication figure.
- `results/PAPER_TEXT.md`: two manuscript-ready sentences.
- `results/evidence_manifest.json`: compact result checksums.

The absolute path in `raw/workspace_summary.json` is the retained execution-time
path. The colocated `raw/workspace_points.jsonl` is byte-identical to that bound
stream (SHA-256 `e8d852062c7c30ea0cafb41a1ebbb815daea7f726cf4478a4864b31dd91c79f1`).
