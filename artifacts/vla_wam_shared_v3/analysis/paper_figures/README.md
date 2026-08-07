# V3 paper figures

This directory holds the shared publication-facing figures generated only from
hash-closed evidence. Figures remain numbered according to the registered paper
outline. Experiment-specific figures stay beside their evidence bundle so the
source, renderer, and output remain co-located.

## Figure 1: instrument sensitivity

`figure1_nano_instrument_sensitivity.{svg,png}` contrasts three near-ceiling
binary gates with the full-sample requested-side depth diagnostic for Cosmos3
Nano. Its 27 paired depth differences include behavioral failures; the figure
does not replace the frozen task predicate with a continuous score.

Regenerate with:

```bash
tmp/pdfs/.venv/bin/python tools/render_v3_nano_instrument.py
```

## Figure 2: three-checkpoint position reflection

`figure2_three_checkpoint_position_reflection.{svg,png}` is produced only after
Nano, π0.5, and DreamZero each have a closed 27-seed × four-cell reflection
result. It shows every seed's requested-side-depth interaction and binary
success difference-in-differences. The renderer refuses missing DreamZero
evidence; it never substitutes or duplicates another checkpoint.

Regenerate with:

```bash
tmp/pdfs/.venv/bin/python tools/render_v3_mirror_core.py
```

## Figures stored with their experiment or analysis

| Figure | Evidence | Publication PNG |
|---|---|---|
| 3 | Nano seven-level lateral dose-response | `artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results/figures/figure3_nano_lateral_dose_response.png` |
| 4 | Five-checkpoint gap versus competence | `artifacts/vla_wam_shared_v3/analysis/mechanism/figures/figure4_gap_vs_competence.png` |
| 6 | Direction-stratified failure taxonomy | `artifacts/vla_wam_shared_v3/analysis/mechanism/figures/figure6_failure_taxonomy_by_direction.png` |
| 7 | Three-checkpoint phrasing × direction | `artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001/results/figures/figure7_phase_c_phrasing_direction.png` |

## Figure 5: cross-arena directional success

`figure5_cross_arena_directional_success.{svg,png}` shows LEFT and RIGHT
direction-specific success with 95% Wilson intervals for five DROID/RoboLab and
three RoboTwin checkpoints. The panels use separate axes and denominators;
their success rates are never pooled. Counts appear directly beside each
interval, and the exact DROID prompt pair is printed in the figure.

Regenerate with:

```bash
tmp/pdfs/.venv/bin/python tools/render_v3_cross_arena_success.py
```

The adjacent manifest records every source, output, and renderer SHA-256.

## Scientific PDF

The compact landscape report uses Figures 1–7 in paper order and places each
plot beside its finding, interpretation, and claim boundary. It fails closed
until Figure 2 and all three 160-episode Phase-C summaries are complete:

```bash
tmp/pdfs/.venv/bin/python tools/build_v3_scientific_report.py
```

Validate the publication figures with:

```bash
tmp/pdfs/.venv/bin/python tools/validate_v3_paper_figures.py
```
