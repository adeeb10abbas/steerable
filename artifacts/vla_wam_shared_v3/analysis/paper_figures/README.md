# V3 paper figures

This directory holds compact, publication-facing figures generated only from
hash-closed evidence. Figures remain numbered according to the registered paper
outline; absent numbers depend on still-running prospective experiments.

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
