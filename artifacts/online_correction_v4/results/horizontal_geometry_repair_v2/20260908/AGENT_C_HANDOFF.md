# Agent C handoff — horizontal geometry repair v2 close-out

**Slice:** `artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908/`

## Start here

1. `evidence_manifest.json` — SHA256-indexed receipt catalog
2. `blocked_scope.json` — blocked families, gate status, quantified squeeze finding
3. `paper/evidence_memo.json` — paper-ready narrative paragraphs and numbers

## Gate status

| Gate | Attempt | Status |
|------|---------|--------|
| G2 | g2r20260908g | **PASSED** — 128/128 seeds, axis review passed |
| G3 | g3r20260908g | **FAILED (scientific)** — information gate at scale 0.5 |
| G5 registry | horizng20260908* | Negative pass; positive blocked (Agent B) |

## Key scientific claim

Geometry repair restored physical feasibility (3072/3072 path checks pass), but **no registered scale** satisfies both large-displacement goal non-emptiness and the 20% information threshold at small displacement. At scale 0.5, exactly **64/128** seeds fail (all `sign=-1`) with removed-area **13.7–19.9%** vs 20% threshold.

## Do not

- Amend criteria or re-open the scale ladder
- Dispatch C1/C3/C4 policy episodes (9,728 blocked)
