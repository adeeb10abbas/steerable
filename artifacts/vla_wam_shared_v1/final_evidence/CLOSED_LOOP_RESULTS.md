# VLA-WAM shared benchmark results

Manifest SHA-256: `eadac021a1b96544daf1300843aa9ec243f7d3bd7f1a7c0814c0c3bc06f23b2c`.

## Closed-loop outcomes

| Tier | Model | Wording | Horizon | Direction | Success | 95% Beta(1,1) | Paper progression | Strict progression | Signed offset |
| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| original_confirmatory | cosmos3_edge_droid_wam | canonical | 32 | left | 7/10 (70%) | [39.0%, 89.1%] | 0.850 | 0.850 | +0.077 m |
| original_confirmatory | cosmos3_edge_droid_wam | canonical | 32 | right | 8/10 (80%) | [48.2%, 94.0%] | 0.900 | 0.900 | +0.347 m |
| post_interim_direct_stress | cosmos3_edge_droid_wam | contrastive_goal | 32 | left | 1/10 (10%) | [2.3%, 41.3%] | 0.550 | 0.550 | -0.098 m |
| post_interim_direct_stress | cosmos3_edge_droid_wam | contrastive_goal | 32 | right | 9/10 (90%) | [58.7%, 97.7%] | 0.950 | 0.950 | +0.318 m |
| post_interim_direct_stress | cosmos3_edge_droid_wam | declarative_goal | 32 | left | 10/10 (100%) | [71.5%, 99.8%] | 0.950 | 0.900 | +0.143 m |
| post_interim_direct_stress | cosmos3_edge_droid_wam | declarative_goal | 32 | right | 9/10 (90%) | [58.7%, 97.7%] | 0.950 | 0.950 | +0.365 m |
| original_confirmatory | cosmos3_edge_droid_wam | short_paraphrase | 32 | left | 4/10 (40%) | [16.7%, 69.2%] | 0.700 | 0.700 | -0.059 m |
| original_confirmatory | cosmos3_edge_droid_wam | short_paraphrase | 32 | right | 10/10 (100%) | [71.5%, 99.8%] | 1.000 | 1.000 | +0.381 m |
| original_confirmatory | pi05_droid_vla | canonical | 15 | left | 0/10 (0%) | [0.2%, 28.5%] | 0.200 | 0.200 | +0.014 m |
| original_confirmatory | pi05_droid_vla | canonical | 15 | right | 8/10 (80%) | [48.2%, 94.0%] | 0.900 | 0.900 | +0.098 m |
| post_interim_direct_stress | pi05_droid_vla | contrastive_goal | 15 | left | 2/10 (20%) | [6.0%, 51.8%] | 0.400 | 0.400 | +0.036 m |
| post_interim_direct_stress | pi05_droid_vla | contrastive_goal | 15 | right | 2/10 (20%) | [6.0%, 51.8%] | 0.500 | 0.500 | +0.010 m |
| post_interim_direct_stress | pi05_droid_vla | declarative_goal | 15 | left | 0/10 (0%) | [0.2%, 28.5%] | 0.250 | 0.250 | +0.024 m |
| post_interim_direct_stress | pi05_droid_vla | declarative_goal | 15 | right | 7/10 (70%) | [39.0%, 89.1%] | 0.850 | 0.850 | +0.138 m |
| original_confirmatory | pi05_droid_vla | short_paraphrase | 15 | left | 1/10 (10%) | [2.3%, 41.3%] | 0.300 | 0.300 | +0.026 m |
| original_confirmatory | pi05_droid_vla | short_paraphrase | 15 | right | 5/10 (50%) | [23.4%, 76.6%] | 0.700 | 0.700 | +0.071 m |

Paper progression is the mean of persistent correct-cube pickup credit and released-object success in the requested location. Relation-only progress is retained in the machine-readable output as a secondary diagnostic. Strict progression additionally requires a post-pick relation transition.
Endpoint relations and signed offsets use rigid-object root poses transformed into the robot frame, matching RoboLab's directional predicate. Rendered bounding-box centroids are used only for the separately labeled visual/settling audit.

## First-action opposite-prompt separation versus same-prompt variation

| Model/condition | Opposite-prompt RMS | Same-prompt seed + renderer RMS | Ratio |
| --- | ---: | ---: | ---: |
| pi05_droid_vla / pi05_canonical_static15 | 0.03487 | 0.10134 | 0.344 |
| pi05_droid_vla / pi05_short_static15 | 0.03468 | 0.11558 | 0.300 |
| cosmos3_edge_droid_wam / cosmos_canonical_static32 | 0.02982 | 0.06710 | 0.444 |
| cosmos3_edge_droid_wam / cosmos_short_static32 | 0.03522 | 0.08607 | 0.409 |
| pi05_droid_vla / pi05_declarative_static15 | 0.03175 | 0.08507 | 0.373 |
| pi05_droid_vla / pi05_contrastive_static15 | 0.00824 | 0.06980 | 0.118 |
| cosmos3_edge_droid_wam / cosmos_declarative_static32 | 0.03932 | 0.06320 | 0.622 |
| cosmos3_edge_droid_wam / cosmos_contrastive_static32 | 0.03242 | 0.06346 | 0.511 |

Realtime rendering was not pixel-repeatable across resets despite one exact physical robot/object reset fingerprint. Full recorder groups differ only because the WAM records two additional camera poses. Opposite-prompt and same-prompt first-action distances therefore both include renderer variation. The ratio is a sensitivity diagnostic, not an isolated causal language effect; the frozen-observation probe supplies that test.

## Guardrail

These estimates apply to the two pinned checkpoints in this shared spatial task. They do not establish a VLA-versus-WAM class difference.
