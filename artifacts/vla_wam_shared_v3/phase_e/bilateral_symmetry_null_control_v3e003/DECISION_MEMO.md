# V3-E003 decision memo — bilateral-symmetry null control

## Completed cohort

π₀.₅ current stack; symmetric movable-object layout; 27 matched seeds × two
static prompts = **54/54 valid behavioral episodes**. There were 7 retained
infrastructure-invalid attempts and two duplicate valid artifacts from the
pre-fix retry race; duplicates are excluded by registered cell ID. The bowl
and cube were on the robot midline and the mirrored clutter-pair residual was
0.0 m for every retained episode. This is symmetric object layout, not a
bilaterally symmetric robot or embodiment: the joint configuration and
wrist-camera mounting remain asymmetric.

## Results

| estimand | result |
|---|---|
| LEFT success | 13/27 = 48.1% (Wilson 95% CI 30.7–66.0%) |
| RIGHT success | 25/27 = 92.6% (Wilson 95% CI 76.6–97.9%) |
| paired success discordance | LEFT fail/RIGHT success 13; LEFT success/RIGHT fail 1; exact McNemar p = 0.00183 |
| success gap (RIGHT−LEFT) | 12/27 = 44.4 percentage points; 20,000-resample CI 22.2–66.7 pp |
| requested-side depth (RIGHT−LEFT) | mean +6.18 cm (CI +3.29 to +9.15 cm); median +6.74 cm (CI +4.71 to +8.56 cm); sign test 22+/5−, p = 0.00151 |
| endpoint shift (RIGHT−LEFT) | mean −23.77 cm (CI −27.54 to −20.00 cm); equivalently LEFT−RIGHT = +23.77 cm, following +Y = robot-left |

Failure taxonomy was directionally concentrated: LEFT had 11 wrong-side, 2
release, and 1 transport failure; RIGHT had 2 wrong-side failures. No pick
failures occurred. The registered equivalence margins were |binary gap| <
4/27 (0.148) and |depth contrast| < 5 cm. The success-gap interval excludes
the binary equivalence margin; the depth interval does not lie wholly within
the 5 cm margin. Thus H1 and H2 are not supported as null/equivalence
findings. H3 is supported: endpoint redirection remains large and consistently
oriented.

## Interpretation and claim boundary

The symmetric object layout did not remove the directional performance gap.
Within this fixed embodiment and controller, π₀.₅ still completed the RIGHT
prompt substantially more often than the LEFT prompt, while endpoint motion
continued to redirect in the requested ordering. This is evidence for a
non-geometric component of the asymmetry, but it does not identify whether the
residual comes from embodiment, camera placement, policy data distribution,
controller calibration, or their interaction. The experiment does not justify
an equivalence claim, a symmetric-robot claim, or a population-level statement
about π₀.₅.

## Manuscript replacement text

“In a layout with the bowl and target centered on the robot midline and all
clutter mirrored, π₀.₅ still succeeded more often for the exact RIGHT prompt
(25/27, 92.6%) than for the exact LEFT prompt (13/27, 48.1%; paired McNemar
p=0.00183). The requested-side depth contrast was +6.18 cm (RIGHT−LEFT),
with a 95% bootstrap interval of +3.29 to +9.15 cm. Meanwhile the endpoint
shift remained strongly directional (LEFT−RIGHT = +23.77 cm in robot Y).
The symmetric-object control therefore rejects the preregistered small-gap
interpretation for this checkpoint: language changes the physical response,
but task completion remains strongly asymmetric even after object-layout
symmetry is imposed.”

## Recommendation

Include this as a primary supplementary control and cite the exact prompt,
paired design, failure taxonomy, and endpoint metric in the main paper. Do not
describe it as proof of equivalence or as isolation of a single causal source.

## Evidence

Machine-readable results are in `results.json`; raw JSONL, videos, state
captures, and action traces remain on the PVC. The evidence manifest binds the
registration, candidate, runtime/release gates, compiler, and all raw source
hashes.
