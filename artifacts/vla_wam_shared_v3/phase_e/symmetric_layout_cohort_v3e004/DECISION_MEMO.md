# V3-E004 decision memo

Status: **complete_hash_closed**.

## Evidence boundary

Valid behavioral evidence: **4096/4096** registered cells. Infrastructure-invalid attempts: **627**, excluded from behavioral denominators.
Discovery-only behavioral artifacts: **79**, excluded from behavioral denominators.
- Pre-R002 DROID s=0 artifacts without prospective R002 attestation: **72**.
- Pre-R001 DROID artifacts without fixed-observation pair identity: **7**.

This experiment manipulates object-layout symmetry. It does not make the robot, reset posture, camera rig, wrist mounting, or embodiment bilaterally symmetric. DROID/RoboLab and RoboTwin remain separate and are never pooled.

## Geometry and visibility quality control

The four registered scene checks are position residual, mirrored-orientation residual, midline residual, and the per-camera occlusion check. The complete per-episode values remain in `results/episodes.jsonl`; the maxima below summarize only currently valid s=1 episodes.

| Checkpoint | Valid s=1 episodes | Max position residual, mm | Max orientation residual, deg | Max midline residual, mm | Occluded camera checks | Reset-pose identities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cosmos3_edge_policy_droid | 54 | 0.058 | 0.094 | 0.235 | 0 | 1 |
| cosmos3_nano_policy_droid | 1042 | 0.058 | 0.096 | 0.236 | 0 | 1 |
| dreamzero_droid_action_cfg | 54 | 0.058 | 0.096 | 0.236 | 0 | 1 |
| fastwam_robotwin | 54 | 0.000 | 0.008 | 0.002 | 0 | 1 |
| pi05_current_stack_droid | 682 | 0.058 | 0.096 | 0.236 | 0 | 1 |

### Recorded arm reset poses

Each identity below hashes the complete recorded reset-pose object, including its measurement provenance. Multiple identities are retained rather than averaged away; if more than three occur, this memo shows the three most frequent and `results.json` retains the complete list.

- **cosmos3_edge_policy_droid**: `938c2ae19d8521e74d52dacff9698d5be31f09fd76577cb9de532c048ffcf3ef` across 108 episodes; arm q (7 joints) = [+0.0000, -0.6283, +0.0000, -2.5133, +0.0000, +1.8850, +0.0000] rad; gripper = [0.0].
- **cosmos3_nano_policy_droid**: `938c2ae19d8521e74d52dacff9698d5be31f09fd76577cb9de532c048ffcf3ef` across 2246 episodes; arm q (7 joints) = [+0.0000, -0.6283, +0.0000, -2.5133, +0.0000, +1.8850, +0.0000] rad; gripper = [0.0].
- **dreamzero_droid_action_cfg**: `938c2ae19d8521e74d52dacff9698d5be31f09fd76577cb9de532c048ffcf3ef` across 108 episodes; arm q (7 joints) = [+0.0000, -0.6283, +0.0000, -2.5133, +0.0000, +1.8850, +0.0000] rad; gripper = [0.0].
- **fastwam_robotwin**: `0d7df5f25521331361d09507de0179486ff079926123b6d00fecda277675b425` across 108 episodes; robot.left: 38 joints, ||q||₂=0.088475 rad, range=[-0.000008, +0.044238] rad; robot.right: 38 joints, ||q||₂=0.088475 rad, range=[-0.000008, +0.044238] rad; exact vectors retained in results.json.
- **pi05_current_stack_droid**: `938c2ae19d8521e74d52dacff9698d5be31f09fd76577cb9de532c048ffcf3ef` across 1526 episodes; arm q (7 joints) = [+0.0000, -0.6283, +0.0000, -2.5133, +0.0000, +1.8850, +0.0000] rad; gripper = [0.0].

Passing these object-layout checks does not establish bilateral robot or embodiment symmetry.

## Registered estimands

| Checkpoint | Binary interaction (s1−s0) | Depth interaction, m (s1−s0) | Endpoint positive control at all levels | Equivalence claims |
| --- | ---: | ---: | --- | --- |
| cosmos3_edge_policy_droid | -0.148 [-0.407, +0.074] | -0.192 [-0.260, -0.115] | pass | none |
| cosmos3_nano_policy_droid | +0.000 [-0.111, +0.111] | -0.139 [-0.203, -0.077] | pass | none |
| dreamzero_droid_action_cfg | -0.556 [-0.815, -0.296] | -0.124 [-0.173, -0.075] | pass | none |
| fastwam_robotwin | +0.037 [-0.074, +0.148] | -0.382 [-0.445, -0.320] | fail closed | none |
| pi05_current_stack_droid | -0.519 [-0.815, -0.222] | -0.124 [-0.197, -0.049] | pass | none |

## H2 — power and equivalence audit

The achieved MDE is the preregistered 80%-power design MDE evaluated at the valid s=1 pair count. Equivalence is authorized only when the registered power status permits it and the paired 90% interval lies wholly inside the registered margin; a nonsignificant difference is never treated as equivalence.

| Checkpoint / estimand | Control effect | s=1 estimate | Margin | Achieved MDE (n) | Paired 90% CI | TOST bootstrap p (lower / upper) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cosmos3_edge_policy_droid / binary R−L | +0.259 | +0.037 | 0.052 | 0.284 (27) | [-0.074, +0.148] | 0.06365 / 0.3748 | no_equivalence_claim (underpowered_no_equivalence_claim) |
| cosmos3_edge_policy_droid / depth R−L, m | +0.237 | +0.072 | 0.047 | 0.088 (27) | [+0.033, +0.111] | 0 / 0.851 | no_equivalence_claim (underpowered_no_equivalence_claim) |
| cosmos3_nano_policy_droid / binary R−L | +0.000 | -0.044 | 0.000 | 0.039 (521) | NR — zero margin | NR | no_equivalence_claim (margin_zero_equivalence_not_defined_test_emergence_only) |
| cosmos3_nano_policy_droid / depth R−L, m | +0.148 | +0.033 | 0.030 | 0.015 (521) | [+0.026, +0.040] | 0 / 0.7878 | no_equivalence_claim (strictly_powered_at_endpoints) |
| dreamzero_droid_action_cfg / binary R−L | +0.111 | -0.037 | 0.022 | 0.306 (27) | [-0.148, +0.074] | 0.6151 / 0.1792 | no_equivalence_claim (underpowered_no_equivalence_claim) |
| dreamzero_droid_action_cfg / depth R−L, m | +0.023 | -0.021 | 0.005 | 0.061 (27) | [-0.039, -0.002] | 0.92 / 0.01315 | no_equivalence_claim (underpowered_no_equivalence_claim) |
| fastwam_robotwin / binary R−L | +0.148 | +0.037 | 0.030 | 0.230 (27) | [-0.074, +0.148] | 0.186 / 0.6083 | no_equivalence_claim (underpowered_no_equivalence_claim_stretch) |
| fastwam_robotwin / depth R−L, m | +0.353 | -0.062 | 0.071 | 0.103 (27) | [-0.100, -0.024] | 0.3584 / 0 | no_equivalence_claim (underpowered_no_equivalence_claim_stretch) |
| pi05_current_stack_droid / binary R−L | +0.778 | +0.202 | 0.156 | 0.078 (341) | [+0.150, +0.252] | 0 / 0.9291 | no_equivalence_claim (strictly_powered_at_endpoints) |
| pi05_current_stack_droid / depth R−L, m | +0.207 | +0.058 | 0.041 | 0.017 (341) | [+0.048, +0.067] | 0 / 0.9982 | no_equivalence_claim (strictly_powered_at_endpoints) |

## H3 — inventory-matched dose response

A is the realised object-layout asymmetry (0 = symmetric). The registered primary slope excludes s=0 because the s=0→s>0 transition changes companion-object inventory.

| Checkpoint | Binary-gap slope per A (95% CI) | Depth-gap slope per A, m (95% CI) | Per-seed binary slope signs (+/−/0) |
| --- | ---: | ---: | ---: |
| cosmos3_edge_policy_droid | NR — only two registered levels | NR | NR |
| cosmos3_nano_policy_droid | -0.039 [-0.082, -0.000] | +0.020 [+0.000, +0.040] | 1/6/20 |
| dreamzero_droid_action_cfg | NR — only two registered levels | NR | NR |
| fastwam_robotwin | NR — only two registered levels | NR | NR |
| pi05_current_stack_droid | +0.191 [+0.086, +0.293] | +0.045 [+0.027, +0.064] | 19/5/3 |

## H5 — failure signature

The preregistered diagnostic is the within-seed slope of wrong-side share among behavioral failures versus realised A. A negative slope means wrong-side failures become more prominent as the object layout approaches symmetry. A level with no failures remains unavailable and is never imputed as zero.

| Checkpoint | Seed clusters | Mean slope | Median slope | Two-sided permutation p | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| cosmos3_edge_policy_droid | 2 | +0.000 | +0.000 | 1 | available |
| cosmos3_nano_policy_droid | 8 | -0.033 | +0.000 | 1 | available |
| dreamzero_droid_action_cfg | 3 | +0.000 | +0.000 | 1 | available |
| fastwam_robotwin | 27 | -0.014 | +0.000 | 0.1216 | available |
| pi05_current_stack_droid | 146 | +0.021 | +0.000 | 0.02625 | available |

## Interpretation rule

Interaction estimates answer whether the measured directional gap changes between the registered asymmetric and symmetric object layouts. Equivalence wording is permitted only where the preregistered power gate passed and the paired interval lies inside the registered margin. Underpowered or zero-margin rows remain descriptive even when their point estimate is near zero.

The s=0→s=1 comparison includes the preregistered companion-object inventory transition. The primary graded dose-response for π0.5 and Nano therefore uses inventory-matched s=0.25, 0.50, 0.75, and 1.00; s=0 is an anchored reference.

## Frozen prompts

- DROID LEFT: “Put the Rubik's cube to the left of the bowl.”
- DROID RIGHT: “Put the Rubik's cube to the right of the bowl.”
- RoboTwin LEFT: “Put the small woodenblock to the left of the red playingcards box.”
- RoboTwin RIGHT: “Put the small woodenblock to the right of the red playingcards box.”
