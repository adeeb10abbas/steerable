# Evidence Map

This map prevents the manuscript from drifting beyond the registered evidence. Repository paths refer to `adeeb10abbas/steerable` at verified `origin/main` commit `3664e741326880aaaf1d3a005896079514b173cc`.

| Manuscript claim | Value used | Primary evidence / audit path | Boundary |
|---|---:|---|---|
| Canonical matched-pair protocol | One word changes; state, seeds, checkpoint, controller fixed | `docs/VLA_WAM_STEERABILITY_V3_PROTOCOL.md`; Phase-A summaries | No oracle, switching, or progress-conditioned prompts |
| Action distinctness | Bitwise inequality on complete common executed prefix | `experiments/v3/pi05_phase_b/diagnostics.py` and analogous pair compilers | Response diagnostic, not semantic orientation |
| Canonical screen | 324/324 distinct; endpoints 125/135 DROID and 128/189 RoboTwin | Phase-A summaries; `docs/PUBLICATION_HANDOFF.md`; `artifacts/vla_wam_shared_v3/analysis/paper_figures/figure5_cross_arena_directional_success.*` | Eight completed checkpoint--arena rows (five DROID, three RoboTwin); arenas remain separate; $\pi_{0.5}$ row uses seeds 8303--8329 |
| Fixed-observation diagnostic | 336 requests; 12 exact repeats RMS 0 | `artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001/results/compiled_results.json` | Zero behavioral episodes; action-space only |
| Reference inversion (C002-R001) | 341 blocks, 1,364 episodes; depth inverse-minus-canonical LEFT -9.783 cm, RIGHT -13.307 cm | `artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v4/final_analysis_v3/results/` | Seeds 12000--12340; independent of E004 seeds 9400--9740; prospective post-gate operational repair; original C002 failed cross-server isolation |
| C002 endpoint controls | Canonical +23.281 cm passes; inverse +0.191 cm, CI [-0.884,+1.272], fails | C002-R001 decision memo and `results.json` | Semantic-equivalence claim withheld |
| Reflection | Depth interactions -34.6/-24.8/-14.1 cm; pi0.5 binary inversion | `artifacts/vla_wam_shared_v3/analysis/paper_figures/figure2_three_checkpoint_position_reflection.*` and Phase-B summaries | DROID only; 27 matched seeds per checkpoint; $\pi_{0.5}$ seeds 9400--9426 are independent of Phase A; redirection equivalence not established |
| Nano lateral sweep | Full-support slope 1.12 m/m, CI [0.72,1.56], 13/15 positive | `phase_b/nano_lateral_sweep_v3b005/results/` and hash-closed scientific report | Registered primary uses all seven levels; middle-five 0.49 m/m sensitivity is post hoc |
| Start side / role swap / stochastic repeats | +0.296; +0.13 m; 41/216 vs 197/216 | prospective Tier-B bundles indexed by `tools/build_v3_scientific_report.py` | Converging scene diagnostics, not pooled checkpoint estimates |
| Symmetric-scene cohort | 4,096 episodes, 2,048 pairs | `artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/DECISION_MEMO.md`; `results/results.json` | Registered scene package includes companion-object inventory transition; FastWAM positive control fails |
| pi0.5 full endpoints | Binary 74.2 to 20.2 pp; depth 17.0 to 5.75 cm; endpoint redirection +23.9 cm, 335/341 positive | E004 `analysis.levels["0.00"]` and `["1.00"]` | 341 pairs per endpoint; descriptive endpoint estimates |
| pi0.5 matched-core interactions | Binary -51.9 pp [-81.5,-22.2]; depth -12.4 cm [-19.7,-4.9] | E004 `interaction_s1_minus_s0_core` | Separate matched 27-seed causal core; do not pool with endpoint inference |
| Other E004 core interactions | DreamZero binary -55.6 pp; Nano depth -13.9 cm; Edge depth -19.2 cm | E004 decision memo and `results.json` | Matched 27-seed core; binary ceiling/power boundaries retained |
| E004 failure composition | pi0.5 P/T/W/R: 1/267/41/2 to 0/158/7/0 | E004 full endpoint `failure_taxonomy` | 341 seed pairs, 682 episodes per scene; not the earlier E003 slice |

## Evidence discipline

- Never pool DROID/RoboLab and RoboTwin numerators, denominators, or intervals.
- Never convert not-reported measurements into zeros.
- Keep every valid behavioral failure in the denominator and infrastructure-invalid attempts in a separate ledger.
- Call C002-R001 a prospective post-gate operational repair, not a passed original isolation gate.
- Describe the prompts as reference-inverted expressions of the same physical relation; do not imply that the policy achieved equivalent endpoints.
- Separate the E004 341-seed endpoint summaries from the 27-seed matched-core interaction tests.
- The symmetric condition changes a registered scene package, including companion-object inventory; it is not a position-only intervention.
- The marginal corpus numbers are excluded from the paper until a pinned artifact exists in the target commit.

## Before submission

Re-run repository validators and publication-figure checks from the pinned commit, record the SHA in the artifact manifest, and have a second author trace each headline number to the decision memo or `results.json`.
