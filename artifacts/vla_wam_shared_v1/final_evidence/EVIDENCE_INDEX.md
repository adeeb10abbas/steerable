# VLA-WAM study evidence index

Status: complete direct-language grid with an original confirmatory tier, a prospectively frozen post-interim stress tier, and a separately labeled retrospective WAM tier.

## Integrity checks

- Closed-loop episodes: **160/160**.
- Original confirmatory episodes: **80/80**.
- Post-interim direct-language stress episodes: **80/80**.
- Oracle or dynamic-prompt episodes in the analysis: **0**.
- Shared physical robot/object reset fingerprints: **1**; full recorder-schema fingerprints: **2**.
- First-recorded cube/bowl physical observations audited: **160**.
- Fixed-observation probe conditions: **16 per model**.
- Exact direct-task prompt conditions: **11/11 per model**.
- Cosmos semantically scored confirmation chunks: **752** across **80 episodes**.
- Prompt-blind semantic scorer coverage: **421/752 chunks**; every abstention remains explicit.
- Frozen-order semantic categories rendered: **5/5 observed**; absent categories remain explicit empty panels.
- Cosmos recorded prediction chunks with verified request/server seeds, step alignment, and shapes: **752**.
- Completed thermal-guard lifecycles without emergency stop: **8/8**.
- Executed trajectory panels indexed: **160/160**, including **83 successes** and **77 failures**.
- Calibration, command-probe, and run-manifest hashes were verified by the compilers.

## Prospective evidence

| Evidence | Artifact | Purpose |
| --- | --- | --- |
| Frozen design | `../preregistration.json` | Questions, fixed grid, primary/secondary metrics, stopping rule |
| Direct-language scope amendment | `../direct_language_scope_amendment_003.json` | Retires the oracle grid and freezes declarative/contrastive task-language stress conditions before those runs |
| Metric amendment | `../metric_amendment_001.json` | Exact paper-style progression after primary-source verification |
| Observation amendment | `../observation_variation_amendment_001.json` | Downgrades closed-loop action contrast after measured renderer variation |
| Initial-state schema amendment | `../initial_state_schema_amendment_006.json` | Separates exact physical reset identity from checkpoint-specific camera recorder schemas |
| Thermal-control amendment | `../thermal_control_amendment_001.json` | Freezes pause/resume and emergency-stop behavior after the first matched-role thermal stop |
| Thermal timing amendment | `../thermal_timing_amendment_002.json` | Treats guarded client request timing as an upper bound and forbids fabricated phase attribution |
| Semantic target parser amendment | `../semantic_target_parser_amendment_004.json` | Uses matched task identity rather than interpreting contrastive prompt negation inside the visual scorer |
| Execution geometry amendment | `../execution_geometry_amendment_005.json` | Aligns derived task/execution relations with RoboLab rigid-object root poses while preserving visual-centroid calibration |
| Trajectory visualization plan | `../trajectory_visualization_plan.json` | Freezes complete-gallery, deterministic social-panel, and retrospective-exemplar rules |
| Semantic-future visualization plan | `../semantic_future_visualization_plan.json` | Freezes first-in-order example selection before confirmation semantic labels |
| Grounded probe plan | `../command_probe_plan.json` | Hash-pinned observation, six command styles, controls, seed |
| Direct-task probe plan | `../direct_task_command_probe_plan.json` | Exact-input syntax, contrastive scope, and target-token-order diagnostic |
| Closed-loop episode table | `episodes.csv` | One row per registered direct-language rollout, with analysis tier |
| Closed-loop summary | `closed_loop_summary.json` | Success, progression, offsets, timing, contrasts |
| Complete trajectory index | `../trajectory_evidence/trajectory_index.csv` and `.json` | Every success and failure with endpoint class, event steps, raw paths, and rendered panel |
| Trajectory evidence gallery | `../trajectory_evidence/gallery/index.html` | Filterable visual audit of every registered episode |
| Cosmos future semantics | `compiled_evidence.json` | Prompt-blind imagined/executed quadrants and coverage |
| Semantic-future examples | `../semantic_future_visualization/selection.json` | Deterministic source rows, videos, caches, and hashes for each observed quadrant |
| Renderer variation audit | `cosmos_observation_variation.csv` | First-conditioning-image differences within and across static conditions |
| Physical settling audit | `initial_physical_variation.csv` | Reset-state identity versus first-recorded cube/bowl centroid differences |
| Human semantic audit | `../semantic_confirmation_audit_plan.json`, `../semantic_confirmation_audit_amendment_002.json`, and `../semantic_confirmation_audit.md` | Outcome-independent sheet samples and completed visual review |
| Publication article | `../../../docs/VLA_VS_WAM_STEERABILITY_STUDY.md` | Long-form interpretation with complete claim boundaries and visual evidence |
| Social launch kit | `../../../docs/VLA_WAM_STEERABILITY_SOCIAL_COPY.md` | Post-ready X/LinkedIn copy, carousel order, alt text, and claim guardrails |
| Command probes | `compiled_evidence.json` | Exact repeat, command sensitivity, semantic futures |
| GPU assignment audit | `../cosmos_gpu_assignment_audit.json` | Quantifies why cross-card Cosmos output was excluded |
| Cosmos resource snapshot | `../operational_snapshot_cosmos_confirmation.json` | Temperatures, memory, utilization, and physical GPU roles during a valid WAM request |
| pi0.5 resource snapshot | `../operational_snapshot_pi05_confirmation.json` | Temperatures, memory, utilization, and physical GPU roles during a valid VLA request |
| Thermal event logs | `../thermal_logs/*.jsonl` | Complete pause/resume lifecycle, cooldown duration, sampled peak, and emergency-stop audit for all eight definitive batches |
| Raw file hash ledger | `raw_evidence_manifest.csv` | Byte size and SHA-256 for every prospective raw/derived evidence file |
| Supporting hash ledger | `supporting_evidence_manifest.csv` | Calibration, exclusions, and separately labeled retrospective raw/derived evidence |
| Setup exclusion | `../setup_exclusions/2026-08-02_cosmos_canonical_driver_check.md` | Failed startup with zero requests, excluded transparently |
| Thermal exclusion | `../setup_exclusions/2026-08-02_cosmos_gpu0_thermal_restart.md` | Interrupted and cross-GPU batches preserved outside estimates |
| Confirmation thermal exclusion | `../setup_exclusions/2026-08-02_cosmos_confirmation_thermal_stop.md` | Whole matched-role batch excluded after the simulator reached the 90 C stop threshold |
| Pre-guard wording exclusion | `../setup_exclusions/2026-08-02_cosmos_vague_pre_thermal_guard.md` | Completed short-paraphrase batch rerun so both wordings share one logged thermal cadence |
| Oracle scope exclusion | `../setup_exclusions/2026-08-02_oracle_scope_change.md` | Preserves the interrupted coached batch while excluding it from every direct-language estimate |

## Figures

- `direct_language_success_with_intervals.png`: binary success for all four static task wordings with Beta(1,1) 95% intervals.
- `direct_language_requested_side_offsets.png`: endpoint directionality, including failures.
- `direct_prompt_robustness.png`: model-by-wording-by-direction success matrix without a coach.
- `../trajectory_evidence/blog/all_executed_paths_and_endpoints.png`: every executed cube path and endpoint, faceted by checkpoint and wording.
- `../trajectory_evidence/blog/failure_progress_anatomy.png`: mutually exclusive action-stage anatomy for every success and failure.
- `../trajectory_evidence/social/first_seed_stress_landscape_1600x900.png`: deterministic same-seed stress-language comparison for social sharing.
- `../trajectory_evidence/social/first_seed_stress_square_1200x1200.png`: square social crop of the same deterministic comparison.
- `../trajectory_evidence/social/steerability_scorecard_1600x900.png` and `...1200x1200.png`: complete 16-cell checkpoint/wording/direction scorecard in share-ready formats.
- `../trajectory_evidence/social/failure_progress_anatomy_1200x1200.png`: square share card retaining every success and failure stage.
- `cosmos_conditioning_image_variation.png`: measured realtime-renderer variation despite exact physical resets.
- `cosmos_imagination_execution_quadrants.png`: WAM-only semantic future/action agreement.
- `semantic_threshold_sensitivity.png`: scorer coverage/agreement at 0.10, 0.15, and frozen 0.20 m reliability thresholds.
- `../semantic_future_visualization/blog/selected_semantic_future_examples.png`: frozen first-in-order generated-video strip for every observed semantic category.
- `../semantic_future_visualization/social/wam_semantic_quadrants_1600x900.png` and `...1200x1200.png`: share-ready actual-future examples for the four certain imagination/execution outcomes.
- `command_probe_action_sensitivity.png`: same-observation six-style prompt response.
- `direct_task_exact_probe.png`: same-input left/right and contrastive word-order action separation.
- `command_probe_selected_futures.png`: selected Cosmos future strips with frozen prompt-blind relation labels.

## Retrospective evidence tier

Efficient-WAM, FastWAM, LingBot-VA, and the earlier π0.5/Cosmos pilots remain in `../../wam_language_gate/`. They inform model selection and failure analysis, but they are not pooled into the prospective confidence intervals.

## Statistical guardrails

- A Beta(1,1) interval accompanies each success proportion.
- Declarative and contrastive conditions are explicitly post-interim stress tests, not retroactively presented as part of the original preregistration.
- Exact paired McNemar tests are exploratory and uncorrected for multiple comparisons.
- Replan chunks are correlated within episodes; semantic quadrant rates are descriptive and receive no pseudo-replicated binomial interval.
- One checkpoint represents each model class, so no result establishes a general VLA-versus-WAM class effect.
- Fixed-observation distances establish sensitivity only. Directionally appropriate closed-loop outcomes establish control.

## Provenance

- steerable: `7502edaa2a2821ae6c603df933bc16b2a75722e8` (codex/wam-language-steerability).
- RoboLab: `992bc34eedb2b909888af8a334a2ac33b86c51d8` (codex/wam-steerability).
- cosmos-framework: `1439c1d5e45a23771e9b1a2ad8f40a5981ea86c0` (main).
- openpi-robolab: `9e46d3aea26417bfb564227734b95d010aa827e5` (main).

Core file hashes are stored under `provenance.files` in `compiled_evidence.json`.
