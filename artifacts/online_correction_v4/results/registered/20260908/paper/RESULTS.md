# V4 registered campaign results

> **PARTIAL EXPORT** — C7 580/768 (75.5%); C6 pilot n/a with confirmatory n/a; C8 pilot 24/24 (100.0%) with confirmatory 204/768 confirmatory accepted; 768/768 dispatched.

**PARTIAL EXPORT** — three achievable families are in flight. C7 confirmatory coverage 580/768 (75.5%) on compiled_ledger_20260908m (580/768; FINAL compile pending); C6 confirmatory 56/768 terminals (wave F re-render); C8 confirmatory 204/768 (milestone compile 20260908f at 204).

Registered campaign scope: 17,664 planned policy episodes; 2,304 achievable (C6, C7, C8 confirmatory families × 768); 15,360 scientifically blocked (C1/C3/C4 horizontal information-gate squeeze 9,728; C2 reference_binding information gate 4,096; C5 vertical IK reachability 768). No eligibility criterion, threshold, or scale ladder was amended to recover blocked scope.

Cross-fixture contrast (compiled ledgers): C7 object_pair on Isaac is dominated by no_grasp (574/580) but is not uniformly no_grasp — compile-m adds 3 transport_incomplete, 2 wrong_goal_region, and 1 support_or_containment_failed (0 successes). C6 containment pilot shows grasps and placement attempts (no accepted outcomes). C8 second_stack WidowX pilot achieves grasp on 12/24 episodes (12 no_grasp, 12 transport_incomplete); transport_incomplete implies successful grasp with incomplete transport.

C8 pilot per-scenario grasp rates (destination_static 6/8 = 75%, move_stop 4/8 = 50%, original_sham 2/8 = 25%) formed a hypothesis about reference-placement effects. At 204/768 confirmatory episodes (ledger compile 20260908f), observed grasp rates are destination_static 4/63 (6.3%), move_stop 2/72 (2.8%), original_sham 2/69 (2.9%). Destination_static has ranked first at milestones 84, 101, 161, and 204, but magnitudes remain far below pilot reference and move_stop ≈ original_sham — the pilot's clean three-way ordering is not reproducing at scale. **Claim limit:** only destination_static ranking first is claimed; registered per-scenario contrast remains insufficient_coverage_not_estimable.

C8 episode accounting at the 200-episode milestone (c8_episode_accounting_verification_20260908e.json): 101 retained plus 103 new, 0 partials counted — verdict: accounting_intact_at_200_milestone_20260908f.

C8 destination_static confound elimination (c8_destination_static_confound_check_20260908a.json): green-to-yellow reset distance is identical at 0.1414 m across all three scenarios over 256 seeds each. Higher destination_static grasp rate is not explained by object starting closer to the goal region; the registered protocol difference is reference placement profile.

C6 confirmatory at 56/768 (c6_wave_d_progress_receipt_20260908e.json): destination_static 15.8%, move_stop 9.5%, original_sham 18.8% (n=19/21/16 per cell). Original_sham is highest, inverting C8's ordering — the second consecutive C6 milestone showing this pattern. Per-cell n is 16–21 and not interpretable yet; the registered platform×scenario interaction estimand (platform_scenario_grasp_contrast.csv) structures this as a testable contrast rather than an anomaly, with neither uniform nor platform-dependent outcome pre-committed.

C7 tail re-shard (c7_released_tail_reshard_analysis_20260908.json): final 180 episodes moved from 4 lanes to 14 lanes (attempt0611-attempt0624), cutting estimated wall time from ~22 h to ~5.4 h. Frozen queue.frozen.jsonl carries episode identity and seeds but no lane_id; episode-to-lane assignment is runtime-only, which makes re-sharding legitimate.

C6 wave D was re-rendered as rendered-c6confirm20260908f after a shared-checkout runner conflict; operating rule in runner_binding_resolution_20260908.json requires re-render when C7 mutates the shared checkout runner.

C7 compile provenance is explicit: partial compiles k (548), l (565), m (580), and stale .compiled-ledger-final (551) are retired in favor of compiled_ledger_20260908_FINAL at 768/768 require-full-coverage; see compile_retirement_provenance.csv.

Capacity finding (revised): a lane pair is a study-design and throughput unit, not a scheduler constraint—policy and simulator are separate single-GPU pods connected over HTTP. A PVC attempt-lock collision from same-attempt redispatch produced immediate simulator preflight failures across C7, C6, and C8, initially misread as Isaac warmup timeouts; remediation uses fresh attempt identities, a redispatch guard, and a 300-second startup grace (c8_execution_status_20260908o.json). Pending failures are attributable to per-node GPU fragmentation and legitimate C7 occupancy. tools/v4_gpu_pool_sweep.py reclaimed 34 GPUs; tools/v4_gpu_scheduling.py provides c8_a40_spread and c6_a10040_spread placement policies.

Three disclosed setup repairs bound this export: (1) NaturalGraspDetector trigger observation wiring now uses finger-contact coupling instead of robot-base pose; (2) eef_tool_length_m=0.14 flange-versus-fingerpad offset applied uniformly across Isaac fixtures; (3) second_stack SimplerEnv observation and sampling defect repaired with control-boundary sampling and provisioned render libraries.

279 pre-repair C7 episodes are excluded from all behavioral claims; only a ledger compile against the confirmatory manifest separates repaired-path accepted rows from infrastructure-invalid pre-repair attempts.

Disclosed geometry repair v2 (cube-only robot-base -X offset -0.03 m before common XY jitter) restored physical feasibility on the horizontal DROID fixture. Witness-delayed PVC contact timing verified clearance across all 128 jittered resets; every confirmatory path check passes at scale 0.5 (0.06 m displacement).

The registered scale ladder rejects larger scales on physical grounds: scale 2.0 fails both information and physical constraints (legal_goal_empty, collisions); scales 1.5, 1.0, and 0.75 fail on physical feasibility (unmodeled_collision, support/drift/pose co-failures) before a viable confirmatory scale is reached.

Scale 0.5 is the only ladder candidate with full physical path feasibility, but the homogeneous 128-seed wave fails the frozen 20% shrinking-area information gate on exactly 64/128 seeds—all with physical_translation_sign=-1—on right and behind goals. Removed-area fractions for those failing cases lie in 13.7–19.9%, just below threshold. All 64 sign=+1 seeds pass. Independent computation audit confirms correct gate math; this is not a sign-inversion bug.

G2 infrastructure gate passed with complete 128-seed aggregate and axis review at pin c401fb4. A40 versus A100-80GB determinism attestation passed at seed 2100000000. G5 negative control (hold_only) passed; positive scripted-grasp control documents fixture-specific transport feasibility limits and remains blocked pending Agent B shared-path fix—not a criterion change.

Primary estimand for C1/C3/C4 (9,728 episodes) is not estimable under the registered fixture. C2 (4,096 episodes) remains separately blocked on reference_binding information-gate evidence. No criteria, thresholds, or ladder scales were amended.

C7 behavioral evidence under verified repaired NaturalGraspDetector timing: timing_audit_confirms_genuine_policy_no_grasp — policy_failure_under_correct_timing. Compiled ledger (compiled_ledger_20260908m): 580 accepted valid with outcome decomposition 574 no_grasp, 3 transport_incomplete, 2 wrong_goal_region, 1 support_or_containment_failed.

C2 primary reference-selectivity (H) remains not estimable; the homogeneous G3 gate finalized at 128/128 as a scientific block with computation-correct information-gate rejection (4096 episodes). C6 is the only fixture passing the information gate on both translation-sign halves; C8 is the cross-platform WidowX/SimplerEnv check.

## Campaign tables and figures

- Tables: `artifacts/online_correction_v4/results/registered/20260908/tables/`
- Figures: `artifacts/online_correction_v4/results/registered/20260908/figures/`
- C7 family export: `artifacts/online_correction_v4/results/registered/20260908/families/C7/20260908c7/`
- C6 family export: awaiting Agent B confirmatory ledger receipt
- C8 family export: `artifacts/online_correction_v4/results/registered/20260908/families/C8/20260908c8pilot/`
- Horizontal squeeze slice: `artifacts/online_correction_v4/results/horizontal_geometry_repair_v2/20260908/`
