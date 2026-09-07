# V4 results export stub

_Generated from accepted ledger; replace TODO markers after review._

## Coverage by cell

| episode_id | family | fixture | block_id | policy_id | goal | wording | scenario | named_reference | planned | accepted_valid | missing_valid | infra_invalid | blocked | trigger_eligible | event_delivered | event_observed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_correction_v4-C8-b005-ebdb3ef349bae6db | C8 | second_stack | 5 | groot_bridge_widowx | left | direct | destination_static | single | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| online_correction_v4-C8-b012-04d926c8f4707973 | C8 | second_stack | 12 | groot_bridge_widowx | front | direct | original_sham | single | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| online_correction_v4-C8-b013-0a4f8fbbc3a8e6ab | C8 | second_stack | 13 | groot_bridge_widowx | right | direct | original_sham | single | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| online_correction_v4-C8-b023-d981b7ab74c7aeec | C8 | second_stack | 23 | groot_bridge_widowx | front | direct | original_sham | single | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| online_correction_v4-C8-b050-c73643a24a57efa9 | C8 | second_stack | 50 | groot_bridge_widowx | right | direct | move_stop | single | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| online_correction_v4-C8-b058-682b9fc5dab4468e | C8 | second_stack | 58 | groot_bridge_widowx | right | direct | original_sham | single | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
## Primary results

| contrast_registry_key | policy_id | robot_stack | estimand | point_estimate | ci95_low | ci95_high | standard_error | p_value | test_status | not_estimable_reason | holm_adjusted_p_value | holm_rejected_alpha_0.05 | n_reset_blocks | n_effective_blocks | bootstrap_resamples | bootstrap_seed | undefined_bootstrap_resamples | zero_or_undefined_se_resamples | descriptive_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_wording_x_motion_success_per_main_policy | cosmos3_nano_droid | robolab_droid | wording_x_motion_success_interaction_pp |  |  |  |  |  | not_estimable | no C1 reset blocks in supplied manifest |  | 0 | 0 | 0 | 10000 | 20260905 | 0 | 0 | "{""reason"": ""no C1 reset blocks in supplied manifest""}" |
| C1_wording_x_motion_success_per_main_policy | pi05_droid | robolab_droid | wording_x_motion_success_interaction_pp |  |  |  |  |  | not_estimable | no C1 reset blocks in supplied manifest |  | 0 | 0 | 0 | 10000 | 20260906 | 0 | 0 | "{""reason"": ""no C1 reset blocks in supplied manifest""}" |
| C2_reference_x_motion_goal_improvement_per_main_policy | cosmos3_nano_droid | robolab_droid | reference_selectivity_equal_goal_m |  |  |  |  |  | not_estimable | no C2 reset blocks in supplied manifest |  | 0 | 0 | 0 | 10000 | 20260907 | 0 | 0 | "{""reason"": ""no C2 reset blocks in supplied manifest""}" |
| C2_reference_x_motion_goal_improvement_per_main_policy | pi05_droid | robolab_droid | reference_selectivity_equal_goal_m |  |  |  |  |  | not_estimable | no C2 reset blocks in supplied manifest |  | 0 | 0 | 0 | 10000 | 20260908 | 0 | 0 | "{""reason"": ""no C2 reset blocks in supplied manifest""}" |
## Scope replications (C5–C8)

| family | robot_stack | status | note |
| --- | --- | --- | --- |
| C8 | simplerenv_bridge_widowx | not_run | scope replication estimators are registered separately; no raw cross-stack pooling |
## Failure composition

| episode_id | family | policy_id | block_id | goal | scenario | accepted_valid | success | failure_stage | failure_label | trigger_eligible | event_delivered | event_observed | motion_truncated_by_release | intervention_exposure_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_correction_v4-C8-b005-ebdb3ef349bae6db | C8 | groot_bridge_widowx | 5 | left | destination_static | 1 | 0 | pickup | no_grasp | 0 | 0 | 0 | 0 |  |
| online_correction_v4-C8-b012-04d926c8f4707973 | C8 | groot_bridge_widowx | 12 | front | original_sham | 1 | 0 | pickup | no_grasp | 0 | 0 | 0 | 0 |  |
| online_correction_v4-C8-b013-0a4f8fbbc3a8e6ab | C8 | groot_bridge_widowx | 13 | right | original_sham | 1 | 0 | pickup | no_grasp | 0 | 0 | 0 | 0 |  |
| online_correction_v4-C8-b023-d981b7ab74c7aeec | C8 | groot_bridge_widowx | 23 | front | original_sham | 1 | 0 | pickup | no_grasp | 0 | 0 | 0 | 0 |  |
| online_correction_v4-C8-b050-c73643a24a57efa9 | C8 | groot_bridge_widowx | 50 | right | move_stop | 1 | 0 | pickup | no_grasp | 0 | 0 | 0 | 0 |  |
| online_correction_v4-C8-b058-682b9fc5dab4468e | C8 | groot_bridge_widowx | 58 | right | original_sham | 1 | 0 | pickup | no_grasp | 0 | 0 | 0 | 0 |  |
## Timing and motion

| episode_id | family | policy_id | block_id | accepted_valid | t_event_planned_s | t_motion_actual_onset_s | motion_fraction_observed | commanded_peak_speed_m_s | achieved_peak_speed_m_s | observation_delay_s | inference_dispatch_delay_s | queue_execution_delay_s | physical_response_delay_s | inference_wall_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_correction_v4-C8-b005-ebdb3ef349bae6db | C8 | groot_bridge_widowx | 5 | 1 |  |  |  |  |  |  |  |  |  |  |
| online_correction_v4-C8-b012-04d926c8f4707973 | C8 | groot_bridge_widowx | 12 | 1 |  |  |  |  |  |  |  |  |  |  |
| online_correction_v4-C8-b013-0a4f8fbbc3a8e6ab | C8 | groot_bridge_widowx | 13 | 1 |  |  |  |  |  |  |  |  |  |  |
| online_correction_v4-C8-b023-d981b7ab74c7aeec | C8 | groot_bridge_widowx | 23 | 1 |  |  |  |  |  |  |  |  |  |  |
| online_correction_v4-C8-b050-c73643a24a57efa9 | C8 | groot_bridge_widowx | 50 | 1 |  |  |  |  |  |  |  |  |  |  |
| online_correction_v4-C8-b058-682b9fc5dab4468e | C8 | groot_bridge_widowx | 58 | 1 |  |  |  |  |  |  |  |  |  |  |
