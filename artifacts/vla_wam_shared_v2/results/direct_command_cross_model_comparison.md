# Direct-command evidence across models

Evidence cutoff: `e4c31fe686618590ca962d16fb606eda78446b7f` (2026-08-03T21:45:50Z). Source-set SHA-256: `3543560f867ccc8ca85fa619c0f05b8aec4c9312ba94f67b9756e34c2d8625d1`.

This is an arena-separated descriptive comparison. Raw DROID and RoboTwin success rates are never pooled. `NR` means the selected compiled evidence did not report a paired action-distinctness statistic; it is not a zero. Infrastructure-invalid attempts remain outside every valid-model denominator.

## DROID / RoboLab

| Class | Model | Valid n | LEFT | RIGHT | Endpoint aligned | Actions distinct | Future interface | Invalid attempts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| VLA | π0-FAST DROID | 20 | 1/10 | 10/10 | 10/10 | 10/10 | `none` | 0 (cell_attempts) |
| VLA | GR00T N1.7 DROID | 6 | 0/3 | 0/3 | 3/3 | 3/3 | `none` | 5 (ledger_entries; 2 behavior cells excluded) |
| WAM | Cosmos3 Edge DROID | 6 | 3/3 | 3/3 | 3/3 | 3/3 | `decoded_rgb_uint8_33_frames_per_policy_request` | 8 (setup_attempts; all before model request) |
| WAM | DreamZero DROID | 6 | 2/3 | 1/3 | 3/3 | 3/3 | `joint_action_and_latent_video_prediction_with_official_decode_path` | 11 (setup_attempts; all excluded from valid behavior) |

## RoboTwin place-A-relative-to-B

| Class | Model | Valid n | LEFT | RIGHT | Endpoint aligned | Actions distinct | Future interface | Invalid attempts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| VLA | LingBot-VLA 4B | 6 | 1/3 | 0/3 | 2/3 | NR | `none` | 3 (technical_setup_attempts) |
| WAM | Efficient-WAM-RT | 14 | 3/7 | 2/7 | 6/7 | 7/7 | `decoded_future_video` | 4 (cell_attempts) |
| WAM | FastWAM | 14 | 1/7 | 1/7 | 3/7 | 7/7 | `action_only_at_test_time` | 18 (cell_attempts) |
| WAM | LingBot-VA | 14 | 3/7 | 4/7 | 6/7 | 7/7 | `latent_only_future_not_decodable` | 5 (cell_attempts) |
| WAM | Light-WAM | 6 | 1/3 | 0/3 | 1/3 | 3/3 | `action_only_infer_action` | 6 (cell_attempts) |

## Exact evidence sources

- `artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json` — 74,549 bytes; SHA-256 `491c74812ed0e4d36c16f8e0ded17a70af3e69740c9bcb87af129bb6d9563073`
- `artifacts/vla_wam_shared_v2/pilot/expansion/groot_n17_droid_v2_registry.json` — 15,172 bytes; SHA-256 `95077a42bb0115bc673ea13ae5acdc6fdef6f476627804662f73c219ebd88bc7`
- `artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_direct_gate.json` — 67,708 bytes; SHA-256 `1c559ee5667ac9d22d7b66eafa7a65551783eedaf7fb3de29a2faf450c2dd029`
- `artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_invalid_attempts.json` — 6,004 bytes; SHA-256 `b3a62c792c82d15143ef6c94b768e2bcf712dd69d9c2f96584c904140a452754`
- `artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json` — 254,288 bytes; SHA-256 `4c76cdc3ca9eaf227d21d160199408f22e1b3dd7a71176a5a5dbe22223714461`
- `artifacts/vla_wam_shared_v2/pilot/expansion/lingbot_vla_4b_direct_gate.json` — 20,538 bytes; SHA-256 `7c0ad19833d6cbb51bb5fbdac8f9546f0e311333e498a15594b55f68dc7b6534`
- `artifacts/vla_wam_shared_v2/pilot/expansion/lingbot_vla_4b_robotwin_readiness.json` — 5,845 bytes; SHA-256 `588699ed912fe900de5a5ca36c350236e15ae0b0a2c55afeea22670759b68c30`
- `artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pair03_integration.json` — 7,275 bytes; SHA-256 `d850f8f2c3d32db705cc65326ca673c150ab7208a203a180bd32d88dfb0e5471`
- `artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pairs04_09_slice.json` — 64,863 bytes; SHA-256 `e7d2d3791323fecedb49e8c1ecc8fda1e0ade91dedd6613b9079f7b290e1fd54`
- `artifacts/vla_wam_shared_v2/pilot/directional_confirmation/fastwam_pairs03_09_slice.json` — 91,023 bytes; SHA-256 `9d52920cded17d0f61c997f02624ea38ffc4c9a3536cdacf9f71115b151d14be`
- `artifacts/vla_wam_shared_v2/pilot/directional_confirmation/lingbot_va_pairs03_09_slice.json` — 80,097 bytes; SHA-256 `8617da77c819ea57d374f463a672bf73414b088bb8188b9b13a6c1c2e1fb9d85`
- `artifacts/vla_wam_shared_v2/pilot/expansion/light_wam_robotwin_direct_gate.json` — 42,232 bytes; SHA-256 `f33e1ff8fdc82c4a035f2cc113b91b311d685d9747dbcbb1104453fc455745d6`
- `artifacts/vla_wam_shared_v2/pilot/expansion/light_wam_robotwin_registry.json` — 16,833 bytes; SHA-256 `c316bbebe8aa73cfe86b8749cb3cf6d8ebf438389a57b1d9e6c52dbfee67bbb5`

## Interpretation limits

- Success is the frozen arena-specific requested-relation completion predicate.
- Endpoint alignment and action distinctness are paired sensitivity measures, not task success.
- Exposed decoded futures are retained; action-only, latent-only, and missing future interfaces are never converted into zero-valued future scores.
- The three pairs03–09 RoboTwin WAM rows are prospective slices and do not synthesize or merge unavailable historical raw pairs00–02.
