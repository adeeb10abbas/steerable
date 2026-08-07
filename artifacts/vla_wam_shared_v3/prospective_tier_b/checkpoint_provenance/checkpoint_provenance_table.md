# V3 checkpoint provenance

Training episode membership, sampling, training preprocessing, and caption exposure are not reconstructed from model names or inference adapters. `unknown` and `not_disclosed` are evidence boundaries, not negative findings.

| Checkpoint | Family / arena | Artifact revision | Inference interface | Future interface | Training multiset | Caption audit |
| --- | --- | --- | --- | --- | --- | --- |
| `pi0_fast_droid_vla` | VLA / droid_robolab | `not_disclosed` | action-only, preserved 10x8 interface | no decoded future | partially_disclosed | not_auditable |
| `pi0_fast_old_name_config_v3a002` | VLA / droid_robolab | `not_disclosed` | 10x8 action-only output | no decoded future | partially_disclosed | not_auditable |
| `pi05_current_stack_droid` | VLA / droid_robolab | `v2a010-manif…` | 15x8 joint-position actions | no decoded future | partially_disclosed | not_auditable |
| `groot_n17_droid_vla` | VLA / droid_robolab | `05e7cc97e40d…` | up to 40x8 returned; first 8 executed per request | no decoded future | partially_disclosed | not_auditable |
| `cosmos3_edge_policy_droid` | WAM / droid_robolab | `3ea407af3e15…` | 32x8 joint-position actions | 33-frame decoded RGB future exposed per request | partially_disclosed | not_auditable |
| `cosmos3_nano_policy_droid` | WAM / droid_robolab | `6706d7680581…` | 32x8 joint-position actions | 33-frame decoded RGB future exposed per request | partially_disclosed | not_auditable |
| `dreamzero_droid_action_cfg` | WAM / droid_robolab | `96ad344138c6…` | 24x8 returned; first 8 executed per request | latent video with official reset decode | partially_disclosed | not_auditable |
| `efficient_wam_rt_robotwin` | WAM / robotwin | `81280a79e8ac…` | 16-action chunks | decoded coarse future video, maximum one retained chunk | partially_disclosed | not_auditable |
| `fastwam_robotwin` | WAM / robotwin | `139eebb6d90c…` | 32-action horizon; replan at 24 | action-only at test time | partially_disclosed | not_auditable |
| `lingbot_va_robotwin` | WAM / robotwin | `lerobot/ling…` | exposed native action trajectory | latent-only tensor [1,48,2,24,20], not decoded | partially_disclosed | not_auditable |

The historical π0-FAST row and the V3-A002 compatibility row are distinct identities and must not be pooled. DROID/RoboLab and RoboTwin remain separate arenas.
