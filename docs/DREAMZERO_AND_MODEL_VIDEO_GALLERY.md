# DreamZero and matched VLA/WAM video evidence

This is the portable index for the embedded [HTML video gallery](DREAMZERO_AND_MODEL_VIDEO_GALLERY.html). DROID and RoboTwin are listed separately and their success rates are never pooled.

## DreamZero status

**Pending — no behavioral video exists in the committed evidence.** This is not a zero. The generator is wired to `artifacts/vla_wam_shared_v2/media/dreamzero_droid/media_manifest.json` and will ingest its `gallery_entries` only after every referenced clip validates.

## DROID / RoboLab

### Cosmos3 Edge DROID — seed 8302 matched pair

[▶ Open video](../artifacts/vla_wam_shared_v2/media/cosmos3_edge_droid/cosmos3_edge_seed8302_paired.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/media/cosmos3_edge_droid/media_manifest.json)

- Outcome: LEFT: success after 397 actions; RIGHT: success after 123 actions
- Future interface: Decoded video futures plus actions; 17 decoded futures retained for this pair
- Evidence status: Valid behavioral pair; both episodes succeeded; committed publication selection
- Video SHA-256: `5eed8b8468aff6070617e126eac1c67a19fd62309279ca1a06f2d3b5abf4bdc9`

> LEFT: “Put the Rubik's cube to the left of the bowl.”
> RIGHT: “Put the Rubik's cube to the right of the bowl.”

### GR00T N1.7 DROID — seed 8301 matched pair

[▶ Open video](../artifacts/vla_wam_shared_v2/media/groot_n17_droid/groot_n17_droid_seed8301_pair.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/media/groot_n17_droid/media_index.json)

- Outcome: LEFT: failure after 450 actions; RIGHT: failure after 450 actions
- Future interface: Action chunks only; no decoded visual future
- Evidence status: Valid behavioral pair; both episodes failed; committed publication selection
- Video SHA-256: `3ff5ac38d8bd224f336531b93c09cd77dee565ab724647f4019a8b2e5d60600d`

> LEFT: “Put the Rubik's cube to the left of the bowl.”
> RIGHT: “Put the Rubik's cube to the right of the bowl.”

### π0-FAST DROID — seed 8300 matched pair

[▶ Open video](../artifacts/vla_wam_shared_v2/media/droid_pi0_fast_pairs/pi0_fast_seed8300_left_failure_right_success.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/media/droid_pi0_fast_pairs/media_index.json)

- Outcome: LEFT: failure: no object interaction; RIGHT: success
- Future interface: Actions only; no decoded visual future
- Evidence status: Valid behavioral pair; LEFT failed and RIGHT succeeded; committed publication selection
- Video SHA-256: `84a424cdfb796bbce02a88eb6d9a22de74fc3b264b4a103bece09aea99ed3a6a`

> LEFT: “Put the Rubik's cube to the left of the bowl.”
> RIGHT: “Put the Rubik's cube to the right of the bowl.”


## RoboTwin

### Efficient-WAM-RT — pair 00 pilot

[▶ Open video](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/efficient_wam_rt_pair00_left_success_right_failure.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/media_index.json)

- Outcome: LEFT: success; RIGHT: failure: picked, never entered requested region
- Future interface: Decoded coarse future video plus actions; publication clip shows simulator execution
- Evidence status: Valid behavioral pair; LEFT succeeded and RIGHT failed; committed publication selection
- Video SHA-256: `bad140d1872ee904bd364c0f8c022b7e5944e3c7529d68f43b93ae72a194a7aa`

> LEFT: “Put the blue soap to the left of the tea-box.”
> RIGHT: “Put the blue soap to the right of the tea-box.”

### Efficient-WAM-RT — pair 05 confirmation

[▶ Open video](../artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation/efficient_wam_rt_pair05_left_right_both_success.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation/media_index.json)

- Outcome: LEFT: success after 46 actions; RIGHT: success after 47 actions
- Future interface: Decoded coarse future video plus actions; publication clip shows simulator execution
- Evidence status: Valid prospective confirmation pair; both episodes succeeded; committed publication selection
- Video SHA-256: `bbc833fc1faf84f38bd8307b8d7f6b73d0fad957ce5a822c7f07a1db6ba2e1a4`

> LEFT: “Put the box of playingcards to the left of the rubikscube.”
> RIGHT: “Put the box of playingcards to the right of the rubikscube.”

### FastWAM — pair 02 pilot

[▶ Open video](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/fastwam_pair02_left_success_right_failure.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/media_index.json)

- Outcome: LEFT: success; RIGHT: failure: picked, never entered requested region
- Future interface: Action-only at inference; world-model objective is training-time only
- Evidence status: Valid behavioral pair; LEFT succeeded and RIGHT failed; committed reconstructed publication selection
- Video SHA-256: `7a152b61292fdfe9d341950d4448dce4c8c5d4b797aa073b246bafafbaaae4f2`

> LEFT: “Put the box with cards inside to the left of the red coffee-box.”
> RIGHT: “Put the box with cards inside to the right of the red coffee-box.”

### LingBot-VA — pair 00 pilot

[▶ Open video](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/lingbot_va_pair00_left_success_right_failure.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/media_index.json)

- Outcome: LEFT: success; RIGHT: failure: picked, never entered requested region
- Future interface: Joint video/action latent retained; no publication-ready visual decoder
- Evidence status: Valid behavioral pair; LEFT succeeded and RIGHT failed; committed publication selection
- Video SHA-256: `d7f5addbfbf0074a2e116d197b12b679e5489e81dd0120334221347c406c8a0b`

> LEFT: “Put the blue soap to the left of the tea-box.”
> RIGHT: “Put the blue soap to the right of the tea-box.”

### LingBot-VA — pair 03 confirmation

[▶ Open video](../artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation/lingbot_va_pair03_left_right_normalized_full_rollouts.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation/media_index.json)

- Outcome: LEFT: failure after 400 actions; RIGHT: success after 139 actions
- Future interface: Joint video/action latent retained; no publication-ready visual decoder
- Evidence status: Valid prospective confirmation pair; LEFT failed and RIGHT succeeded; committed publication selection
- Video SHA-256: `074abb4d5ef3b39328535d6e86037c680da5afb45384884745df0624bcc09341`

> LEFT: “Put the small woodenblock to the left of the red playingcards box.”
> RIGHT: “Put the small woodenblock to the right of the red playingcards box.”

### LingBot-VLA 4B — pair 00 direct gate

[▶ Open video](../artifacts/vla_wam_shared_v2/pilot/expansion/media/lingbot_vla_4b_pair00_matched.mp4) · [Evidence manifest](../artifacts/vla_wam_shared_v2/pilot/expansion/lingbot_vla_4b_direct_gate.json)

- Outcome: LEFT: success after 127 actions; RIGHT: failure after 400 actions: picked, never entered requested region
- Future interface: Actions only; no decoded visual future
- Evidence status: Valid behavioral pair; LEFT succeeded and RIGHT failed; committed matched clip
- Video SHA-256: `6fdf9383bcee0abf5cd6910afd477b3aa6bd69a0fd36ee4b8ab3eaf059b697ae`

> LEFT: “Put the blue soap to the left of the tea-box.”
> RIGHT: “Put the blue soap to the right of the tea-box.”

## Missing publication media

- **dreamzero_droid — pending_no_behavioral_video:** No valid DreamZero direct-command rollout or committed publication video exists yet.
- **pi05_droid — no_selected_v2_publication_video:** The checkpoint is an existing v1 reference, but this branch contains no committed selected v2 behavioral clip for it.
- **light_wam_robotwin — raw_video_not_selected_for_publication:** All six valid episodes reference simulator videos on the ali PVC, but no compact selected Light-WAM publication clip is committed.
- **lawam_robotwin — no_behavioral_episode:** Blocked before inference on gated DINOv3 access; no behavioral video exists.
- **cosmos_reason2_2b — not_behavioral_media:** Static diagnostic calls are not robot rollouts and are intentionally excluded from the behavioral video gallery.

Regenerate and validate with:

```bash
python3 tools/render_vla_wam_video_first_gallery.py
```
