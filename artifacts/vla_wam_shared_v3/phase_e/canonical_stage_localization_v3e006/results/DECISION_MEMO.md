# V3-E006 decision memo

Status: **gate_failed_no_valid_candidate_stop_before_registration**.

The exact E004 `s=1` full-reset state passed its reset, companion-pose, and camera checks. The final identical deterministic `canonical_grasp` construction did not pass the frozen validity gate: the physics, OOD-distance, and companion-pose components failed, while the camera component passed.

The largest diagnostic violations were arm speed 0.949 rad/s against a strict 0.01 rad/s threshold, cube angular speed 1.040 rad/s against 0.05 rad/s, cube midline residual 6.77 mm against 1 mm, OOD distance 9.722 against 7.663, and bowl displacement 13.63 mm against 5 mm. Normal cube–gripper contact was not sustained. These values describe why the candidate was rejected; they are not behavioral outcomes.

Per the prospective stopping rule, no alternative candidate was selected, no threshold was loosened, and V3-E006 was not registered, queued, or released for inference. Model requests, behavioral episodes, and accepted state candidates are all zero. Canonical-stage localization therefore remains inconclusive.
