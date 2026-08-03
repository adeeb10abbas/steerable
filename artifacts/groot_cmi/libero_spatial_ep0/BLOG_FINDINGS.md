# GR00T N1.7 x LIBERO-Spatial CMI pilot

## Result snapshot

This frozen-state intervention used **10 prompts**, **8 stochastic action samples per prompt**, and **8 observations** from LIBERO-Spatial episode 0. For the first predicted arm action, language dependence cleared the per-frame permutation null at alpha=0.05 in **8/8 frames**, and **8/8** remained below the Bonferroni threshold **0.00625**. The median KDE CMI estimate was **2.8372 bits** (range **2.4918–2.9976**) against a **3.3219-bit** prompt-entropy ceiling. The peak was **2.9976 bits** at 85.3% episode progress. First-action gripper MI was nonzero in **1/8 frames** and reached **0.6605 bits**.

The pilot therefore found **no instruction-blind point** among the eight probed observations: GR00T's continuous arm distribution remained strongly prompt-dependent from the beginning to the end of this demonstration. That is a sensitivity finding, not a success claim—the swapped prompts are counterfactual and usually inconsistent with the frozen scene.

As a basic Monte Carlo check, the median first-action estimate was **2.5851 bits at N=2** and **2.8372 bits at N=8** using nested samples. The stable qualitative direction is encouraging, but the increasing magnitude shows that this small pilot has not established numerical convergence.

## What was measured

At each fixed visual/proprioceptive observation `S=s`, the model received every prompt in the pinned LIBERO-Spatial task pool. GR00T's flow-matching sampler generated repeated action chunks. The main metric is

`I(A; L | S=s) = H(A | S=s) - E_L[H(A | S=s, L)]`.

The six continuous arm dimensions use a shared-bandwidth, leave-one-out Gaussian KDE estimator expressed through the prompt posterior, so the raw estimate is bounded above by the prompt entropy. The binarized gripper uses categorical mutual information and is reported separately. Prompt labels are permuted while preserving group sizes to estimate the finite-sample null. `null_centered_score_bits` in the CSV is the raw estimate minus the permutation-null median; it is a diagnostic effect score, **not CMI**, and is not bounded by prompt entropy.

## Interpretation boundary

A frame that does not reject the permutation null is an **instruction-blind candidate**, not proof that the model never uses language. Conversely, high CMI proves sensitivity, not correct grounding: a model can change its action for the wrong linguistic reason. These are off-policy prompt swaps on states from one successful demonstration, and many swapped spatial descriptions are inconsistent with the rendered scene. LIBERO-Spatial also keeps the manipulated object and destination nearly constant across prompts, so low CMI can be rational state-based control rather than a model defect. With only eight Monte Carlo samples per prompt, these point estimates should be treated as a pilot and repeated with more samples, episodes, and random seeds.

## Why LIBERO-Spatial

This suite is a controlled first probe because the ten tasks keep the object (black bowl) and destination (plate) fixed while varying the bowl's spatial relation. The exact model and dataset are published in the LeRobot ecosystem: [GR00T policy guide](https://huggingface.co/docs/lerobot/groot), [LIBERO integration](https://huggingface.co/docs/lerobot/main/libero), [fine-tuned checkpoint](https://huggingface.co/nvidia/gr00t17-lerobot-libero_spatial-640), and [demonstration dataset](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot).

## Reproducibility pins

- Model: `nvidia/gr00t17-lerobot-libero_spatial-640` at `32a6ec786d6509df31b40392b4e4dcdda78c0f11`
- Base model: `nvidia/GR00T-N1.7-3B` at `2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- Processor assets: `Qwen/Qwen3-VL-2B-Instruct` at `89644892e4d85e24eaac8bacfd4f463576704203` (public Qwen3-VL architecture-compatible assets; the Cosmos-Reason2-2B asset repo is gated)
- Dataset: `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` at `bf14d6258218d12c2e3c1a3b9922e163cdf6455d`
- Episode: `0`
- Seed: `20260723`
- Continuous estimator: `kde` (bandwidth `None`; Gaussian diagnostic covariance `diagonal`, ridge `1e-06`)
- Permutations/bootstrap replicates: `200/200`

See `frame_metrics.csv` for every action-horizon estimate, `frame_summary.csv` for plot-level values, `sample_size_sensitivity.csv` for the nested-sample check, `cmi_over_rollout.png` for the main figure, and `probe_frames.png` for the frozen observations. The next decisive run is a preregistered replication over multiple episodes with at least 32–64 samples per prompt, paired with actual counterfactual rollout success so sensitivity can be separated from correct instruction following.
