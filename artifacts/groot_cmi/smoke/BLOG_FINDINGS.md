# GR00T N1.7 x LIBERO-Spatial CMI pilot

## Result snapshot

This frozen-state intervention used **2 prompts**, **2 stochastic action samples per prompt**, and **1 observations** from LIBERO-Spatial episode 0. For the first predicted arm action, language dependence cleared the per-frame permutation null at alpha=0.05 in **0/1 frames**. The median permutation-adjusted CMI was **18.6058 bits**. The peak was **18.6058 bits** at 0.0% episode progress. Mean first-action gripper MI was **0.0000 bits**.

## What was measured

At each fixed visual/proprioceptive observation `S=s`, the model received every prompt in the pinned LIBERO-Spatial task pool. GR00T's flow-matching sampler generated repeated action chunks. The main metric is

`I(A; L | S=s) = H(A | S=s) - E_L[H(A | S=s, L)]`.

The six continuous arm dimensions use a diagonal Gaussian moment estimator. The binarized gripper uses categorical mutual information and is reported separately. Prompt labels are permuted while preserving group sizes to estimate the finite-sample null.

## Interpretation boundary

A frame that does not reject the permutation null is an **instruction-blind candidate**, not proof that the model never uses language. These are off-policy prompt swaps on states from one successful demonstration, and many swapped spatial descriptions are inconsistent with the rendered scene. LIBERO-Spatial also keeps the manipulated object and destination nearly constant across prompts, so low CMI can be rational state-based control rather than a model defect.

## Reproducibility pins

- Model: `nvidia/gr00t17-lerobot-libero_spatial-640` at `32a6ec786d6509df31b40392b4e4dcdda78c0f11`
- Base model: `nvidia/GR00T-N1.7-3B` at `2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- Processor assets: `Qwen/Qwen3-VL-2B-Instruct` at `89644892e4d85e24eaac8bacfd4f463576704203` (public Qwen3-VL architecture-compatible assets; the Cosmos-Reason2-2B asset repo is gated)
- Dataset: `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` at `bf14d6258218d12c2e3c1a3b9922e163cdf6455d`
- Episode: `0`
- Seed: `20260723`
- Continuous covariance estimator: `diagonal`, ridge `1e-06`
- Permutations/bootstrap replicates: `19/0`

See `frame_metrics.csv` for every action-horizon estimate, `frame_summary.csv` for plot-level values, `cmi_over_rollout.png` for the main figure, and `probe_frames.png` for the frozen observations.
