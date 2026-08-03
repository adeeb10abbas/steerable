# GR00T N1.7 / LIBERO-Spatial CMI experiment

This isolated Python 3.12–3.13 environment keeps LeRobot's GPU dependency stack
out of the Python 3.10-compatible dataset-audit environment at the repository
root.

```bash
cd experiments/groot_cmi
uv sync

# Cheap integration check: 1 state x 2 prompts x 2 stochastic samples.
uv run steerable-groot-cmi --smoke \
  --output-dir ../../artifacts/groot_cmi/smoke

# Pinned pilot used for the blog artifacts.
uv run steerable-groot-cmi \
  --output-dir ../../artifacts/groot_cmi/libero_spatial_ep0 \
  --dataset-root ../../data/groot_cmi/libero_spatial \
  --episode 0 \
  --num-frames 8 \
  --samples-per-prompt 8 \
  --max-prompts 10
```

Sampling is checkpointed one frame at a time under `samples/`. Re-running the
same command keeps completed frames. Use `--analyze-only` to regenerate the
CSV, plots, and blog summary without loading GR00T, or `--force` to replace
samples.

The default estimator treats the six arm commands as continuous and the final,
binarized gripper command as categorical. The main plot shows first-action and
action-chunk arm CMI plus the prompt-label permutation p-value at each frozen
observation.
The CSV's `null_centered_score_bits` subtracts the finite-sample null median;
it is a diagnostic score rather than mutual information and can exceed the
prompt-entropy ceiling. Blog claims use the bounded raw `cmi_bits` estimate.
`sample_size_sensitivity.csv` recomputes the first-action estimate with nested
subsets of the saved Monte Carlo samples, without another model forward pass.

The GR00T checkpoint points its processor at the gated
`nvidia/Cosmos-Reason2-2B` repository. The harness uses pinned public
`Qwen/Qwen3-VL-2B-Instruct` tokenizer/image/video processor assets instead;
GR00T N1.7 identifies Cosmos-Reason2-2B as a Qwen3-VL architecture, and this
substitution is recorded explicitly in every run manifest.

The pinned demonstration release uses LeRobot dataset format v2.1, while the
generic LeRobot 0.6 dataset loader requires v3.0. To avoid silently converting
or substituting data, the harness has a minimal read-only v2.1 episode adapter
for the exact pinned Parquet and MP4 files used by this experiment.
