# Excluded Cosmos paraphrase batch: thermal safety restart

The first `v1_cosmos_vague` attempt was stopped during left-direction run 4
after physical GPU 0 reached 92°C, its fan reached 95–96%, and
`clocks_throttle_reasons.sw_thermal_slowdown` became active. The host would not
allow a lower NVIDIA power limit without an interactive sudo password. Keeping
the long overnight sweep on that card was not hardware-safe.

At interruption, four episodes were complete and run 4 was partial. The output
contained five HDF5 files, 58 recorded future chunks, and four serialized
episode results. The entire directory was preserved rather than modified:

`/home/ali/projects/RoboLab/output/v1_cosmos_vague_interrupted_hot_gpu0`

These four complete episodes and the partial episode are excluded from every
estimate. The registered `v1_cosmos_vague` directory is rerun from the start,
with the same checkpoint, prompts, seeds, simulator, driver workaround, and
prediction recording. Only the physical roles of the two identical RTX 3090s
are swapped: the Cosmos server moves to the cooler physical GPU 1 and Isaac Sim
moves to physical GPU 0.

Before accepting the restarted batch, seed 6100 was compared against its
preserved counterpart. The simulator initial-state arrays were exactly equal
(maximum absolute difference 0.0), as were the prompt, request/server seed,
joint state, gripper state, and output shapes. The physical-GPU swap was **not**
pixel reproducible:

- conditioning-image MAE was 0.193934 on the 0–255 scale;
- 17.43% of color-channel values differed, with a 99th-percentile absolute
  difference of 2 and a maximum of 15;
- first action-chunk RMS was 0.010875;
- none of 15 conditioning PNG, action NPY, or future MP4 hashes matched.

The safety restart is therefore a real renderer/model-input factor despite the
identical simulator state. The original canonical batch, which used the old GPU
assignment, is also excluded and rerun from seed 6100 under the new assignment.
Only the fresh canonical and short-paraphrase batches—both Cosmos server on
physical GPU 1 and Isaac Sim on physical GPU 0—enter the prospective comparison.
