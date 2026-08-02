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

Before accepting the restarted batch, seed 6100 is compared against its
preserved counterpart for exact first-chunk action and future-video equality.
If the two cards do not reproduce, the restart is invalid and the GPU identity
must be treated as an experimental factor.
