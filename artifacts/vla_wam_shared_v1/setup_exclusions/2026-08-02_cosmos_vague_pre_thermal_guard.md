# Excluded complete Cosmos short-paraphrase batch before thermal guard

- Status: excluded in full; no episode enters any prospective estimate.
- Raw output:
  `/home/ali/projects/RoboLab/output/v1_cosmos_vague_pre_thermal_guard`
- Policy role: physical GPU 1, Cosmos3 Edge DROID with decoded futures.
- Simulator role: physical GPU 0, RoboLab Isaac Sim.
- Frozen seeds: 6100--6109.
- Completed outcomes: left 1/10 (`FFFFSFFFFF`), right 9/10
  (`SSSSSSSSSF`).

This batch completed safely and is not excluded for model behavior or corrupt
data. It predates `thermal_control_amendment_001.json`. After the matched-role
canonical batch reached the 90 C hard stop, the study activated a logged 87 C
pause / 80 C resume guard for every remaining definitive rollout. Because a
host pause may influence wall-clock-dependent realtime rendering, the static
wording comparison is rerun in full under one common active guard rather than
comparing this older short-paraphrase batch with guarded canonical episodes.

All HDF5, logs, conditioning images, action chunks, and generated futures are
preserved as supporting evidence. They remain useful as a disclosed robustness
check but cannot substitute for the definitive rerun.
