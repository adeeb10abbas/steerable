# V3-E006-R003 diagnostic closure

R003 did not release a canonical state or any behavioral inference. The first registered known-reachable diagnostic, the LEFT historical canonical-grasp anchor, failed the unchanged final-window position tolerance. Its final ten samples were 1.22750–1.22794 mm from the registered base-link target, versus the inclusive 1.00000 mm limit. Orientation was 0.22079–0.22104 degrees, within the unchanged 1-degree limit.

The historical state was restored exactly before physics (zero position and orientation error), the full reset and all four camera checks passed, every state was finite and inside arm soft limits, every base-link/eef-frame identity check passed, no termination occurred, and the environment closed cleanly. This isolates a small steady-state controller residual rather than the frame bug corrected by R003.

Per the registered hard gate, diagnostics 2–4 and all four candidate pairs were not evaluated. Counts are 1 diagnostic, 0 candidates, 0 model requests, and 0 behavioral episodes. The R003 schedule, targets, tolerances, OOD/physics/camera/companion gates, and downstream behavioral design were not changed.
