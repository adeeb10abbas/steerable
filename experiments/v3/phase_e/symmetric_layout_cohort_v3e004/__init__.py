"""V3-E004 symmetric-object-layout cohort.

The package is intentionally separate from V3-E003.  E003 remains immutable
evidence; this namespace implements the revised full-pose and visibility
gates used by E004.
"""

from .layout_contract import (  # noqa: F401
    ASYMMETRY_LEVELS,
    E004Candidate,
    LayoutContractError,
    PoseSE2,
    SymmetryWeights,
    build_candidate,
    evaluate_layout,
    interpolate_pose,
    load_candidate,
    pose_map_sha256,
)
