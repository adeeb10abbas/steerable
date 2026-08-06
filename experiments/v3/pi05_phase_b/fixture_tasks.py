"""V3-B002 fixture/task primitives, inherited byte-for-byte from Nano B001.

The physical fixture implementation is intentionally not duplicated.  The
hash-pinned Nano module is the registered implementation of reflecting only
the three movable-object centers about the robot sagittal plane.  B002 changes
the policy checkpoint, not the fixture or task predicate.
"""

from experiments.v3.cosmos_nano_phase_b.fixture_tasks import (  # noqa: F401
    LEFT_PROMPT,
    RIGHT_PROMPT,
    _LeftTermination,
    _RightTermination,
    _scene,
    _subtask,
)
