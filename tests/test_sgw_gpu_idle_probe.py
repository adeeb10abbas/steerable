import pytest

from experiments.workshops.spatial_grounding_v1.gpu_idle_probe import select_idle


def gpu(index, used=100, utilization=0):
    return {"index": index, "uuid": f"gpu-{index}", "memory_used_mib": used,
            "memory_free_mib": 97000 - used, "utilization_percent": utilization}


def test_choose_only_idle_allocated_gpu():
    values = [gpu(0, 94000), gpu(1), gpu(2), gpu(3, utilization=100)]
    assert select_idle(values, {"gpu-1"}, 4)["uuid"] == "gpu-2"


def test_no_idle_gpu_blocks():
    with pytest.raises(RuntimeError, match="No allocated GPU"):
        select_idle([gpu(0, 94000)], set(), 1)


def test_allocation_mismatch_blocks():
    with pytest.raises(ValueError, match="count differs"):
        select_idle([gpu(0), gpu(1)], set(), 1)
