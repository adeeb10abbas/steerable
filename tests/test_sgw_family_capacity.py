from copy import deepcopy
import json

import pytest

from experiments.workshops.spatial_grounding_v1.prospective_family_designs import _digest
from tools.audit_sgw_family_capacity import CAPTURE, INFRA, ROOT, audit, capacity_bound


@pytest.fixture
def plan():
    return json.loads((ROOT / INFRA / "family-plan-freeze-20260923bm/height-plan.json").read_bytes())


def side_ids(plan, side):
    return [row["design_id"] for row in plan["designs"] if row["side"] == side and row["status"] == CAPTURE]


def test_unknown_slots_are_potential_passes_not_failures(plan):
    bound = capacity_bound(plan, {})
    assert bound["status"] == "not_ruled_out_not_a_release"
    assert bound["by_side"]["left"]["maximum_qualified_possible"] == 28
    assert bound["by_side"]["right"]["maximum_qualified_possible"] == 29
    assert bound["by_side"]["left"]["required_qualified"] == 15
    assert bound["by_side"]["right"]["required_qualified"] == 14
    assert bound["release_permitted"] is False


def test_exact_right_capacity_is_not_blocked_but_one_more_failure_blocks(plan):
    ids = side_ids(plan, "right")
    outcomes = {key: False for key in ids[:15]}
    boundary = capacity_bound(plan, outcomes)
    assert boundary["by_side"]["right"]["maximum_qualified_possible"] == 14
    assert boundary["status"] == "not_ruled_out_not_a_release"
    outcomes[ids[15]] = False
    blocked = capacity_bound(plan, outcomes)
    assert blocked["status"] == "mathematically_blocked_by_frozen_stratum_capacity"
    assert blocked["by_side"]["right"]["unavoidable_shortfall"] == 1


def test_pilot_stratum_requires_fifteen_not_fourteen(plan):
    ids = side_ids(plan, "left")
    bound = capacity_bound(plan, {key: False for key in ids[:14]})
    assert bound["by_side"]["left"]["maximum_qualified_possible"] == 14
    assert bound["status"] == "mathematically_blocked_by_frozen_stratum_capacity"


def test_odd_seed_assigns_extra_pilot_to_right(plan):
    plan["seed"] += 1
    plan["plan_sha256"] = _digest(plan, "plan_sha256")
    bound = capacity_bound(plan, {})
    assert bound["pilot_side"] == "right"
    assert bound["by_side"]["right"]["required_qualified"] == 15
    assert bound["by_side"]["left"]["required_qualified"] == 14


def test_physical_passes_never_release_historical_or_runtime_gates(plan):
    outcomes = {key: True for key in plan["accepted_design_ids"]}
    result = capacity_bound(plan, outcomes)
    assert result["release_permitted"] is False
    assert result["status"] == "not_ruled_out_not_a_release"
    assert result["by_side"]["left"]["unresolved_slots"] == 0


@pytest.mark.parametrize("outcomes", [
    {"SGW-HEIGHT-DESIGN-002": False},
    {"unknown": True},
    {"SGW-HEIGHT-DESIGN-000": 0},
    {"SGW-HEIGHT-DESIGN-000": None},
])
def test_invalid_or_geometrically_rejected_outcomes_cannot_enter_bound(plan, outcomes):
    with pytest.raises(ValueError, match="terminal outcomes"):
        capacity_bound(plan, outcomes)


@pytest.mark.parametrize(("key", "value"), [("seed", True), ("family", "LAT"), ("design_slot_count", 99)])
def test_nonregistered_plan_is_rejected(plan, key, value):
    plan[key] = value
    plan["plan_sha256"] = _digest(plan, "plan_sha256")
    with pytest.raises(ValueError):
        capacity_bound(plan, {})


def test_unknown_proposal_status_is_not_silently_a_geometric_rejection(plan):
    plan["designs"][2]["status"] = "infrastructure_invalid"
    plan["plan_sha256"] = _digest(plan, "plan_sha256")
    with pytest.raises(ValueError, match="proposal status"):
        capacity_bound(plan, {})


def prefix_paths():
    return [ROOT / INFRA / f"family-partition-20260923bt-prefix-{suffix}/manifest.json"
            for suffix in ("bv", "bw", "bx", "bz", "ca", "cb", "cc", "cd")]


def test_retained_eighth_wave_and_smoke_reproduce_exact_capacity():
    result = audit(prefix_paths())
    assert len(result["terminal_evidence"]) == 36
    height = result["families"]["HEIGHT"]["by_side"]
    assert (height["left"]["physical_accepted"], height["left"]["physical_rejected"]) == (11, 7)
    assert (height["right"]["physical_accepted"], height["right"]["physical_rejected"]) == (1, 15)
    assert (height["left"]["maximum_qualified_possible"], height["right"]["maximum_qualified_possible"]) == (21, 14)
    dist = result["families"]["DIST"]["by_side"]
    assert (dist["left"]["maximum_qualified_possible"], dist["right"]["maximum_qualified_possible"]) == (45, 18)
    assert result["fixture_release_permitted"] is result["model_release_permitted"] is False


def test_duplicate_prefix_does_not_double_count_failures():
    with pytest.raises(ValueError, match="duplicate terminal"):
        audit(prefix_paths() + prefix_paths()[:1])


def test_ninth_prefix_proves_height_block_without_releasing_dist():
    result = audit(prefix_paths() + [
        ROOT / INFRA / "family-partition-20260923bt-prefix-ce/manifest.json",
    ])
    assert len(result["terminal_evidence"]) == 38
    height = result["families"]["HEIGHT"]
    assert height["status"] == "mathematically_blocked_by_frozen_stratum_capacity"
    assert height["by_side"]["right"]["maximum_qualified_possible"] == 12
    assert height["by_side"]["right"]["unavoidable_shortfall"] == 2
    assert result["families"]["DIST"]["status"] == "not_ruled_out_not_a_release"
    assert result["fixture_release_permitted"] is result["model_release_permitted"] is False
    assert result["live_workers_modified"] is False


def test_tenth_prefix_also_exhausts_total_height_capacity():
    result = audit(prefix_paths() + [
        ROOT / INFRA / f"family-partition-20260923bt-prefix-{suffix}/manifest.json"
        for suffix in ("ce", "cf")
    ])
    assert len(result["terminal_evidence"]) == 44
    height = result["families"]["HEIGHT"]["by_side"]
    assert sum(row["maximum_qualified_possible"] for row in height.values()) == 27
    assert sum(row["required_qualified"] for row in height.values()) == 29
    assert height["right"]["maximum_qualified_possible"] == 8
    assert sum(row["accepted"] for row in result["terminal_evidence"]) == 13


def test_changed_projection_cannot_reclassify_a_valid_outcome(monkeypatch):
    import tools.audit_sgw_family_capacity as module

    load = module._load

    def altered(path):
        value = load(path)
        if path == prefix_paths()[-1]:
            value = deepcopy(value)
            value["records"][0]["trials"][0]["passed"] = False
        return value

    monkeypatch.setattr(module, "_load", altered)
    with pytest.raises(ValueError, match="projected full-trial"):
        audit(prefix_paths())
