from experiments.v3.pi05_stochastic_probe import SEED_INDICES, evaluate_sample_hashes


def _hashes(*, distinct: bool = True, broken_repeat: bool = False):
    result = {}
    for relation in ("left", "right"):
        result[relation] = {}
        for seed in SEED_INDICES:
            value = f"{relation}-{seed}" if distinct else relation
            result[relation][seed] = [value, value]
    if broken_repeat:
        result["left"][3][1] = "unexpected-repeat"
    return result


def test_registered_effective_seed_rule_passes_distinct_exact_repeats() -> None:
    metrics, passed = evaluate_sample_hashes(_hashes())
    assert passed
    assert metrics["left"]["unique_raw_policy_samples_across_seed_indices"] == 8
    assert metrics["right"]["all_exact_repeats_bit_identical"]


def test_registered_effective_seed_rule_rejects_ignored_seed() -> None:
    metrics, passed = evaluate_sample_hashes(_hashes(distinct=False))
    assert not passed
    assert not metrics["left"]["at_least_two_seed_indices_bitwise_distinct"]


def test_registered_effective_seed_rule_rejects_unstable_repeat() -> None:
    metrics, passed = evaluate_sample_hashes(_hashes(broken_repeat=True))
    assert not passed
    assert not metrics["left"]["all_exact_repeats_bit_identical"]

