"""Tests for Phase 572 — Biphasic Drug Dissolution Model."""

import math

import pytest

from omega_pbpk.core.biphasic_dissolution import (
    BiphasicDissolutionResult,
    compare_fast_fractions,
    simulate_biphasic_dissolution,
)

BASE_PARAMS = dict(
    drug_name="DrugX",
    dose_mg=100.0,
    fast_fraction=0.6,
    k_fast_per_min=0.1,
    k_slow_per_min=0.01,
    t_end_min=120.0,
    dt_min=0.5,
)


def _run(**overrides):
    params = {**BASE_PARAMS, **overrides}
    return simulate_biphasic_dissolution(**params)


class TestReturnType:
    def test_returns_correct_type(self):
        r = _run()
        assert isinstance(r, BiphasicDissolutionResult)

    def test_list_lengths(self):
        r = _run()
        assert len(r.times_min) == len(r.dissolved_pct)

    def test_expected_points(self):
        r = _run(t_end_min=60.0, dt_min=0.5)
        assert len(r.times_min) == 121


class TestDissolutionProfile:
    def test_starts_at_zero(self):
        r = _run()
        assert r.dissolved_pct[0] == pytest.approx(0.0)

    def test_ends_less_than_or_equal_100(self):
        r = _run()
        assert r.dissolved_pct[-1] <= 100.0 + 1e-9

    def test_monotonically_increasing(self):
        r = _run()
        for i in range(1, len(r.dissolved_pct)):
            assert r.dissolved_pct[i] >= r.dissolved_pct[i - 1] - 1e-9

    def test_all_fast_release(self):
        r = _run(fast_fraction=1.0)
        assert r.slow_fraction == pytest.approx(0.0)
        # Should dissolve quickly
        assert r.t50_min < 20.0

    def test_all_slow_release(self):
        r = _run(fast_fraction=0.0)
        assert r.fast_fraction == 0.0
        # Should dissolve slowly
        assert r.t50_min > 30.0


class TestFastFractionEffect:
    def test_higher_fast_fraction_lower_t50(self):
        r_low = _run(fast_fraction=0.2)
        r_high = _run(fast_fraction=0.8)
        assert r_high.t50_min < r_low.t50_min

    def test_higher_fast_fraction_lower_t80(self):
        r_low = _run(fast_fraction=0.2)
        r_high = _run(fast_fraction=0.8)
        assert r_high.t80_min < r_low.t80_min


class TestTimepoints:
    def test_t50_less_than_t80(self):
        r = _run()
        assert r.t50_min <= r.t80_min

    def test_t80_less_than_t90(self):
        r = _run()
        assert r.t80_min <= r.t90_min

    def test_t90_can_be_inf(self):
        # Very slow dissolution: short time, low rates
        r = _run(k_fast_per_min=0.005, k_slow_per_min=0.001, t_end_min=30.0)
        assert r.t90_min == math.inf


class TestBiphasicIndex:
    def test_biphasic_index_less_than_one(self):
        r = _run()
        assert r.biphasic_index < 1.0

    def test_biphasic_index_value(self):
        r = _run(k_fast_per_min=0.1, k_slow_per_min=0.01)
        assert r.biphasic_index == pytest.approx(0.1)


class TestDissolutionClass:
    def test_rapid_class(self):
        # Very fast dissolution
        r = _run(fast_fraction=0.9, k_fast_per_min=0.5, k_slow_per_min=0.2)
        assert r.dissolution_class == "rapid"

    def test_slow_class(self):
        r = _run(fast_fraction=0.1, k_fast_per_min=0.01, k_slow_per_min=0.002, t_end_min=300.0)
        assert r.dissolution_class == "slow"

    def test_intermediate_class(self):
        r = _run(fast_fraction=0.5, k_fast_per_min=0.05, k_slow_per_min=0.01)
        assert r.dissolution_class == "intermediate"


class TestCompareFastFractions:
    def test_returns_list(self):
        results = compare_fast_fractions("DrugX", 100.0, 0.1, 0.01)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_t80_ascending(self):
        results = compare_fast_fractions("DrugX", 100.0, 0.1, 0.01)
        for i in range(1, len(results)):
            assert results[i].t80_min >= results[i - 1].t80_min

    def test_custom_fractions(self):
        results = compare_fast_fractions("DrugX", 100.0, 0.1, 0.01, fast_fractions=[0.3, 0.7])
        assert len(results) == 2


class TestValidation:
    def test_dose_positive(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_fast_fraction_lower_bound(self):
        with pytest.raises(ValueError):
            _run(fast_fraction=-0.1)

    def test_fast_fraction_upper_bound(self):
        with pytest.raises(ValueError):
            _run(fast_fraction=1.1)

    def test_k_fast_positive(self):
        with pytest.raises(ValueError):
            _run(k_fast_per_min=0)

    def test_k_slow_positive(self):
        with pytest.raises(ValueError):
            _run(k_slow_per_min=0)

    def test_k_slow_less_than_k_fast(self):
        with pytest.raises(ValueError):
            _run(k_fast_per_min=0.05, k_slow_per_min=0.05)
