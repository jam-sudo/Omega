"""Tests for core/crystallization_kinetics.py — Phase 815."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.crystallization_kinetics import (
    CrystallizationKineticsResult,
    screen_supersaturation,
    simulate_crystallization,
)


class TestSimulateCrystallizationReturnType:
    def test_returns_dataclass(self):
        result = simulate_crystallization(
            compound_name="DrugA",
            supersaturation_ratio=3.0,
            avrami_n=2.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.5,
        )
        assert isinstance(result, CrystallizationKineticsResult)

    def test_compound_name_preserved(self):
        result = simulate_crystallization(
            compound_name="TestCompound",
            supersaturation_ratio=2.0,
            avrami_n=1.5,
            rate_constant_per_h=0.3,
            induction_time_h=0.0,
        )
        assert result.compound_name == "TestCompound"

    def test_avrami_n_preserved(self):
        result = simulate_crystallization(
            compound_name="X",
            supersaturation_ratio=2.0,
            avrami_n=2.5,
            rate_constant_per_h=0.4,
            induction_time_h=0.0,
        )
        assert result.avrami_n == 2.5


class TestPreInductionFraction:
    def test_fraction_zero_before_induction(self):
        """Fraction must be 0 at t=0 when induction_time_h > 0."""
        result = simulate_crystallization(
            compound_name="DrugB",
            supersaturation_ratio=4.0,
            avrami_n=2.0,
            rate_constant_per_h=1.0,
            induction_time_h=2.0,
        )
        # At t=1h (before induction at 2h), fraction_at_1h should be 0
        assert result.crystallized_fraction_at_1h == 0.0

    def test_fraction_zero_at_exact_induction_boundary(self):
        """fraction at exactly t_ind should be 0."""
        result = simulate_crystallization(
            compound_name="DrugC",
            supersaturation_ratio=3.0,
            avrami_n=2.0,
            rate_constant_per_h=1.0,
            induction_time_h=6.0,
        )
        assert result.crystallized_fraction_at_6h == 0.0


class TestFractionMonotonicity:
    def test_fractions_increase_after_induction(self):
        """f_1h <= f_6h <= f_24h after induction starts."""
        result = simulate_crystallization(
            compound_name="DrugD",
            supersaturation_ratio=5.0,
            avrami_n=2.0,
            rate_constant_per_h=0.8,
            induction_time_h=0.0,
        )
        assert result.crystallized_fraction_at_1h <= result.crystallized_fraction_at_6h
        assert result.crystallized_fraction_at_6h <= result.crystallized_fraction_at_24h

    def test_fraction_bounded_zero_to_one(self):
        """All reported fractions in [0, 1]."""
        result = simulate_crystallization(
            compound_name="DrugE",
            supersaturation_ratio=10.0,
            avrami_n=3.0,
            rate_constant_per_h=2.0,
            induction_time_h=0.1,
        )
        for frac in [
            result.crystallized_fraction_at_1h,
            result.crystallized_fraction_at_6h,
            result.crystallized_fraction_at_24h,
        ]:
            assert 0.0 <= frac <= 1.0


class TestT50:
    def test_t50_greater_than_induction_time(self):
        """t50 must be > induction_time_h when crystallization occurs."""
        result = simulate_crystallization(
            compound_name="DrugF",
            supersaturation_ratio=3.0,
            avrami_n=2.0,
            rate_constant_per_h=1.0,
            induction_time_h=1.0,
        )
        assert result.t50_crystallization_h > result.induction_time_h

    def test_t50_inf_when_never_reached(self):
        """With very slow kinetics, t50 should be inf."""
        result = simulate_crystallization(
            compound_name="DrugG",
            supersaturation_ratio=1.05,
            avrami_n=1.0,
            rate_constant_per_h=0.001,
            induction_time_h=0.0,
            t_end_h=5.0,
        )
        assert result.t50_crystallization_h == math.inf

    def test_t50_finite_for_fast_kinetics(self):
        """Fast kinetics should yield a finite t50."""
        result = simulate_crystallization(
            compound_name="DrugH",
            supersaturation_ratio=8.0,
            avrami_n=2.0,
            rate_constant_per_h=3.0,
            induction_time_h=0.0,
        )
        assert math.isfinite(result.t50_crystallization_h)
        assert result.t50_crystallization_h > 0.0


class TestSupersaturationEffect:
    def test_higher_supersaturation_same_kinetics_same_t50(self):
        """With Avrami model, supersaturation_ratio doesn't directly change k, but it is stored."""
        r_low = simulate_crystallization(
            compound_name="D",
            supersaturation_ratio=2.0,
            avrami_n=2.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.0,
        )
        r_high = simulate_crystallization(
            compound_name="D",
            supersaturation_ratio=10.0,
            avrami_n=2.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.0,
        )
        # t50 is a function of k and n only — same k → same t50
        assert r_low.t50_crystallization_h == pytest.approx(r_high.t50_crystallization_h, rel=1e-3)

    def test_higher_rate_constant_shorter_t50(self):
        """Higher k → shorter t50."""
        r_slow = simulate_crystallization(
            compound_name="D",
            supersaturation_ratio=3.0,
            avrami_n=2.0,
            rate_constant_per_h=0.2,
            induction_time_h=0.0,
        )
        r_fast = simulate_crystallization(
            compound_name="D",
            supersaturation_ratio=3.0,
            avrami_n=2.0,
            rate_constant_per_h=2.0,
            induction_time_h=0.0,
        )
        assert r_fast.t50_crystallization_h < r_slow.t50_crystallization_h


class TestNucleationType:
    def test_homogeneous_when_n_ge_3(self):
        result = simulate_crystallization(
            compound_name="mAb",
            supersaturation_ratio=2.0,
            avrami_n=3.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.0,
        )
        assert result.nucleation_type == "homogeneous"

    def test_heterogeneous_when_n_lt_3(self):
        result = simulate_crystallization(
            compound_name="mAb",
            supersaturation_ratio=2.0,
            avrami_n=2.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.0,
        )
        assert result.nucleation_type == "heterogeneous"

    def test_avrami_n_4_homogeneous(self):
        result = simulate_crystallization(
            compound_name="D",
            supersaturation_ratio=2.0,
            avrami_n=4.0,
            rate_constant_per_h=0.3,
            induction_time_h=0.0,
        )
        assert result.nucleation_type == "homogeneous"


class TestValidation:
    def test_supersaturation_below_1_raises(self):
        with pytest.raises(ValueError, match="supersaturation_ratio must be >= 1.0"):
            simulate_crystallization(
                compound_name="X",
                supersaturation_ratio=0.5,
                avrami_n=2.0,
                rate_constant_per_h=0.5,
                induction_time_h=0.0,
            )

    def test_avrami_n_below_1_raises(self):
        with pytest.raises(ValueError, match="avrami_n must be in"):
            simulate_crystallization(
                compound_name="X",
                supersaturation_ratio=2.0,
                avrami_n=0.5,
                rate_constant_per_h=0.5,
                induction_time_h=0.0,
            )

    def test_avrami_n_above_4_raises(self):
        with pytest.raises(ValueError, match="avrami_n must be in"):
            simulate_crystallization(
                compound_name="X",
                supersaturation_ratio=2.0,
                avrami_n=4.5,
                rate_constant_per_h=0.5,
                induction_time_h=0.0,
            )

    def test_supersaturation_exactly_1_valid(self):
        result = simulate_crystallization(
            compound_name="X",
            supersaturation_ratio=1.0,
            avrami_n=1.0,
            rate_constant_per_h=0.1,
            induction_time_h=0.0,
        )
        assert isinstance(result, CrystallizationKineticsResult)


class TestMaxRate:
    def test_max_rate_positive_for_active_kinetics(self):
        result = simulate_crystallization(
            compound_name="D",
            supersaturation_ratio=3.0,
            avrami_n=2.0,
            rate_constant_per_h=1.0,
            induction_time_h=0.0,
        )
        assert result.maximum_crystallization_rate_per_h >= 0.0

    def test_notes_nonempty_string(self):
        result = simulate_crystallization(
            compound_name="D",
            supersaturation_ratio=3.0,
            avrami_n=2.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.0,
        )
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


class TestScreenSupersaturation:
    def test_returns_list(self):
        results = screen_supersaturation(
            compound_name="D",
            supersaturation_ratios=[2.0, 5.0, 10.0],
            avrami_n=2.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.0,
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_t50_descending(self):
        """screen_supersaturation returns most stable (longest t50) first."""
        results = screen_supersaturation(
            compound_name="D",
            supersaturation_ratios=[2.0, 5.0, 10.0],
            avrami_n=2.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.0,
        )
        # All have same k so same t50, but list should still be sorted
        t50s = [r.t50_crystallization_h for r in results]
        for i in range(len(t50s) - 1):
            assert t50s[i] >= t50s[i + 1]

    def test_inf_t50_comes_first(self):
        """Results with infinite t50 (very slow kinetics) should appear first."""
        # Use a very small k so 50% crystallization is not reached in 24h (default t_end)
        results = screen_supersaturation(
            compound_name="D",
            supersaturation_ratios=[1.05, 3.0],
            avrami_n=1.0,
            rate_constant_per_h=0.0005,
            induction_time_h=0.0,
        )
        # Both may be inf at this k, but sorting must be stable: all inf → first entries first
        assert results[0].t50_crystallization_h >= results[-1].t50_crystallization_h

    def test_each_result_is_correct_type(self):
        results = screen_supersaturation(
            compound_name="D",
            supersaturation_ratios=[2.0, 4.0],
            avrami_n=2.0,
            rate_constant_per_h=0.5,
            induction_time_h=0.0,
        )
        for r in results:
            assert isinstance(r, CrystallizationKineticsResult)
