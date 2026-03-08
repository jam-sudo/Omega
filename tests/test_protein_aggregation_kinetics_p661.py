"""Tests for protein aggregation kinetics model (phase 661)."""

import pytest

from omega_pbpk.core.protein_aggregation_kinetics_p661 import (
    ProteinAggregationKineticsResult,
    simulate_aggregation_kinetics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = dict(
    drug_name="TestMAb",
    initial_concentration_mg_mL=10.0,
    nucleation_rate_per_h=1e-4,
    elongation_rate_per_mL_mg_h=0.01,
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_aggregation_kinetics(**params)


# ---------------------------------------------------------------------------
# Basic return-type & structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, ProteinAggregationKineticsResult)

    def test_drug_name_stored(self):
        r = _run()
        assert r.drug_name == "TestMAb"

    def test_dose_stored(self):
        r = _run()
        assert r.dose_mg_mL == 10.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)

    def test_list_lengths_match(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_monomer_mg_mL) == n
        assert len(r.c_aggregate_mg_mL) == n
        assert len(r.c_nucleus_mg_mL) == n
        assert len(r.monomer_fraction) == n


# ---------------------------------------------------------------------------
# Non-negativity
# ---------------------------------------------------------------------------


class TestNonNegativity:
    def test_c_monomer_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_monomer_mg_mL)

    def test_c_aggregate_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_aggregate_mg_mL)

    def test_c_nucleus_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_nucleus_mg_mL)

    def test_monomer_fraction_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.monomer_fraction)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_monomer_fraction_starts_at_one(self):
        r = _run()
        assert abs(r.monomer_fraction[0] - 1.0) < 1e-10

    def test_aggregate_starts_at_zero(self):
        r = _run()
        assert r.c_aggregate_mg_mL[0] == 0.0

    def test_nucleus_starts_at_zero(self):
        r = _run()
        assert r.c_nucleus_mg_mL[0] == 0.0


# ---------------------------------------------------------------------------
# Aggregation dynamics
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_monomer_generally_decreases(self):
        r = _run(
            nucleation_rate_per_h=1e-3,
            elongation_rate_per_mL_mg_h=0.05,
            t_end_h=200.0,
        )
        # monomer at end should be less than at start
        assert r.c_monomer_mg_mL[-1] < r.c_monomer_mg_mL[0]

    def test_aggregate_increases(self):
        r = _run(
            nucleation_rate_per_h=1e-3,
            elongation_rate_per_mL_mg_h=0.05,
            t_end_h=200.0,
        )
        assert r.c_aggregate_mg_mL[-1] > r.c_aggregate_mg_mL[0]

    def test_t_half_greater_than_lag(self):
        r = _run(
            nucleation_rate_per_h=1e-3,
            elongation_rate_per_mL_mg_h=0.05,
            t_end_h=500.0,
        )
        assert r.t_half_aggregation_h >= r.lag_time_h

    def test_aggregation_rate_positive(self):
        r = _run(
            nucleation_rate_per_h=1e-3,
            elongation_rate_per_mL_mg_h=0.05,
            t_end_h=200.0,
        )
        assert r.aggregation_rate_per_h > 0


# ---------------------------------------------------------------------------
# Rate effects
# ---------------------------------------------------------------------------


class TestRateEffects:
    def test_higher_nucleation_faster_aggregation(self):
        r_slow = _run(
            nucleation_rate_per_h=1e-5,
            elongation_rate_per_mL_mg_h=0.01,
            t_end_h=300.0,
        )
        r_fast = _run(
            nucleation_rate_per_h=5e-5,
            elongation_rate_per_mL_mg_h=0.01,
            t_end_h=300.0,
        )
        # faster nucleation → shorter lag time (earlier onset)
        assert r_fast.lag_time_h < r_slow.lag_time_h

    def test_higher_temperature_faster(self):
        r_cold = _run(
            nucleation_rate_per_h=1e-4,
            elongation_rate_per_mL_mg_h=0.005,
            temperature_C=4.0,
            t_end_h=300.0,
        )
        r_hot = _run(
            nucleation_rate_per_h=1e-4,
            elongation_rate_per_mL_mg_h=0.005,
            temperature_C=50.0,
            t_end_h=300.0,
        )
        # higher temperature → lower monomer fraction at end
        assert r_hot.monomer_fraction[-1] < r_cold.monomer_fraction[-1]

    def test_extreme_ph_faster(self):
        r_neutral = _run(
            nucleation_rate_per_h=1e-4,
            elongation_rate_per_mL_mg_h=0.005,
            ph=7.4,
            t_end_h=300.0,
        )
        r_acid = _run(
            nucleation_rate_per_h=1e-4,
            elongation_rate_per_mL_mg_h=0.005,
            ph=3.0,
            t_end_h=300.0,
        )
        # extreme pH → lower monomer fraction at end
        assert r_acid.monomer_fraction[-1] < r_neutral.monomer_fraction[-1]


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


class TestStabilityClass:
    def test_stable_classification(self):
        r = _run(
            nucleation_rate_per_h=1e-8,
            elongation_rate_per_mL_mg_h=1e-5,
            t_end_h=24.0,
        )
        assert r.stability_class == "stable"

    def test_unstable_classification(self):
        r = _run(
            initial_concentration_mg_mL=2.0,
            nucleation_rate_per_h=0.005,
            elongation_rate_per_mL_mg_h=0.5,
            t_end_h=500.0,
            dt_h=0.1,
        )
        assert r.stability_class == "unstable"

    def test_stability_values(self):
        r = _run()
        assert r.stability_class in {"stable", "borderline", "unstable"}


# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------


class TestMassConservation:
    def test_mass_approximately_conserved(self):
        r = _run(t_end_h=100.0)
        c0 = r.dose_mg_mL
        for i in range(len(r.times_h)):
            total = r.c_monomer_mg_mL[i] + r.c_nucleus_mg_mL[i] + r.c_aggregate_mg_mL[i]
            assert total <= c0 * 1.1, f"Mass exceeded at t={r.times_h[i]}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_concentration_raises(self):
        with pytest.raises(ValueError, match="initial_concentration"):
            _run(initial_concentration_mg_mL=0.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="initial_concentration"):
            _run(initial_concentration_mg_mL=-1.0)

    def test_zero_nucleation_raises(self):
        with pytest.raises(ValueError, match="nucleation_rate"):
            _run(nucleation_rate_per_h=0.0)

    def test_negative_elongation_raises(self):
        with pytest.raises(ValueError, match="elongation_rate"):
            _run(elongation_rate_per_mL_mg_h=-0.1)

    def test_nucleus_size_too_small_raises(self):
        with pytest.raises(ValueError, match="critical_nucleus_size"):
            _run(critical_nucleus_size=1)
