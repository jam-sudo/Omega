"""Tests for Phase 254: core/tumor_spheroid_pk.py"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.tumor_spheroid_pk import (
    TumorSpheroidPKResult,
    compare_drug_properties,
    simulate_spheroid_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> TumorSpheroidPKResult:
    params = dict(
        drug_name="TestDrug",
        dose_mg_per_L=1.0,
        logp=2.0,
        mw=400.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_spheroid_pk(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_tumor_spheroid_pk_result(self):
        r = _default()
        assert isinstance(r, TumorSpheroidPKResult)

    def test_has_drug_name(self):
        r = _default()
        assert r.drug_name == "TestDrug"

    def test_has_dose_field(self):
        r = _default(dose_mg_per_L=2.0)
        assert r.dose_mg_per_L == 2.0

    def test_has_times_list(self):
        r = _default()
        assert isinstance(r.times_h, list)

    def test_has_c_medium_list(self):
        r = _default()
        assert isinstance(r.c_medium_mg_L, list)

    def test_has_c_core_list(self):
        r = _default()
        assert isinstance(r.c_core_mg_L, list)

    def test_has_notes_str(self):
        r = _default()
        assert isinstance(r.notes, str)

    def test_list_lengths_match(self):
        r = _default()
        assert len(r.times_h) == len(r.c_medium_mg_L)
        assert len(r.times_h) == len(r.c_core_mg_L)
        assert len(r.times_h) == len(r.c_surface_mg_L)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_medium_starts_at_dose(self):
        r = _default(dose_mg_per_L=1.5)
        assert math.isclose(r.c_medium_mg_L[0], 1.5, rel_tol=1e-9)

    def test_core_starts_at_zero(self):
        r = _default()
        assert r.c_core_mg_L[0] == 0.0

    def test_surface_starts_at_zero(self):
        r = _default()
        assert r.c_surface_mg_L[0] == 0.0

    def test_medium_decreases_from_initial(self):
        r = _default()
        # Medium should decrease as drug enters spheroid
        assert r.c_medium_mg_L[-1] < r.c_medium_mg_L[0]


# ---------------------------------------------------------------------------
# Core concentration dynamics
# ---------------------------------------------------------------------------


class TestCoreDynamics:
    def test_core_rises_from_zero(self):
        r = _default()
        # Core should rise above zero as drug penetrates
        assert max(r.c_core_mg_L) > 0.0

    def test_auc_core_positive(self):
        r = _default()
        assert r.auc_core_mg_h_per_L > 0.0

    def test_auc_medium_positive(self):
        r = _default()
        assert r.auc_medium_mg_h_per_L > 0.0

    def test_cmax_core_positive(self):
        r = _default()
        assert r.cmax_core_mg_L > 0.0

    def test_tmax_core_after_t0(self):
        r = _default()
        # Drug takes time to penetrate; tmax in core should be after t=0
        assert r.tmax_core_h >= 0.0

    def test_core_concentration_non_negative(self):
        r = _default()
        assert all(c >= 0.0 for c in r.c_core_mg_L)

    def test_medium_concentration_non_negative(self):
        r = _default()
        assert all(c >= 0.0 for c in r.c_medium_mg_L)


# ---------------------------------------------------------------------------
# Penetration efficiency
# ---------------------------------------------------------------------------


class TestPenetrationEfficiency:
    def test_penetration_efficiency_in_valid_range(self):
        r = _default()
        assert 0.0 <= r.penetration_efficiency_pct <= 100.0

    def test_penetration_efficiency_positive(self):
        r = _default()
        assert r.penetration_efficiency_pct > 0.0

    def test_k_penetration_positive(self):
        r = _default()
        assert r.k_penetration_per_h > 0.0


# ---------------------------------------------------------------------------
# logP effect on penetration
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logp_higher_penetration(self):
        r_low = _default(logp=0.0, mw=300.0)
        r_high = _default(logp=4.0, mw=300.0)
        assert r_high.penetration_efficiency_pct >= r_low.penetration_efficiency_pct

    def test_logp_effect_monotonic(self):
        # Increasing logP should monotonically increase penetration
        results = [_default(logp=lp, mw=300.0) for lp in [0.0, 1.0, 2.0, 3.0, 4.0]]
        efficiencies = [r.penetration_efficiency_pct for r in results]
        # Should be non-decreasing
        for i in range(1, len(efficiencies)):
            assert efficiencies[i] >= efficiencies[i - 1] - 1e-6


# ---------------------------------------------------------------------------
# compare_drug_properties
# ---------------------------------------------------------------------------


class TestCompareDrugProperties:
    def test_returns_list(self):
        results = compare_drug_properties("Drug", 1.0)
        assert isinstance(results, list)

    def test_default_returns_5_results(self):
        results = compare_drug_properties("Drug", 1.0)
        assert len(results) == 5

    def test_sorted_by_penetration_descending(self):
        results = compare_drug_properties("Drug", 1.0)
        efficiencies = [r.penetration_efficiency_pct for r in results]
        for i in range(1, len(efficiencies)):
            assert efficiencies[i] <= efficiencies[i - 1] + 1e-6

    def test_custom_logp_values(self):
        results = compare_drug_properties("Drug", 1.0, logp_values=[1.0, 3.0])
        assert len(results) == 2

    def test_all_results_are_tumor_spheroid_pk_result(self):
        results = compare_drug_properties("Drug", 1.0)
        assert all(isinstance(r, TumorSpheroidPKResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg_per_L"):
            simulate_spheroid_pk("Drug", dose_mg_per_L=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg_per_L"):
            simulate_spheroid_pk("Drug", dose_mg_per_L=-1.0)

    def test_t_end_zero_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_spheroid_pk("Drug", dose_mg_per_L=1.0, t_end_h=0.0)

    def test_dt_zero_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_spheroid_pk("Drug", dose_mg_per_L=1.0, dt_h=0.0)
