"""Tests for Phase 475 — Cutaneous Drug Permeation."""

import math

import pytest

from omega_pbpk.core.cutaneous_permeation import (
    CutaneousPermeationResult,
    _compute_kp,
    compare_formulation_sites,
    simulate_cutaneous_permeation,
)

# ---------------------------------------------------------------------------
# Helper: default simulation
# ---------------------------------------------------------------------------
DRUG = "TestDrug"
DOSE = 10.0
MW = 300.0
LOGP = 2.0


def _run(**kwargs):
    defaults = dict(drug_name=DRUG, dose_mg=DOSE, mw_Da=MW, logP=LOGP)
    defaults.update(kwargs)
    return simulate_cutaneous_permeation(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_correct_type(self):
        result = _run()
        assert isinstance(result, CutaneousPermeationResult)

    def test_drug_name_stored(self):
        result = _run()
        assert result.drug_name == DRUG

    def test_dose_stored(self):
        result = _run()
        assert result.dose_mg == DOSE

    def test_logp_stored(self):
        result = _run()
        assert result.logP == LOGP


# ---------------------------------------------------------------------------
# Kp
# ---------------------------------------------------------------------------
class TestKp:
    def test_kp_positive(self):
        result = _run()
        assert result.kp_cm_per_h > 0.0

    def test_higher_logp_higher_kp(self):
        r_low = _run(logP=1.0)
        r_high = _run(logP=4.0)
        assert r_high.kp_cm_per_h > r_low.kp_cm_per_h

    def test_higher_mw_lower_kp(self):
        r_small = _run(mw_Da=100.0)
        r_large = _run(mw_Da=500.0)
        assert r_small.kp_cm_per_h > r_large.kp_cm_per_h

    def test_kp_helper_function(self):
        kp = _compute_kp(logP=2.0, mw_Da=300.0)
        assert kp > 0.0
        expected_log = -2.7 + 0.71 * 2.0 - 0.0061 * 300.0
        assert math.isclose(kp, 10**expected_log, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Absorption totals
# ---------------------------------------------------------------------------
class TestAbsorption:
    def test_total_absorbed_positive(self):
        result = _run()
        assert result.total_absorbed_mg > 0.0

    def test_total_absorbed_not_exceed_dose(self):
        result = _run()
        assert result.total_absorbed_mg <= result.dose_mg * 1.001  # allow tiny float error

    def test_bioavailability_in_range(self):
        result = _run()
        assert 0.0 <= result.bioavailability_pct <= 100.0

    def test_larger_area_more_absorption(self):
        r_small = _run(area_cm2=50.0)
        r_large = _run(area_cm2=200.0)
        # More area → same dose diffuses faster → higher cumulative in 24 h
        assert r_large.total_absorbed_mg >= r_small.total_absorbed_mg

    def test_24h_significant_absorption(self):
        # Lipophilic drug should show measurable absorption
        result = _run(logP=3.0, mw_Da=200.0, t_end_h=24.0)
        assert result.total_absorbed_mg > 0.01


# ---------------------------------------------------------------------------
# Time-series validity
# ---------------------------------------------------------------------------
class TestTimeSeries:
    def test_times_start_at_zero(self):
        result = _run()
        assert result.times_h[0] == 0.0

    def test_times_end_approximately_at_t_end(self):
        result = _run(t_end_h=12.0, dt_h=0.1)
        assert abs(result.times_h[-1] - 12.0) <= 0.15

    def test_cumulative_absorbed_monotonically_increasing(self):
        result = _run()
        cumul = result.cumulative_absorbed_mg
        for i in range(1, len(cumul)):
            assert cumul[i] >= cumul[i - 1] - 1e-12

    def test_flux_non_negative(self):
        result = _run()
        for f in result.flux_mg_per_cm2_per_h:
            assert f >= 0.0

    def test_c_sc_non_negative(self):
        result = _run()
        for c in result.c_sc_mg_per_cm2:
            assert c >= 0.0

    def test_flux_decreases_over_time(self):
        # SC depletes, so flux should decrease overall
        result = _run(t_end_h=24.0, dt_h=0.1)
        # First half average flux > second half average
        n = len(result.flux_mg_per_cm2_per_h)
        first_half = sum(result.flux_mg_per_cm2_per_h[: n // 2]) / (n // 2)
        second_half = sum(result.flux_mg_per_cm2_per_h[n // 2 :]) / (n - n // 2)
        assert first_half >= second_half


# ---------------------------------------------------------------------------
# Lag time
# ---------------------------------------------------------------------------
class TestLagTime:
    def test_t_lag_non_negative(self):
        result = _run()
        assert result.t_lag_h >= 0.0

    def test_t_lag_within_simulation(self):
        result = _run(t_end_h=24.0)
        assert result.t_lag_h <= 24.0


# ---------------------------------------------------------------------------
# compare_formulation_sites
# ---------------------------------------------------------------------------
class TestCompareFormulationSites:
    def test_returns_list(self):
        results = compare_formulation_sites(DRUG, DOSE, MW, LOGP, [50.0, 100.0, 200.0])
        assert isinstance(results, list)

    def test_correct_length(self):
        results = compare_formulation_sites(DRUG, DOSE, MW, LOGP, [50.0, 100.0])
        assert len(results) == 2

    def test_sorted_descending_by_absorbed(self):
        results = compare_formulation_sites(DRUG, DOSE, MW, LOGP, [50.0, 100.0, 200.0])
        absorbed = [r.total_absorbed_mg for r in results]
        assert absorbed == sorted(absorbed, reverse=True)

    def test_all_results_correct_type(self):
        results = compare_formulation_sites(DRUG, DOSE, MW, LOGP, [100.0])
        for r in results:
            assert isinstance(r, CutaneousPermeationResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_cutaneous_permeation(DRUG, -1.0, MW, LOGP)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_cutaneous_permeation(DRUG, 0.0, MW, LOGP)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_cutaneous_permeation(DRUG, DOSE, -100.0, LOGP)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_cutaneous_permeation(DRUG, DOSE, 0.0, LOGP)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="area_cm2"):
            simulate_cutaneous_permeation(DRUG, DOSE, MW, LOGP, area_cm2=-10.0)

    def test_zero_area_raises(self):
        with pytest.raises(ValueError, match="area_cm2"):
            simulate_cutaneous_permeation(DRUG, DOSE, MW, LOGP, area_cm2=0.0)
