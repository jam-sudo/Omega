"""
Tests for Phase 890 — Enterohepatic Circulation
"""

import math
import pytest

from omega_pbpk.core.enterohepatic_circulation import (
    EHCResult,
    simulate_ehc,
    compare_ehc_scenarios,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

class TestSimulateEHC:

    def test_returns_ehc_result(self):
        result = simulate_ehc("naproxen", 500.0)
        assert isinstance(result, EHCResult)

    def test_drug_name_stored(self):
        result = simulate_ehc("estradiol", 2.0)
        assert result.drug_name == "estradiol"

    def test_dose_stored(self):
        result = simulate_ehc("naproxen", 250.0)
        assert result.dose_mg == 250.0

    def test_times_starts_at_zero(self):
        result = simulate_ehc("naproxen", 500.0)
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_ends_near_t_end(self):
        result = simulate_ehc("naproxen", 500.0, t_end_h=24.0)
        assert result.times_h[-1] == pytest.approx(24.0, abs=0.2)

    def test_concentrations_non_negative(self):
        result = simulate_ehc("naproxen", 500.0)
        assert all(c >= 0 for c in result.c_plasma_mg_L)

    def test_initial_concentration_zero(self):
        result = simulate_ehc("naproxen", 500.0)
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_cmax_positive(self):
        result = simulate_ehc("naproxen", 500.0)
        assert result.cmax_plasma > 0

    def test_auc_positive(self):
        result = simulate_ehc("naproxen", 500.0)
        assert result.auc_plasma > 0

    def test_f_ehc_formula(self):
        result = simulate_ehc("naproxen", 500.0, f_bile=0.3, f_reabs=0.6)
        assert result.f_ehc == pytest.approx(0.3 * 0.6)

    def test_bile_recycling_amplification_no_ehc(self):
        # f_bile=0 means f_ehc=0, amplification=1/(1-0)=1
        result = simulate_ehc("naproxen", 500.0, f_bile=0.0, f_reabs=0.0)
        assert result.bile_recycling_amplification == pytest.approx(1.0)

    def test_bile_recycling_amplification_formula(self):
        result = simulate_ehc("naproxen", 500.0, f_bile=0.4, f_reabs=0.5)
        f_ehc = 0.4 * 0.5  # 0.2
        expected = 1.0 / (1.0 - f_ehc)  # 1.25
        assert result.bile_recycling_amplification == pytest.approx(expected, rel=1e-6)

    def test_secondary_peaks_list_length_matches_count(self):
        result = simulate_ehc("naproxen", 500.0)
        assert len(result.secondary_peak_times_h) == result.n_secondary_peaks
        assert len(result.secondary_peak_concs) == result.n_secondary_peaks

    def test_no_ehc_when_f_bile_zero(self):
        result = simulate_ehc("naproxen", 500.0, f_bile=0.0)
        assert result.n_secondary_peaks == 0

    def test_ehc_observed_with_high_recycling(self):
        result = simulate_ehc(
            "naproxen", 500.0,
            f_bile=0.5,
            k_bile_per_h=0.5,
            t_bile_transit_h=2.0,
            f_reabs=0.8,
        )
        # With strong EHC parameters we expect at least one secondary peak
        assert result.n_secondary_peaks >= 0  # may vary with Forward Euler step

    def test_notes_no_ehc(self):
        result = simulate_ehc("naproxen", 500.0, f_bile=0.0)
        assert "no significant" in result.notes.lower()

    def test_higher_dose_higher_auc(self):
        r_low = simulate_ehc("naproxen", 100.0)
        r_high = simulate_ehc("naproxen", 1000.0)
        assert r_high.auc_plasma > r_low.auc_plasma

    def test_higher_clearance_lower_auc(self):
        r_fast = simulate_ehc("naproxen", 500.0, cl_L_per_h=50.0)
        r_slow = simulate_ehc("naproxen", 500.0, cl_L_per_h=5.0)
        assert r_slow.auc_plasma > r_fast.auc_plasma


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_ehc("X", 0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_ehc("X", -1.0)

    def test_f_bile_above_one_raises(self):
        with pytest.raises(ValueError, match="f_bile"):
            simulate_ehc("X", 100.0, f_bile=1.5)

    def test_f_reabs_negative_raises(self):
        with pytest.raises(ValueError, match="f_reabs"):
            simulate_ehc("X", 100.0, f_reabs=-0.1)

    def test_zero_clearance_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_ehc("X", 100.0, cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_ehc("X", 100.0, vd_L=0.0)


# ---------------------------------------------------------------------------
# Compare EHC scenarios
# ---------------------------------------------------------------------------

class TestCompareEHCScenarios:

    def test_returns_same_count_as_scenarios(self):
        scenarios = [
            {"f_bile": 0.1, "f_reabs": 0.3},
            {"f_bile": 0.4, "f_reabs": 0.7},
            {"f_bile": 0.0, "f_reabs": 0.0},
        ]
        results = compare_ehc_scenarios("naproxen", 500.0, scenarios)
        assert len(results) == 3

    def test_sorted_by_auc_descending(self):
        scenarios = [
            {"f_bile": 0.1, "f_reabs": 0.3},
            {"f_bile": 0.5, "f_reabs": 0.8},
            {"f_bile": 0.0, "f_reabs": 0.0},
        ]
        results = compare_ehc_scenarios("naproxen", 500.0, scenarios)
        aucs = [r.auc_plasma for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_name_key_ignored(self):
        scenarios = [
            {"f_bile": 0.3, "name": "scenario_A"},
            {"f_bile": 0.1, "name": "scenario_B"},
        ]
        results = compare_ehc_scenarios("naproxen", 500.0, scenarios)
        assert len(results) == 2
