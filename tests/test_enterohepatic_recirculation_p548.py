"""Tests for Phase 548 - Enterohepatic Recirculation PK Model."""

import pytest

from omega_pbpk.core.enterohepatic_recirculation_p548 import (
    EHCResult,
    simulate_ehc,
)

# Common parameters for a typical drug
BASE_PARAMS = dict(
    drug_name="testdrug",
    dose_mg=100.0,
    ka_per_h=1.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    f_biliary=0.3,
    bile_release_h=4.0,
    t_end_h=24.0,
    dt_h=0.05,
)


class TestSimulateEHC:
    """Tests for simulate_ehc function."""

    def test_returns_ehc_result(self):
        """Should return an EHCResult."""
        result = simulate_ehc(**BASE_PARAMS)
        assert isinstance(result, EHCResult)

    def test_times_start_at_zero(self):
        """Time series should start at 0."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.times_h[0] == 0.0

    def test_concentrations_start_at_zero(self):
        """Initial concentration should be 0."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.c_systemic_mg_L[0] == 0.0

    def test_times_and_concentrations_same_length(self):
        """Times and concentrations should have equal length."""
        result = simulate_ehc(**BASE_PARAMS)
        assert len(result.times_h) == len(result.c_systemic_mg_L)

    def test_primary_peak_exists(self):
        """Should find a primary peak before bile release."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.cmax_primary_mg_L > 0.0
        assert result.tmax_primary_h > 0.0
        assert result.tmax_primary_h < BASE_PARAMS["bile_release_h"]

    def test_secondary_peak_exists(self):
        """Should find a secondary peak after bile release."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.cmax_secondary_mg_L > 0.0
        assert result.tmax_secondary_h > BASE_PARAMS["bile_release_h"] + 0.5

    def test_secondary_peak_smaller_than_primary(self):
        """Secondary peak should be smaller than primary."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.cmax_secondary_mg_L < result.cmax_primary_mg_L

    def test_secondary_peak_ratio(self):
        """secondary_peak_ratio = cmax_secondary / cmax_primary."""
        result = simulate_ehc(**BASE_PARAMS)
        expected = result.cmax_secondary_mg_L / result.cmax_primary_mg_L
        assert abs(result.secondary_peak_ratio - expected) < 1e-10

    def test_auc_positive(self):
        """AUC should be positive."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.auc_total_mg_h_per_L > 0.0

    def test_no_biliary_no_secondary_peak(self):
        """With f_biliary=0, secondary peak ratio should be lower than with recirculation."""
        r_no_bile = simulate_ehc(**{**BASE_PARAMS, "f_biliary": 0.0})
        r_with_bile = simulate_ehc(**{**BASE_PARAMS, "f_biliary": 0.5})
        # Without bile recirculation, secondary "peak" is just the decay tail
        # so its ratio relative to primary should be lower
        assert r_no_bile.secondary_peak_ratio < r_with_bile.secondary_peak_ratio

    def test_high_biliary_larger_secondary(self):
        """Higher f_biliary should produce larger secondary peak ratio."""
        r_low = simulate_ehc(**{**BASE_PARAMS, "f_biliary": 0.1})
        r_high = simulate_ehc(**{**BASE_PARAMS, "f_biliary": 0.6})
        assert r_high.secondary_peak_ratio > r_low.secondary_peak_ratio

    def test_dose_linearity_cmax(self):
        """Doubling dose should approximately double Cmax."""
        r1 = simulate_ehc(**{**BASE_PARAMS, "dose_mg": 100.0})
        r2 = simulate_ehc(**{**BASE_PARAMS, "dose_mg": 200.0})
        ratio = r2.cmax_primary_mg_L / r1.cmax_primary_mg_L
        assert 1.9 < ratio < 2.1

    def test_dose_linearity_auc(self):
        """Doubling dose should approximately double AUC."""
        r1 = simulate_ehc(**{**BASE_PARAMS, "dose_mg": 100.0})
        r2 = simulate_ehc(**{**BASE_PARAMS, "dose_mg": 200.0})
        ratio = r2.auc_total_mg_h_per_L / r1.auc_total_mg_h_per_L
        assert 1.9 < ratio < 2.1

    def test_higher_ka_earlier_tmax(self):
        """Higher absorption rate should give earlier primary Tmax."""
        r_slow = simulate_ehc(**{**BASE_PARAMS, "ka_per_h": 0.5})
        r_fast = simulate_ehc(**{**BASE_PARAMS, "ka_per_h": 3.0})
        assert r_fast.tmax_primary_h < r_slow.tmax_primary_h

    def test_higher_clearance_lower_cmax(self):
        """Higher clearance should give lower Cmax."""
        r_low_cl = simulate_ehc(**{**BASE_PARAMS, "cl_L_per_h": 2.0})
        r_high_cl = simulate_ehc(**{**BASE_PARAMS, "cl_L_per_h": 20.0})
        assert r_high_cl.cmax_primary_mg_L < r_low_cl.cmax_primary_mg_L

    def test_higher_vd_lower_cmax(self):
        """Higher Vd should give lower Cmax (C = amount/Vd)."""
        r_small_vd = simulate_ehc(**{**BASE_PARAMS, "vd_L": 20.0})
        r_large_vd = simulate_ehc(**{**BASE_PARAMS, "vd_L": 200.0})
        assert r_large_vd.cmax_primary_mg_L < r_small_vd.cmax_primary_mg_L

    def test_concentrations_non_negative(self):
        """All concentrations should be non-negative."""
        result = simulate_ehc(**BASE_PARAMS)
        assert all(c >= 0.0 for c in result.c_systemic_mg_L)

    def test_drug_name_preserved(self):
        """Drug name should be preserved in result."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.drug_name == "testdrug"

    def test_dose_mg_preserved(self):
        """Dose should be preserved in result."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.dose_mg == 100.0

    def test_f_biliary_preserved(self):
        """f_biliary should be preserved in result."""
        result = simulate_ehc(**BASE_PARAMS)
        assert result.f_biliary == 0.3

    def test_invalid_dose(self):
        """Should raise ValueError for non-positive dose."""
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_ehc(**{**BASE_PARAMS, "dose_mg": 0.0})

    def test_invalid_ka(self):
        """Should raise ValueError for non-positive ka."""
        with pytest.raises(ValueError, match="ka_per_h"):
            simulate_ehc(**{**BASE_PARAMS, "ka_per_h": -1.0})

    def test_invalid_cl(self):
        """Should raise ValueError for non-positive clearance."""
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_ehc(**{**BASE_PARAMS, "cl_L_per_h": 0.0})

    def test_invalid_vd(self):
        """Should raise ValueError for non-positive Vd."""
        with pytest.raises(ValueError, match="vd_L"):
            simulate_ehc(**{**BASE_PARAMS, "vd_L": -10.0})

    def test_invalid_f_biliary(self):
        """Should raise ValueError for f_biliary outside [0, 1]."""
        with pytest.raises(ValueError, match="f_biliary"):
            simulate_ehc(**{**BASE_PARAMS, "f_biliary": 1.5})

    def test_notes_non_empty(self):
        """Notes should be non-empty."""
        result = simulate_ehc(**BASE_PARAMS)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_auc_trapezoidal_manual_check(self):
        """AUC should match manual trapezoidal calculation."""
        result = simulate_ehc(**BASE_PARAMS)
        times = result.times_h
        concs = result.c_systemic_mg_L
        manual_auc = sum(
            0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
            for i in range(1, len(times))
        )
        assert abs(result.auc_total_mg_h_per_L - manual_auc) < 1e-10
