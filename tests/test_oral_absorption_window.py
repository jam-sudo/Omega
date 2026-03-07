"""Tests for oral_absorption_window module (Phase 531)."""

import pytest

from omega_pbpk.core.oral_absorption_window import (
    AbsorptionWindowResult,
    simulate_absorption_window,
    window_sensitivity,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(window_h=3.0, peff=1e-4, sol=10.0, dose=100.0):
    return simulate_absorption_window(
        drug_name="TestDrug",
        dose_mg=dose,
        peff_cm_s=peff,
        solubility_mg_mL=sol,
        absorption_window_h=window_h,
    )


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


class TestSimulateAbsorptionWindowBasic:
    def test_returns_correct_type(self):
        result = _run()
        assert isinstance(result, AbsorptionWindowResult)

    def test_times_and_plasma_same_length(self):
        result = _run()
        assert len(result.times_h) == len(result.c_plasma_mg_L)

    def test_times_start_at_zero(self):
        result = _run()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_near_t_end(self):
        result = simulate_absorption_window(
            "D",
            dose_mg=50,
            peff_cm_s=1e-4,
            solubility_mg_mL=5.0,
            absorption_window_h=3.0,
            t_end_h=12.0,
            dt_h=0.05,
        )
        assert result.times_h[-1] == pytest.approx(12.0, abs=0.1)

    def test_cmax_positive(self):
        result = _run(window_h=3.0)
        assert result.cmax > 0

    def test_auc_positive(self):
        result = _run(window_h=3.0)
        assert result.auc > 0

    def test_fa_in_range(self):
        result = _run(window_h=3.0)
        assert 0.0 <= result.fa <= 1.0


# ---------------------------------------------------------------------------
# Absorption window effect
# ---------------------------------------------------------------------------


class TestAbsorptionWindowEffect:
    def test_longer_window_higher_fa(self):
        short = _run(window_h=1.0)
        long = _run(window_h=6.0)
        assert long.fa > short.fa

    def test_longer_window_higher_cmax(self):
        # Use low peff so that drug absorption rate is slower than elimination,
        # making Cmax sensitive to the window duration.

        short = simulate_absorption_window(
            "D",
            dose_mg=100,
            peff_cm_s=1e-6,
            solubility_mg_mL=10.0,
            absorption_window_h=1.0,
            cl_L_per_h=10.0,
            vd_L=50.0,
        )
        long = simulate_absorption_window(
            "D",
            dose_mg=100,
            peff_cm_s=1e-6,
            solubility_mg_mL=10.0,
            absorption_window_h=6.0,
            cl_L_per_h=10.0,
            vd_L=50.0,
        )
        assert long.cmax > short.cmax

    def test_longer_window_higher_auc(self):
        short = _run(window_h=1.0)
        long = _run(window_h=6.0)
        assert long.auc > short.auc

    def test_zero_window_zero_fa(self):
        result = _run(window_h=0.0)
        assert result.fa == pytest.approx(0.0, abs=1e-6)

    def test_zero_window_zero_cmax(self):
        result = _run(window_h=0.0)
        assert result.cmax == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Permeability effect
# ---------------------------------------------------------------------------


class TestPermeabilityEffect:
    def test_high_peff_higher_fa(self):
        low = simulate_absorption_window(
            "D",
            dose_mg=100,
            peff_cm_s=1e-5,
            solubility_mg_mL=10.0,
            absorption_window_h=3.0,
        )
        high = simulate_absorption_window(
            "D",
            dose_mg=100,
            peff_cm_s=1e-3,
            solubility_mg_mL=10.0,
            absorption_window_h=3.0,
        )
        assert high.fa > low.fa

    def test_very_low_peff_low_fa(self):
        result = simulate_absorption_window(
            "D",
            dose_mg=100,
            peff_cm_s=1e-7,
            solubility_mg_mL=10.0,
            absorption_window_h=3.0,
        )
        assert result.fa < 0.05  # very poor absorber


# ---------------------------------------------------------------------------
# Solubility effect
# ---------------------------------------------------------------------------


class TestSolubilityEffect:
    def test_low_solubility_limits_fa(self):
        # Only 2.5 mg of 100 mg dose dissolved
        low_sol = simulate_absorption_window(
            "D",
            dose_mg=100,
            peff_cm_s=1e-3,
            solubility_mg_mL=0.01,
            dose_volume_mL=250.0,
            absorption_window_h=6.0,
        )
        high_sol = simulate_absorption_window(
            "D",
            dose_mg=100,
            peff_cm_s=1e-3,
            solubility_mg_mL=10.0,
            dose_volume_mL=250.0,
            absorption_window_h=6.0,
        )
        assert low_sol.fa < high_sol.fa

    def test_low_solubility_note_present(self):
        result = simulate_absorption_window(
            "D",
            dose_mg=100,
            peff_cm_s=1e-3,
            solubility_mg_mL=0.01,
            dose_volume_mL=250.0,
            absorption_window_h=3.0,
        )
        assert "undissolved" in result.notes.lower()


# ---------------------------------------------------------------------------
# window_sensitivity
# ---------------------------------------------------------------------------


class TestWindowSensitivity:
    def test_returns_list(self):
        results = window_sensitivity("D", dose_mg=100, peff_cm_s=1e-4, solubility_mg_mL=5.0)
        assert isinstance(results, list)

    def test_default_five_results(self):
        results = window_sensitivity("D", dose_mg=100, peff_cm_s=1e-4, solubility_mg_mL=5.0)
        assert len(results) == 5

    def test_sorted_by_window_hours(self):
        results = window_sensitivity(
            "D",
            dose_mg=100,
            peff_cm_s=1e-4,
            solubility_mg_mL=5.0,
            window_hours=[4.0, 1.0, 6.0, 2.0, 3.0],
        )
        windows = [r.absorption_window_h for r in results]
        assert windows == sorted(windows)

    def test_fa_increases_with_window(self):
        results = window_sensitivity("D", dose_mg=100, peff_cm_s=1e-4, solubility_mg_mL=5.0)
        fas = [r.fa for r in results]
        for i in range(len(fas) - 1):
            assert fas[i] <= fas[i + 1] + 1e-6

    def test_custom_window_hours(self):
        results = window_sensitivity(
            "D",
            dose_mg=100,
            peff_cm_s=1e-4,
            solubility_mg_mL=5.0,
            window_hours=[2.0, 5.0],
        )
        assert len(results) == 2

    def test_cmax_positive_for_nonzero_window(self):
        results = window_sensitivity(
            "D",
            dose_mg=100,
            peff_cm_s=1e-4,
            solubility_mg_mL=5.0,
            window_hours=[1.0, 3.0],
        )
        for r in results:
            assert r.cmax > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_absorption_window("D", dose_mg=-1, peff_cm_s=1e-4, solubility_mg_mL=5.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_absorption_window("D", dose_mg=0, peff_cm_s=1e-4, solubility_mg_mL=5.0)

    def test_negative_peff_raises(self):
        with pytest.raises(ValueError, match="peff_cm_s"):
            simulate_absorption_window("D", dose_mg=100, peff_cm_s=-1e-4, solubility_mg_mL=5.0)

    def test_zero_peff_raises(self):
        with pytest.raises(ValueError, match="peff_cm_s"):
            simulate_absorption_window("D", dose_mg=100, peff_cm_s=0, solubility_mg_mL=5.0)

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            simulate_absorption_window("D", dose_mg=100, peff_cm_s=1e-4, solubility_mg_mL=-1.0)
