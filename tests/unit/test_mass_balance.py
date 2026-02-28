"""Unit tests for mass balance validation functions.

Tests both IV (mass_balance_check) and oral (oral_mass_balance_check)
routes, including boundary cases and the effects of ODE state structure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from omega_pbpk.validation import mass_balance_check, oral_mass_balance_check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_amounts(n_time: int, n_states: int, dose_mg: float) -> np.ndarray:
    """Create amounts array where mass is perfectly conserved (all in state 0)."""
    amounts = np.zeros((n_time, n_states))
    amounts[:, 0] = dose_mg  # all mass in first compartment
    return amounts


def _depleting_amounts(n_time: int, n_states: int, dose_mg: float) -> np.ndarray:
    """Create amounts array that exponentially depletes (IV elimination)."""
    t = np.linspace(0, 24, n_time)
    ke = np.log(2) / 4.0  # half-life = 4h
    total = dose_mg * np.exp(-ke * t)
    amounts = np.zeros((n_time, n_states))
    amounts[:, 0] = total * 0.7
    amounts[:, 1] = total * 0.3
    return amounts


# ---------------------------------------------------------------------------
# mass_balance_check (IV route)
# ---------------------------------------------------------------------------


class TestMassBalanceCheckIV:
    def test_perfect_conservation_no_warnings(self):
        amounts = _perfect_amounts(100, 35, 100.0)
        time_h = np.linspace(0, 24, 100)
        result = mass_balance_check(amounts, 100.0, time_h)
        assert result == []

    def test_depleting_amounts_no_warning_if_sink_states(self):
        """If sink states are included, total mass should be conserved."""
        n_time, dose = 100, 100.0
        amounts = _depleting_amounts(n_time, 35, dose)
        # Add sink amounts to preserve total
        total = np.sum(amounts, axis=1)
        deficit = dose - total
        amounts[:, -1] += deficit  # add to last state as metabolic sink
        result = mass_balance_check(amounts, dose)
        assert result == []

    def test_large_deviation_triggers_warning(self):
        amounts = _perfect_amounts(100, 35, 100.0)
        # Introduce 20% mass loss at t=12h
        amounts[50:, :] = amounts[50:, :] * 0.80
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            result = mass_balance_check(amounts, 100.0, tolerance_frac=0.005)
        assert len(result) > 0
        assert len(w_list) > 0

    def test_deviation_within_tolerance_no_warning(self):
        amounts = _perfect_amounts(100, 35, 100.0)
        amounts[:, 0] *= 0.999  # 0.1% loss — within default 0.5% tolerance
        result = mass_balance_check(amounts, 100.0)
        assert result == []

    def test_deviation_just_above_tolerance_triggers_warning(self):
        amounts = _perfect_amounts(100, 35, 100.0)
        amounts[:, 0] *= 0.994  # 0.6% loss — above 0.5% default tolerance
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mass_balance_check(amounts, 100.0, tolerance_frac=0.005)
        assert len(result) > 0

    def test_zero_dose_skips_check(self):
        amounts = np.zeros((10, 35))
        result = mass_balance_check(amounts, 0.0)
        assert result == []

    def test_negative_dose_skips_check(self):
        amounts = _perfect_amounts(10, 35, 50.0)
        result = mass_balance_check(amounts, -1.0)
        assert result == []

    def test_custom_tolerance(self):
        amounts = _perfect_amounts(100, 35, 100.0)
        amounts[:, 0] *= 0.97  # 3% loss
        # Should pass at 5% tolerance
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mass_balance_check(amounts, 100.0, tolerance_frac=0.05)
        assert result == []
        # Should fail at 1% tolerance
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mass_balance_check(amounts, 100.0, tolerance_frac=0.01)
        assert len(result) > 0

    def test_time_h_included_in_warning_message(self):
        amounts = _perfect_amounts(50, 35, 100.0)
        amounts[25:, :] *= 0.80  # mass drop at mid-simulation
        time_h = np.linspace(0, 24, 50)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mass_balance_check(amounts, 100.0, time_h=time_h, tolerance_frac=0.005)
        assert len(result) > 0
        assert "t=" in result[0]

    def test_returns_list_type(self):
        amounts = _perfect_amounts(10, 35, 50.0)
        result = mass_balance_check(amounts, 50.0)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# oral_mass_balance_check
# ---------------------------------------------------------------------------


class TestOralMassBalanceCheck:
    def test_perfect_conservation_no_warnings(self):
        n_time, n_states, dose = 200, 35, 100.0
        time_h = np.linspace(0, 24, n_time)
        amounts = _perfect_amounts(n_time, n_states, dose)
        result = oral_mass_balance_check(amounts, dose, time_h)
        assert result == []

    def test_skip_when_gi_residual_too_high(self):
        """If GI absorption is incomplete, check should be skipped (returns [])."""
        n_time, dose = 100, 100.0
        time_h = np.linspace(0, 12, n_time)
        amounts = np.zeros((n_time, 35))
        amounts[:, 0] = dose * 0.3  # only 30% absorbed
        # GI still holds 70% of dose
        gi_states = np.ones((n_time, 8)) * dose * 0.7 / 8.0
        result = oral_mass_balance_check(
            amounts, dose, time_h, gi_states=gi_states, min_absorption_frac=0.99
        )
        assert result == []

    def test_checks_when_gi_absorbed(self):
        """When GI residual < 1%, mass balance check is performed."""
        n_time, dose = 100, 100.0
        time_h = np.linspace(0, 24, n_time)
        amounts = _perfect_amounts(n_time, 35, dose)
        # Simulate complete absorption: GI residual drops to near 0
        gi_states = np.zeros((n_time, 8))
        gi_states[:5, :] = dose / 8.0  # GI drains completely by t≈5
        result = oral_mass_balance_check(amounts, dose, time_h, gi_states=gi_states)
        assert result == []

    def test_large_deviation_detected_at_final_point(self):
        n_time, dose = 100, 100.0
        time_h = np.linspace(0, 24, n_time)
        amounts = _perfect_amounts(n_time, 35, dose)
        amounts[-1, :] *= 0.70  # 30% mass loss at final timepoint
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = oral_mass_balance_check(amounts, dose, time_h, tolerance_frac=0.02)
        assert len(result) > 0

    def test_zero_dose_skips_check(self):
        amounts = np.zeros((50, 35))
        time_h = np.linspace(0, 24, 50)
        result = oral_mass_balance_check(amounts, 0.0, time_h)
        assert result == []

    def test_custom_tolerance_more_lenient(self):
        n_time, dose = 100, 100.0
        time_h = np.linspace(0, 24, n_time)
        amounts = _perfect_amounts(n_time, 35, dose)
        amounts[-1, :] *= 0.96  # 4% deviation
        # Should fail at 2% tolerance
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = oral_mass_balance_check(amounts, dose, time_h, tolerance_frac=0.02)
        assert len(result) > 0
        # Should pass at 5% tolerance
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = oral_mass_balance_check(amounts, dose, time_h, tolerance_frac=0.05)
        assert result == []

    def test_returns_list_type(self):
        amounts = _perfect_amounts(10, 35, 50.0)
        time_h = np.linspace(0, 24, 10)
        result = oral_mass_balance_check(amounts, 50.0, time_h)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Integration: IV simulation with real ODE
# ---------------------------------------------------------------------------


class TestMassBalanceWithRealSimulation:
    def test_iv_simulation_conserves_mass(self):
        """Real 35-state ODE IV simulation must pass mass balance (±0.5%)."""
        pytest.importorskip("omega_pbpk.core.body")
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(
            name="test_mb",
            mw=200.0,
            logP=1.5,
            fup=0.5,
            rbp=1.0,
            clint_hepatic_L_per_h=5.0,
        )
        model = WholeBodyPBPK(drug=drug)
        dose_mg = 100.0
        model.setup_iv(dose_mg=dose_mg)
        result = model.simulate(t_end_h=24.0)

        warnings_list = mass_balance_check(result.amounts, dose_mg, result.time_h)
        assert warnings_list == [], f"IV mass balance failed: {warnings_list}"
