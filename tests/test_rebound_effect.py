"""Tests for Phase 273: drug rebound/withdrawal PK/PD model."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.rebound_effect import ReboundResult, simulate_rebound

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_rebound(**kwargs) -> ReboundResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=50.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        n_doses_chronic=10,
        dosing_interval_h=12.0,
        ec50_mg_L=0.5,
        emax=0.8,
        k_receptor_up=0.1,
        k_receptor_down=0.05,
        receptor_max=2.5,
        ka_per_h=1.0,
        f_oral=0.9,
        t_washout_h=48.0,
        dt_h=0.5,
    )
    defaults.update(kwargs)
    return simulate_rebound(**defaults)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        default_rebound(dose_mg=0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        default_rebound(dose_mg=-10)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        default_rebound(cl_L_per_h=0)


def test_invalid_cl_negative():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        default_rebound(cl_L_per_h=-1)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        default_rebound(vd_L=0)


def test_invalid_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        default_rebound(vd_L=-5)


def test_invalid_n_doses_zero():
    with pytest.raises(ValueError, match="n_doses_chronic"):
        default_rebound(n_doses_chronic=0)


def test_invalid_dosing_interval_zero():
    with pytest.raises(ValueError, match="dosing_interval_h"):
        default_rebound(dosing_interval_h=0)


def test_invalid_ec50_zero():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        default_rebound(ec50_mg_L=0)


def test_invalid_ec50_negative():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        default_rebound(ec50_mg_L=-0.1)


def test_invalid_emax_zero():
    with pytest.raises(ValueError, match="emax"):
        default_rebound(emax=0)


def test_invalid_emax_greater_than_one():
    with pytest.raises(ValueError, match="emax"):
        default_rebound(emax=1.5)


def test_invalid_k_receptor_up_negative():
    with pytest.raises(ValueError, match="k_receptor rates"):
        default_rebound(k_receptor_up=-0.1)


def test_invalid_k_receptor_down_negative():
    with pytest.raises(ValueError, match="k_receptor rates"):
        default_rebound(k_receptor_down=-0.1)


def test_invalid_receptor_max_below_one():
    with pytest.raises(ValueError, match="receptor_max"):
        default_rebound(receptor_max=0.5)


def test_invalid_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        default_rebound(f_oral=0)


# ---------------------------------------------------------------------------
# Basic output structure
# ---------------------------------------------------------------------------


def test_c_plasma_has_positive_values():
    result = default_rebound()
    assert any(c > 0 for c in result.c_plasma_mg_L)


def test_receptor_density_starts_at_one():
    result = default_rebound()
    assert result.receptor_density[0] == pytest.approx(1.0)


def test_receptor_upregulates_with_chronic_dosing():
    result = default_rebound(n_doses_chronic=20, k_receptor_up=0.15, receptor_max=3.0)
    assert max(result.receptor_density) > 1.0


def test_receptor_returns_toward_baseline_after_withdrawal():
    result = default_rebound(
        n_doses_chronic=20,
        k_receptor_up=0.2,
        k_receptor_down=0.1,
        t_washout_h=72.0,
        receptor_max=3.0,
        dt_h=0.5,
    )
    withdrawal_idx = round(result.withdrawal_onset_h / 0.5)
    # peak receptor density during dosing
    peak_rec = max(result.receptor_density[:withdrawal_idx])
    # receptor at end of simulation
    final_rec = result.receptor_density[-1]
    assert final_rec < peak_rec


def test_peak_rebound_positive():
    result = default_rebound()
    assert result.peak_rebound >= 0


def test_withdrawal_onset_equals_n_doses_times_interval():
    result = default_rebound(n_doses_chronic=7, dosing_interval_h=24.0)
    assert result.withdrawal_onset_h == pytest.approx(7 * 24.0)


def test_array_lengths_equal():
    result = default_rebound()
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.effect) == n
    assert len(result.receptor_density) == n
    assert len(result.drug_phase) == n


def test_drug_phase_contains_both():
    result = default_rebound()
    assert "dosing" in result.drug_phase
    assert "withdrawal" in result.drug_phase


def test_rebound_duration_nonnegative():
    result = default_rebound()
    assert result.rebound_duration_h >= 0


def test_drug_name_stored():
    result = default_rebound(drug_name="Propranolol")
    assert result.drug_name == "Propranolol"


def test_notes_contains_drug_name():
    result = default_rebound(drug_name="Metoprolol")
    assert "Metoprolol" in result.notes


def test_emax_one_is_valid():
    # emax=1.0 should not raise
    result = default_rebound(emax=1.0)
    assert result.peak_rebound >= 0


def test_single_dose_withdrawal():
    # n_doses_chronic=1 should work
    result = default_rebound(n_doses_chronic=1, dosing_interval_h=12.0)
    assert result.withdrawal_onset_h == pytest.approx(12.0)
    assert len(result.times_h) > 0
