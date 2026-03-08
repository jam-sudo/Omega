"""Tests for Phase 1061 — Drug Implant PK."""

import pytest

from omega_pbpk.core.drug_implant_pk import (
    VALID_IMPLANT_TYPES,
    ImplantPKResult,
    simulate_implant_pk,
)

# ---------------------------------------------------------------------------
# Basic return-type and structure tests
# ---------------------------------------------------------------------------


def test_returns_implant_pk_result():
    result = simulate_implant_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, ImplantPKResult)


def test_drug_name_preserved():
    result = simulate_implant_pk("Leuprolide", dose_mg=50.0)
    assert result.drug_name == "Leuprolide"


def test_implant_type_preserved():
    result = simulate_implant_pk("Drug", dose_mg=100.0, implant_type="EVA_pellet")
    assert result.implant_type == "EVA_pellet"


def test_c_plasma_initial_burst():
    """c_plasma[0] should be > 0 due to the burst release."""
    result = simulate_implant_pk("Drug", dose_mg=100.0)
    assert result.c_plasma_mg_L[0] > 0.0


def test_fraction_released_initial():
    """fraction_released[0] should be > 0 (burst)."""
    result = simulate_implant_pk("Drug", dose_mg=100.0)
    assert result.fraction_released[0] > 0.0


def test_fraction_released_monotonic():
    result = simulate_implant_pk("Drug", dose_mg=100.0, dt_h=2.0)
    fr = result.fraction_released
    for i in range(len(fr) - 1):
        assert fr[i + 1] >= fr[i] - 1e-9, f"Non-monotone at index {i}"


def test_fraction_released_leq_one():
    result = simulate_implant_pk("Drug", dose_mg=100.0)
    assert result.fraction_released[-1] <= 1.0 + 1e-9


def test_auc_plasma_positive():
    result = simulate_implant_pk("Drug", dose_mg=100.0)
    assert result.auc_plasma > 0.0


def test_cmax_plasma_positive():
    result = simulate_implant_pk("Drug", dose_mg=100.0)
    assert result.cmax_plasma > 0.0


def test_duration_of_action_positive_low_mec():
    result = simulate_implant_pk("Drug", dose_mg=100.0, mec_mg_L=0.0001)
    assert result.duration_of_action_h > 0.0


def test_steady_state_concentration_positive():
    result = simulate_implant_pk("Drug", dose_mg=100.0)
    assert result.steady_state_concentration_mg_L > 0.0


def test_notes_nonempty():
    result = simulate_implant_pk("Drug", dose_mg=100.0)
    assert len(result.notes) > 0


def test_silicone_rod_lower_cmax_than_plga():
    """Silicone rod (slower release) should produce lower Cmax than PLGA rod."""
    plga = simulate_implant_pk("Drug", dose_mg=100.0, implant_type="PLGA_rod", t_end_h=720.0)
    sil = simulate_implant_pk("Drug", dose_mg=100.0, implant_type="silicone_rod", t_end_h=720.0)
    assert sil.cmax_plasma < plga.cmax_plasma


def test_plga_lower_t50_than_silicone():
    """PLGA rod releases 50% faster than silicone rod (lower t50)."""
    plga = simulate_implant_pk("Drug", dose_mg=100.0, implant_type="PLGA_rod", t_end_h=720.0)
    sil = simulate_implant_pk("Drug", dose_mg=100.0, implant_type="silicone_rod", t_end_h=2000.0)
    assert plga.t50_release_h < sil.t50_release_h


def test_times_array_length():
    result = simulate_implant_pk("Drug", dose_mg=50.0, t_end_h=100.0, dt_h=1.0)
    assert len(result.times_h) == 101  # 0..100 inclusive


def test_all_implant_types_run():
    for itype in VALID_IMPLANT_TYPES:
        result = simulate_implant_pk("Drug", dose_mg=50.0, implant_type=itype)
        assert result.cmax_plasma > 0.0


def test_higher_dose_higher_auc():
    r_lo = simulate_implant_pk("Drug", dose_mg=50.0)
    r_hi = simulate_implant_pk("Drug", dose_mg=200.0)
    assert r_hi.auc_plasma > r_lo.auc_plasma


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_implant_pk("Drug", dose_mg=0.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_implant_pk("Drug", dose_mg=100.0, cl_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_implant_pk("Drug", dose_mg=100.0, vd_L=0.0)


def test_validation_mec_zero():
    with pytest.raises(ValueError, match="mec_mg_L"):
        simulate_implant_pk("Drug", dose_mg=100.0, mec_mg_L=0.0)


def test_validation_invalid_implant_type():
    with pytest.raises(ValueError, match="implant_type"):
        simulate_implant_pk("Drug", dose_mg=100.0, implant_type="unknown_device")
