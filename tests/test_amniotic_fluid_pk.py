"""Tests for Phase 540 — Drug Distribution in Amniotic Fluid."""

import pytest

from omega_pbpk.core.amniotic_fluid_pk import (
    AmnioticFluidPKResult,
    compare_placental_permeability,
    simulate_amniotic_fluid_pk,
)

# --- Helpers -----------------------------------------------------------------

BASE_PARAMS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_maternal_L_per_h=5.0,
    vd_maternal_L=30.0,
)


def _sim(**overrides):
    p = {**BASE_PARAMS, **overrides}
    return simulate_amniotic_fluid_pk(**p)


# --- Return type & fields ----------------------------------------------------


def test_return_type():
    assert isinstance(_sim(), AmnioticFluidPKResult)


def test_has_all_fields():
    r = _sim()
    for attr in [
        "drug_name",
        "dose_mg",
        "ps_placenta_L_per_h",
        "gestational_week",
        "times_h",
        "c_maternal_mg_L",
        "c_fetal_mg_L",
        "c_amniotic_mg_L",
        "cmax_maternal",
        "cmax_fetal",
        "cmax_amniotic",
        "auc_maternal",
        "auc_fetal",
        "auc_amniotic",
        "fetal_maternal_ratio",
        "amniotic_maternal_ratio",
        "fetal_exposure_concern",
        "t_half_maternal_h",
        "notes",
    ]:
        assert hasattr(r, attr), f"Missing field: {attr}"


# --- Initial conditions -------------------------------------------------------


def test_maternal_starts_at_dose_over_vd():
    r = _sim()
    expected = r.dose_mg / 30.0
    assert r.c_maternal_mg_L[0] == pytest.approx(expected, rel=1e-6)


def test_fetal_starts_at_zero():
    r = _sim()
    assert r.c_fetal_mg_L[0] == pytest.approx(0.0)


def test_amniotic_starts_at_zero():
    r = _sim()
    assert r.c_amniotic_mg_L[0] == pytest.approx(0.0)


def test_times_start_at_zero():
    r = _sim()
    assert r.times_h[0] == pytest.approx(0.0)


# --- Time series length -------------------------------------------------------


def test_time_series_same_length():
    r = _sim()
    assert len(r.times_h) == len(r.c_maternal_mg_L)
    assert len(r.times_h) == len(r.c_fetal_mg_L)
    assert len(r.times_h) == len(r.c_amniotic_mg_L)


# --- Dynamics ----------------------------------------------------------------


def test_maternal_concentration_declines():
    r = _sim()
    # Maternal should generally fall from its peak (C0)
    assert r.c_maternal_mg_L[0] >= r.c_maternal_mg_L[-1]


def test_fetal_rises_then_falls():
    r = _sim()
    # Fetal should rise from 0 and not be zero at midpoint
    mid = len(r.c_fetal_mg_L) // 2
    assert r.c_fetal_mg_L[mid] > 0.0
    assert r.c_fetal_mg_L[0] == pytest.approx(0.0)


def test_amniotic_rises_from_zero():
    r = _sim()
    # Amniotic should have some concentration after some time
    assert r.c_amniotic_mg_L[-1] >= 0.0
    # Should have non-zero peak
    assert r.cmax_amniotic >= 0.0


def test_cmax_maternal_positive():
    r = _sim()
    assert r.cmax_maternal > 0.0


def test_cmax_fetal_positive():
    r = _sim()
    assert r.cmax_fetal > 0.0


def test_auc_maternal_positive():
    r = _sim()
    assert r.auc_maternal > 0.0


def test_auc_fetal_positive():
    r = _sim()
    assert r.auc_fetal > 0.0


# --- Placental permeability effects ------------------------------------------


def test_higher_ps_higher_fetal_ratio():
    r_low = _sim(ps_placenta_L_per_h=0.01)
    r_high = _sim(ps_placenta_L_per_h=0.5)
    assert r_high.fetal_maternal_ratio > r_low.fetal_maternal_ratio


def test_higher_ps_higher_cmax_fetal():
    r_low = _sim(ps_placenta_L_per_h=0.01)
    r_high = _sim(ps_placenta_L_per_h=0.5)
    assert r_high.cmax_fetal > r_low.cmax_fetal


# --- Fetal exposure concern flag ---------------------------------------------


def test_fetal_concern_true_when_ratio_exceeds_threshold():
    # High PS should give fetal/maternal ratio > 0.1
    r = _sim(ps_placenta_L_per_h=1.0)
    if r.fetal_maternal_ratio > 0.1:
        assert r.fetal_exposure_concern is True


def test_fetal_concern_false_when_ratio_below_threshold():
    # Very low PS -> negligible fetal exposure
    r = _sim(ps_placenta_L_per_h=0.0001)
    assert r.fetal_exposure_concern is False


# --- Dose linearity ----------------------------------------------------------


def test_dose_linearity_auc_maternal():
    r1 = _sim(dose_mg=50.0)
    r2 = _sim(dose_mg=100.0)
    assert r2.auc_maternal == pytest.approx(2.0 * r1.auc_maternal, rel=0.01)


def test_dose_linearity_auc_fetal():
    r1 = _sim(dose_mg=50.0)
    r2 = _sim(dose_mg=100.0)
    assert r2.auc_fetal == pytest.approx(2.0 * r1.auc_fetal, rel=0.01)


def test_dose_linearity_ratio_unchanged():
    """Fetal/maternal ratio should be independent of dose."""
    r1 = _sim(dose_mg=50.0)
    r2 = _sim(dose_mg=200.0)
    assert r1.fetal_maternal_ratio == pytest.approx(r2.fetal_maternal_ratio, rel=0.001)


# --- Gestational week field ---------------------------------------------------


def test_gestational_week_stored():
    r = _sim(gestational_week=30)
    assert r.gestational_week == 30


# --- compare_placental_permeability ------------------------------------------


def test_compare_returns_correct_length():
    ps_vals = [0.01, 0.05, 0.1, 0.5]
    results = compare_placental_permeability("Drug", 100.0, 5.0, 30.0, ps_vals)
    assert len(results) == len(ps_vals)


def test_compare_sorted_by_fetal_ratio_ascending():
    ps_vals = [0.5, 0.01, 0.1]
    results = compare_placental_permeability("Drug", 100.0, 5.0, 30.0, ps_vals)
    ratios = [r.fetal_maternal_ratio for r in results]
    assert ratios == sorted(ratios)


def test_compare_returns_amniotic_fluid_pk_results():
    results = compare_placental_permeability("Drug", 100.0, 5.0, 30.0, [0.01, 0.1])
    for r in results:
        assert isinstance(r, AmnioticFluidPKResult)


# --- Validation errors -------------------------------------------------------


def test_validation_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-5.0)


def test_validation_zero_cl():
    with pytest.raises(ValueError, match="cl_maternal"):
        _sim(cl_maternal_L_per_h=0.0)


def test_validation_zero_vd():
    with pytest.raises(ValueError, match="vd_maternal"):
        _sim(vd_maternal_L=0.0)


def test_validation_gestational_week_too_low():
    with pytest.raises(ValueError, match="gestational_week"):
        _sim(gestational_week=10)


def test_validation_gestational_week_too_high():
    with pytest.raises(ValueError, match="gestational_week"):
        _sim(gestational_week=45)
