"""Tests for the prodrug activation model (Phase 209)."""

from __future__ import annotations

import pytest
import numpy as np

from omega_pbpk.clinical.prodrug import (
    ProdrugResult,
    simulate_prodrug,
    compare_prodrug_doses,
)

# ---------------------------------------------------------------------------
# Default parameter set for convenience
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    drug_name="ActiveDrug",
    prodrug_name="MyProdrug",
    dose_mg=100.0,
    route="oral",
    ka_per_h=1.0,
    k_convert_per_h=0.5,
    f_convert=0.8,
    cl_prodrug_L_per_h=5.0,
    cl_active_L_per_h=10.0,
    vd_prodrug_L=20.0,
    vd_active_L=40.0,
    f_oral=1.0,
    t_end_h=24.0,
    dt_h=0.05,
)


def _run(**overrides):
    params = {**_DEFAULTS, **overrides}
    return simulate_prodrug(**params)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-50.0)


def test_cl_prodrug_zero_raises():
    with pytest.raises(ValueError, match="cl_prodrug_L_per_h"):
        _run(cl_prodrug_L_per_h=0.0)


def test_cl_prodrug_negative_raises():
    with pytest.raises(ValueError, match="cl_prodrug_L_per_h"):
        _run(cl_prodrug_L_per_h=-1.0)


def test_cl_active_zero_raises():
    with pytest.raises(ValueError, match="cl_active_L_per_h"):
        _run(cl_active_L_per_h=0.0)


def test_vd_prodrug_zero_raises():
    with pytest.raises(ValueError, match="vd_prodrug_L"):
        _run(vd_prodrug_L=0.0)


def test_vd_active_zero_raises():
    with pytest.raises(ValueError, match="vd_active_L"):
        _run(vd_active_L=0.0)


def test_f_convert_too_low_raises():
    with pytest.raises(ValueError, match="f_convert"):
        _run(f_convert=-0.1)


def test_f_convert_too_high_raises():
    with pytest.raises(ValueError, match="f_convert"):
        _run(f_convert=1.1)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        _run(route="subcutaneous")


def test_k_convert_negative_raises():
    with pytest.raises(ValueError, match="k_convert_per_h"):
        _run(k_convert_per_h=-0.1)


# ---------------------------------------------------------------------------
# IV route
# ---------------------------------------------------------------------------


def test_iv_prodrug_peaks_at_t0():
    """IV bolus: prodrug should be highest at t=0."""
    r = _run(route="iv", k_convert_per_h=0.3)
    assert r.tmax_prodrug_h == pytest.approx(0.0, abs=0.1)


def test_iv_prodrug_initial_concentration():
    """IV bolus: C_prodrug(0) = dose / Vd_prodrug."""
    r = _run(route="iv", dose_mg=200.0, vd_prodrug_L=20.0)
    expected_c0 = 200.0 / 20.0  # 10 mg/L
    assert r.c_prodrug_mg_L[0] == pytest.approx(expected_c0, rel=0.01)


def test_iv_active_peaks_after_t0():
    """IV route: active drug should peak after t=0 due to formation lag."""
    r = _run(route="iv", k_convert_per_h=0.5)
    assert r.tmax_active_h > 0.0


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------


def test_oral_prodrug_peaks_later_than_t0():
    """Oral route: prodrug peaks after t=0 due to absorption rate limit."""
    r = _run(route="oral", ka_per_h=1.0)
    assert r.tmax_prodrug_h > 0.0


def test_oral_active_peaks_after_prodrug():
    """Active drug Tmax should be at least as late as prodrug Tmax."""
    r = _run(route="oral", ka_per_h=1.0, k_convert_per_h=0.3)
    assert r.tmax_active_h >= r.tmax_prodrug_h - 0.1


# ---------------------------------------------------------------------------
# f_convert edge cases
# ---------------------------------------------------------------------------


def test_f_convert_zero_gives_negligible_active_auc():
    """f_convert=0: active drug AUC should be essentially zero."""
    r = _run(f_convert=0.0)
    assert r.auc_active < 1e-6
    assert "f_convert=0" in " ".join(r.notes)


def test_f_convert_one_maximizes_active():
    """f_convert=1: more active drug formed vs f_convert=0.5."""
    r1 = _run(f_convert=1.0)
    r05 = _run(f_convert=0.5)
    assert r1.auc_active > r05.auc_active


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------


def test_double_dose_doubles_active_cmax():
    """Linear system: 2x dose should give ~2x active Cmax."""
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    assert r2.cmax_active == pytest.approx(r1.cmax_active * 2.0, rel=0.05)


def test_double_dose_doubles_active_auc():
    """Linear system: 2x dose should give ~2x active AUC."""
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    assert r2.auc_active == pytest.approx(r1.auc_active * 2.0, rel=0.05)


# ---------------------------------------------------------------------------
# k_convert effect on tmax_active
# ---------------------------------------------------------------------------


def test_higher_k_convert_earlier_tmax_active():
    """Higher k_convert should lead to earlier active drug Tmax."""
    r_slow = _run(k_convert_per_h=0.1, route="iv")
    r_fast = _run(k_convert_per_h=2.0, route="iv")
    assert r_fast.tmax_active_h < r_slow.tmax_active_h


def test_zero_k_convert_no_active_drug():
    """k_convert=0: no conversion, active drug stays at zero."""
    r = _run(k_convert_per_h=0.0)
    assert r.auc_active < 1e-6
    assert r.cmax_active < 1e-6


# ---------------------------------------------------------------------------
# compare_prodrug_doses
# ---------------------------------------------------------------------------


def test_compare_prodrug_doses_returns_list():
    results = compare_prodrug_doses(
        drug_name="ActiveDrug",
        prodrug_name="MyProdrug",
        doses_mg=[50.0, 100.0, 200.0],
        route="oral",
        ka_per_h=1.0,
        k_convert_per_h=0.5,
        f_convert=0.8,
        cl_prodrug_L_per_h=5.0,
        cl_active_L_per_h=10.0,
        vd_prodrug_L=20.0,
        vd_active_L=40.0,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_prodrug_doses_sorted_by_auc_active():
    """compare_prodrug_doses returns results sorted by auc_active ascending."""
    results = compare_prodrug_doses(
        drug_name="ActiveDrug",
        prodrug_name="MyProdrug",
        doses_mg=[200.0, 50.0, 100.0],
        route="oral",
        ka_per_h=1.0,
        k_convert_per_h=0.5,
        f_convert=0.8,
        cl_prodrug_L_per_h=5.0,
        cl_active_L_per_h=10.0,
        vd_prodrug_L=20.0,
        vd_active_L=40.0,
    )
    aucs = [r.auc_active for r in results]
    assert aucs == sorted(aucs)


# ---------------------------------------------------------------------------
# Result field types and values
# ---------------------------------------------------------------------------


def test_result_is_prodrug_result():
    r = _run()
    assert isinstance(r, ProdrugResult)


def test_result_fields_are_floats():
    r = _run()
    for attr in [
        "cmax_prodrug",
        "cmax_active",
        "tmax_prodrug_h",
        "tmax_active_h",
        "auc_prodrug",
        "auc_active",
        "activation_efficiency",
    ]:
        assert isinstance(getattr(r, attr), float), f"{attr} should be float"


def test_result_arrays_correct_length():
    r = _run(t_end_h=12.0, dt_h=0.1)
    expected = int(round(12.0 / 0.1)) + 1
    assert len(r.times_h) == expected
    assert len(r.c_prodrug_mg_L) == expected
    assert len(r.c_active_mg_L) == expected


def test_auc_positive():
    r = _run()
    assert r.auc_prodrug > 0.0
    assert r.auc_active > 0.0


def test_activation_efficiency_bounded():
    """Activation efficiency should be a positive float."""
    r = _run()
    assert r.activation_efficiency >= 0.0


def test_notes_is_list():
    r = _run()
    assert isinstance(r.notes, list)


def test_concentrations_non_negative():
    r = _run()
    assert np.all(r.c_prodrug_mg_L >= 0.0)
    assert np.all(r.c_active_mg_L >= 0.0)
