"""Tests for Phase 334 — Dermal Absorption Flux Calculator."""

import pytest

from omega_pbpk.prediction.dermal_absorption_flux import (
    DermalAbsorptionResult,
    calculate_dermal_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PERM_CLASSES = {"low", "moderate", "high"}
_VALID_ABS_RISKS = {"negligible", "low", "moderate", "high"}


def _make_result(
    drug_name="TestDrug",
    logP=2.0,
    mw_Da=300.0,
    c_donor_mg_mL=1.0,
    skin_area_cm2=100.0,
    cl_sys_L_per_h=1.0,
    membrane_thickness_cm=0.05,
):
    return calculate_dermal_absorption(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        c_donor_mg_mL=c_donor_mg_mL,
        skin_area_cm2=skin_area_cm2,
        cl_sys_L_per_h=cl_sys_L_per_h,
        membrane_thickness_cm=membrane_thickness_cm,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _make_result()
    assert isinstance(result, DermalAbsorptionResult)


def test_result_is_frozen():
    result = _make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.kp_cm_per_h = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. kp > 0
# ---------------------------------------------------------------------------


def test_kp_positive():
    result = _make_result()
    assert result.kp_cm_per_h > 0.0


def test_kp_positive_for_low_logp():
    result = _make_result(logP=-2.0)
    assert result.kp_cm_per_h > 0.0


# ---------------------------------------------------------------------------
# 3. lag_time > 0
# ---------------------------------------------------------------------------


def test_lag_time_positive():
    result = _make_result()
    assert result.lag_time_h > 0.0


# ---------------------------------------------------------------------------
# 4. steady_state_flux > 0
# ---------------------------------------------------------------------------


def test_steady_state_flux_positive():
    result = _make_result()
    assert result.steady_state_flux_mg_cm2_per_h > 0.0


# ---------------------------------------------------------------------------
# 5. Higher logP → higher kp (in the range where Potts-Guy applies)
# ---------------------------------------------------------------------------


def test_higher_logp_higher_kp():
    r1 = _make_result(logP=1.0)
    r2 = _make_result(logP=4.0)
    assert r2.kp_cm_per_h > r1.kp_cm_per_h


# ---------------------------------------------------------------------------
# 6. Higher MW → lower kp
# ---------------------------------------------------------------------------


def test_higher_mw_lower_kp():
    r1 = _make_result(mw_Da=200.0)
    r2 = _make_result(mw_Da=600.0)
    assert r2.kp_cm_per_h < r1.kp_cm_per_h


# ---------------------------------------------------------------------------
# 7. Higher c_donor → proportionally higher flux
# ---------------------------------------------------------------------------


def test_higher_c_donor_higher_flux():
    r1 = _make_result(c_donor_mg_mL=0.5)
    r2 = _make_result(c_donor_mg_mL=2.0)
    assert r2.steady_state_flux_mg_cm2_per_h == pytest.approx(
        r1.steady_state_flux_mg_cm2_per_h * 4.0, rel=1e-9
    )


def test_flux_proportional_to_c_donor():
    r1 = _make_result(c_donor_mg_mL=1.0)
    r3 = _make_result(c_donor_mg_mL=3.0)
    assert r3.steady_state_flux_mg_cm2_per_h == pytest.approx(
        r1.steady_state_flux_mg_cm2_per_h * 3.0, rel=1e-9
    )


# ---------------------------------------------------------------------------
# 8. permeability_class valid
# ---------------------------------------------------------------------------


def test_permeability_class_valid():
    result = _make_result()
    assert result.permeability_class in _VALID_PERM_CLASSES


def test_permeability_class_low_for_high_mw_low_logp():
    result = _make_result(logP=-1.0, mw_Da=800.0)
    assert result.permeability_class == "low"


# ---------------------------------------------------------------------------
# 9. absorption_risk valid
# ---------------------------------------------------------------------------


def test_absorption_risk_valid():
    result = _make_result()
    assert result.absorption_risk in _VALID_ABS_RISKS


# ---------------------------------------------------------------------------
# 10. daily_absorption > 0
# ---------------------------------------------------------------------------


def test_daily_absorption_positive():
    result = _make_result()
    assert result.daily_absorption_mg > 0.0


# ---------------------------------------------------------------------------
# 11. css_systemic > 0
# ---------------------------------------------------------------------------


def test_css_systemic_positive():
    result = _make_result()
    assert result.css_systemic_mg_L > 0.0


# ---------------------------------------------------------------------------
# 12. Validation errors
# ---------------------------------------------------------------------------


def test_error_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _make_result(mw_Da=0.0)


def test_error_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        _make_result(mw_Da=-100.0)


def test_error_c_donor_zero():
    with pytest.raises(ValueError, match="c_donor"):
        _make_result(c_donor_mg_mL=0.0)


def test_error_c_donor_negative():
    with pytest.raises(ValueError, match="c_donor"):
        _make_result(c_donor_mg_mL=-1.0)


def test_error_skin_area_zero():
    with pytest.raises(ValueError, match="skin_area"):
        _make_result(skin_area_cm2=0.0)


def test_error_skin_area_negative():
    with pytest.raises(ValueError, match="skin_area"):
        _make_result(skin_area_cm2=-50.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        _make_result(cl_sys_L_per_h=0.0)


def test_error_cl_negative():
    with pytest.raises(ValueError, match="cl_sys"):
        _make_result(cl_sys_L_per_h=-2.0)


def test_error_membrane_thickness_zero():
    with pytest.raises(ValueError, match="membrane_thickness"):
        _make_result(membrane_thickness_cm=0.0)


def test_error_membrane_thickness_negative():
    with pytest.raises(ValueError, match="membrane_thickness"):
        _make_result(membrane_thickness_cm=-0.01)


def test_error_logp_too_low():
    with pytest.raises(ValueError, match="logP"):
        _make_result(logP=-6.0)


def test_error_logp_too_high():
    with pytest.raises(ValueError, match="logP"):
        _make_result(logP=11.0)


# ---------------------------------------------------------------------------
# 13. notes nonempty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _make_result()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 14. skin_area scales daily absorption proportionally
# ---------------------------------------------------------------------------


def test_daily_absorption_scales_with_skin_area():
    r1 = _make_result(skin_area_cm2=50.0)
    r2 = _make_result(skin_area_cm2=200.0)
    assert r2.daily_absorption_mg == pytest.approx(r1.daily_absorption_mg * 4.0, rel=1e-9)
