"""Tests for Phase 355 — Peritoneal Drug Absorption Model."""

import pytest

from omega_pbpk.core.peritoneal_absorption_pk import (
    PeritonealAbsorptionResult,
    simulate_peritoneal_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    k_abs_peritoneal_per_h=0.5,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    f_hepatic=0.0,
    k_portal_to_sys_per_h=30.0,
    v_peritoneal_mL=50.0,
    v_portal_mL=100.0,
    t_end_h=24.0,
    dt_h=0.05,
)


def default_result() -> PeritonealAbsorptionResult:
    return simulate_peritoneal_absorption(**DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = default_result()
    assert isinstance(result, PeritonealAbsorptionResult)


def test_times_list_not_empty():
    result = default_result()
    assert len(result.times_h) > 0


def test_array_lengths_equal():
    result = default_result()
    n = len(result.times_h)
    assert len(result.c_peritoneal_mg_mL) == n
    assert len(result.c_portal_mg_L) == n
    assert len(result.c_systemic_mg_L) == n


def test_times_start_at_zero():
    result = default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_or_near_t_end():
    result = default_result()
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.1)


# ---------------------------------------------------------------------------
# Peritoneal compartment dynamics
# ---------------------------------------------------------------------------


def test_peritoneal_starts_high():
    result = default_result()
    # Initial peritoneal conc = dose / v_peritoneal = 100 mg / 50 mL = 2 mg/mL
    assert result.c_peritoneal_mg_mL[0] == pytest.approx(2.0, rel=1e-4)


def test_peritoneal_decreases_monotonically():
    result = default_result()
    peri = result.c_peritoneal_mg_mL
    # Should be monotonically decreasing (strictly)
    for i in range(1, len(peri)):
        assert peri[i] <= peri[i - 1] + 1e-10


def test_peritoneal_approaches_zero():
    result = default_result()
    assert result.c_peritoneal_mg_mL[-1] < 0.01 * result.c_peritoneal_mg_mL[0]


# ---------------------------------------------------------------------------
# Systemic compartment dynamics
# ---------------------------------------------------------------------------


def test_systemic_starts_at_zero():
    result = default_result()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


def test_systemic_rises_and_falls():
    result = default_result()
    c_sys = result.c_systemic_mg_L
    max_val = max(c_sys)
    # Must rise above zero
    assert max_val > 0.0
    # Must fall below half-max at the end
    assert c_sys[-1] < 0.5 * max_val


def test_cmax_positive():
    result = default_result()
    assert result.cmax_systemic_mg_L > 0.0


def test_tmax_positive():
    result = default_result()
    assert result.tmax_systemic_h > 0.0


def test_auc_positive():
    result = default_result()
    assert result.auc_systemic_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Hepatic extraction effects
# ---------------------------------------------------------------------------


def test_higher_f_hepatic_lower_bioavailability():
    result_low = simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "f_hepatic": 0.1})
    result_high = simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "f_hepatic": 0.7})
    assert result_high.bioavailability_pct < result_low.bioavailability_pct


def test_zero_f_hepatic_bioavailability_near_100():
    # With f_hepatic=0 and fast absorption, bioavailability approaches 100%.
    # The numerical AUC is truncated at t_end_h so we allow a wider tolerance
    # to account for the residual AUC beyond the simulation window.
    result = simulate_peritoneal_absorption(
        **{**DEFAULT_PARAMS, "f_hepatic": 0.0, "k_abs_peritoneal_per_h": 5.0, "t_end_h": 48.0}
    )
    # Should be close to 100% when no hepatic extraction
    assert result.bioavailability_pct == pytest.approx(100.0, abs=5.0)


def test_first_pass_loss_pct_matches_f_hepatic():
    result = simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "f_hepatic": 0.4})
    assert result.first_pass_loss_pct == pytest.approx(40.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_auc():
    result_1x = simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "dose_mg": 50.0})
    result_2x = simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "dose_mg": 100.0})
    ratio = result_2x.auc_systemic_mg_h_per_L / result_1x.auc_systemic_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_dose_linearity_cmax():
    result_1x = simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "dose_mg": 50.0})
    result_2x = simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "dose_mg": 100.0})
    ratio = result_2x.cmax_systemic_mg_L / result_1x.cmax_systemic_mg_L
    assert ratio == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_nonpositive():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "dose_mg": 0.0})


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "dose_mg": -10.0})


def test_validation_cl_nonpositive():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "cl_sys_L_per_h": 0.0})


def test_validation_vd_nonpositive():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "vd_sys_L": -1.0})


def test_validation_f_hepatic_exactly_one():
    with pytest.raises(ValueError, match="f_hepatic"):
        simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "f_hepatic": 1.0})


def test_validation_f_hepatic_above_one():
    with pytest.raises(ValueError, match="f_hepatic"):
        simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "f_hepatic": 1.5})


def test_validation_f_hepatic_negative():
    with pytest.raises(ValueError, match="f_hepatic"):
        simulate_peritoneal_absorption(**{**DEFAULT_PARAMS, "f_hepatic": -0.1})


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_contains_drug_name():
    result = default_result()
    assert "TestDrug" in result.notes
