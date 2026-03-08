"""Tests for Phase 360 — PK/PD Efficacy Score Calculator."""

import pytest

from omega_pbpk.clinical.pkpd_efficacy_score import (
    PKPDEfficacyResult,
    compute_pkpd_efficacy_score,
    optimize_dose,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs) -> PKPDEfficacyResult:
    params = dict(
        drug_name="DrugA",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_per_h=1.0,
        f_oral=0.8,
        tau_h=8.0,
        ec50_mg_L=1.0,
        emax=0.9,
        hill_n=1.0,
        pd_model="emax",
        mec_mg_L=0.5,
        mtc_mg_L=20.0,
        target_auc_mg_h_per_L=50.0,
    )
    params.update(kwargs)
    return compute_pkpd_efficacy_score(**params)


# ---------------------------------------------------------------------------
# Return type (frozen dataclass)
# ---------------------------------------------------------------------------


def test_return_type():
    result = default_result()
    assert isinstance(result, PKPDEfficacyResult)


def test_frozen_dataclass():
    result = default_result()
    with pytest.raises((AttributeError, TypeError)):
        result.composite_score = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Score ranges
# ---------------------------------------------------------------------------


def test_composite_score_in_range():
    result = default_result()
    assert 0.0 <= result.composite_score <= 100.0


def test_efficacy_score_in_range():
    result = default_result()
    assert 0.0 <= result.efficacy_score <= 100.0


def test_safety_score_in_range():
    result = default_result()
    assert 0.0 <= result.safety_score <= 100.0


def test_time_above_mec_pct_in_range():
    result = default_result()
    assert 0.0 <= result.time_above_mec_pct <= 100.0


def test_time_above_mtc_pct_in_range():
    result = default_result()
    assert 0.0 <= result.time_above_mtc_pct <= 100.0


# ---------------------------------------------------------------------------
# PK values
# ---------------------------------------------------------------------------


def test_auc_ss_positive():
    result = default_result()
    assert result.auc_ss_mg_h_per_L > 0.0


def test_cmax_ss_positive():
    result = default_result()
    assert result.cmax_ss_mg_L > 0.0


def test_trough_ss_nonneg():
    result = default_result()
    assert result.trough_ss_mg_L >= 0.0


# ---------------------------------------------------------------------------
# Emax model: higher dose -> higher efficacy
# ---------------------------------------------------------------------------


def test_higher_dose_higher_efficacy_emax():
    low = default_result(dose_mg=50.0, pd_model="emax")
    high = default_result(dose_mg=500.0, pd_model="emax")
    assert high.efficacy_score >= low.efficacy_score


# ---------------------------------------------------------------------------
# PD models
# ---------------------------------------------------------------------------


def test_time_above_ec50_model():
    result = default_result(pd_model="time_above_ec50")
    assert isinstance(result, PKPDEfficacyResult)
    assert 0.0 <= result.efficacy_score <= 100.0


def test_auc_ec50_model():
    result = default_result(pd_model="auc_ec50")
    assert isinstance(result, PKPDEfficacyResult)
    assert 0.0 <= result.efficacy_score <= 100.0


# ---------------------------------------------------------------------------
# therapeutic_window_adequate
# ---------------------------------------------------------------------------


def test_therapeutic_window_adequate_is_bool():
    result = default_result()
    assert isinstance(result.therapeutic_window_adequate, bool)


def test_therapeutic_window_adequate_low_mtc():
    """Very low MTC -> drug above toxic threshold -> adequate=False."""
    result = default_result(mtc_mg_L=0.001)  # near zero -> always above
    assert result.therapeutic_window_adequate is False


# ---------------------------------------------------------------------------
# recommended_dose_adjustment
# ---------------------------------------------------------------------------


def test_recommended_dose_adjustment_valid_values():
    result = default_result()
    assert result.recommended_dose_adjustment in {"increase", "maintain", "decrease"}


def test_recommend_decrease_when_toxic():
    """Very low MTC -> over MTC -> recommend decrease."""
    result = default_result(mtc_mg_L=0.001)
    assert result.recommended_dose_adjustment == "decrease"


def test_recommend_increase_low_dose():
    """Tiny dose -> very low efficacy -> recommend increase."""
    result = default_result(dose_mg=0.001)
    assert result.recommended_dose_adjustment == "increase"


# ---------------------------------------------------------------------------
# optimize_dose
# ---------------------------------------------------------------------------


def test_optimize_dose_returns_list():
    doses = [50.0, 100.0, 200.0]
    results = optimize_dose(
        drug_name="DrugB",
        doses_mg=doses,
        cl=5.0,
        vd=50.0,
        ka=1.0,
        f_oral=0.8,
        tau_h=8.0,
        ec50=1.0,
        emax=0.9,
        hill_n=1.0,
        pd_model="emax",
        mec=0.5,
        mtc=20.0,
        target_auc=50.0,
    )
    assert isinstance(results, list)
    assert len(results) == len(doses)


def test_optimize_dose_sorted_descending():
    doses = [10.0, 100.0, 500.0]
    results = optimize_dose(
        drug_name="DrugB",
        doses_mg=doses,
        cl=5.0,
        vd=50.0,
        ka=1.0,
        f_oral=0.8,
        tau_h=8.0,
        ec50=1.0,
        emax=0.9,
        hill_n=1.0,
        pd_model="emax",
        mec=0.5,
        mtc=20.0,
        target_auc=50.0,
    )
    scores = [r.composite_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_optimize_dose_all_results_are_pkpdresult():
    doses = [100.0, 200.0]
    results = optimize_dose(
        drug_name="DrugC",
        doses_mg=doses,
        cl=5.0,
        vd=50.0,
        ka=1.0,
        f_oral=0.8,
        tau_h=8.0,
        ec50=1.0,
        emax=0.9,
        hill_n=1.0,
        pd_model="emax",
        mec=0.5,
        mtc=20.0,
        target_auc=50.0,
    )
    for r in results:
        assert isinstance(r, PKPDEfficacyResult)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        default_result(dose_mg=0.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        default_result(dose_mg=-1.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        default_result(cl_L_per_h=0.0)


def test_error_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        default_result(vd_L=0.0)


def test_error_ec50_zero():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        default_result(ec50_mg_L=0.0)


def test_error_emax_zero():
    with pytest.raises(ValueError, match="emax"):
        default_result(emax=0.0)


def test_error_emax_too_high():
    with pytest.raises(ValueError, match="emax"):
        default_result(emax=1.5)


def test_error_invalid_pd_model():
    with pytest.raises(ValueError, match="pd_model"):
        default_result(pd_model="invalid_model")
