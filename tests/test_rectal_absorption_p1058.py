"""Tests for Phase 1058 — Drug Rectal Absorption Predictor."""

import pytest

from omega_pbpk.prediction.rectal_absorption_p1058 import (
    RectalAbsorptionResult,
    compare_rectal_formulations,
    predict_rectal_absorption,
)

# A very hydrophilic, high-MW drug that stays below the fa saturation cap,
# allowing enema vs suppository differentiation (mw=800, logP=-1).
_HYDROPHILIC_KW = {"mw_Da": 800.0, "logp": -1.0}


def test_return_type():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert isinstance(result, RectalAbsorptionResult)


def test_result_is_frozen():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    with pytest.raises((AttributeError, TypeError)):
        result.fa_rectal = 0.5  # type: ignore[misc]


def test_fa_rectal_in_valid_range():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert 0.0 < result.fa_rectal <= 0.9


def test_f_systemic_in_valid_range():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert 0.0 <= result.f_systemic <= 1.0


def test_tmax_rectal_h_positive():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert result.tmax_rectal_h > 0.0


def test_absorption_rate_constant_positive():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert result.absorption_rate_constant > 0.0


def test_onset_classification_valid():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert result.onset_classification in {"rapid", "moderate", "slow"}


def test_f_portal_bypass_pct_equals_50():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert result.f_portal_bypass_pct == pytest.approx(50.0)


def test_portal_contribution_pct_equals_50():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert result.portal_contribution_pct == pytest.approx(50.0)


def test_enema_higher_fa_than_suppository():
    """Enema gives higher fa_rectal than fatty suppository for a hydrophilic drug."""
    result_enema = predict_rectal_absorption(
        "HydrophilicDrug", formulation="enema", **_HYDROPHILIC_KW
    )
    result_supp = predict_rectal_absorption(
        "HydrophilicDrug", formulation="suppository", suppository_base="fatty", **_HYDROPHILIC_KW
    )
    assert result_enema.fa_rectal > result_supp.fa_rectal


def test_high_logp_higher_ka_than_low_logp():
    """Higher logP → higher permeability → higher ka."""
    result_high = predict_rectal_absorption("HighLogP", mw_Da=300.0, logp=4.0)
    result_low = predict_rectal_absorption("LowLogP", mw_Da=300.0, logp=-2.0)
    assert result_high.absorption_rate_constant > result_low.absorption_rate_constant


def test_water_soluble_base_faster_than_fatty():
    result_ws = predict_rectal_absorption(
        "HydrophilicDrug",
        formulation="suppository",
        suppository_base="water_soluble",
        **_HYDROPHILIC_KW,
    )
    result_fatty = predict_rectal_absorption(
        "HydrophilicDrug",
        formulation="suppository",
        suppository_base="fatty",
        **_HYDROPHILIC_KW,
    )
    assert result_ws.absorption_rate_constant > result_fatty.absorption_rate_constant


def test_notes_nonempty():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0)
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = predict_rectal_absorption("Diazepam", mw_Da=284.7, logp=2.8)
    assert result.drug_name == "Diazepam"


def test_formulation_preserved():
    result = predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0, formulation="enema")
    assert result.formulation == "enema"


def test_compare_rectal_formulations_returns_list_of_4():
    results = compare_rectal_formulations("TestDrug", mw_Da=300.0, logp=2.0)
    assert len(results) == 4


def test_compare_rectal_formulations_sorted_by_f_systemic_desc():
    results = compare_rectal_formulations("TestDrug", mw_Da=300.0, logp=2.0)
    for i in range(1, len(results)):
        assert results[i - 1].f_systemic >= results[i].f_systemic


def test_compare_rectal_formulations_all_results_valid():
    results = compare_rectal_formulations("TestDrug", mw_Da=300.0, logp=2.0)
    for r in results:
        assert isinstance(r, RectalAbsorptionResult)
        assert 0.0 < r.fa_rectal <= 0.9
        assert 0.0 <= r.f_systemic <= 1.0


def test_high_hepatic_extraction_lowers_systemic():
    result_high_cl = predict_rectal_absorption(
        "TestDrug",
        mw_Da=300.0,
        logp=2.0,
        cl_hepatic_L_per_h=80.0,
        q_hepatic_L_per_h=90.0,
    )
    result_low_cl = predict_rectal_absorption(
        "TestDrug",
        mw_Da=300.0,
        logp=2.0,
        cl_hepatic_L_per_h=5.0,
        q_hepatic_L_per_h=90.0,
    )
    assert result_high_cl.f_systemic < result_low_cl.f_systemic


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0, dose_mg=0.0)


def test_validation_invalid_formulation_raises():
    with pytest.raises(ValueError, match="formulation"):
        predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0, formulation="tablet")


def test_validation_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_rectal"):
        predict_rectal_absorption("TestDrug", mw_Da=300.0, logp=2.0, fu_rectal=0.0)
