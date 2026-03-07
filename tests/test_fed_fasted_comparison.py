"""Tests for fed_fasted_comparison — Phase 444."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.fed_fasted_comparison import (
    FedFastedResult,
    bcs_food_effect_prediction,
    simulate_fed_fasted_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_kwargs(**overrides):
    """Return default simulate_fed_fasted_pk keyword arguments."""
    kwargs = dict(
        drug_name="TestDrug",
        bcs_class=2,
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        fasted_solubility_mg_mL=0.05,
        fed_solubility_mg_mL=0.5,
        fasted_ka_per_h=0.5,
        fed_ka_per_h=0.8,
        fasted_f=0.4,
        fed_f=0.75,
        t_end_h=24.0,
        dt_h=0.1,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# FedFastedResult dataclass
# ---------------------------------------------------------------------------


def test_result_dataclass_default_values():
    r = FedFastedResult(drug_name="drug", bcs_class=1, dose_mg=100.0)
    assert r.times_h == []
    assert r.c_fasted_mg_L == []
    assert r.food_effect_classification == "no_effect"


def test_result_dataclass_mutability():
    """FedFastedResult is NOT frozen — fields can be updated."""
    r = FedFastedResult(drug_name="drug", bcs_class=1, dose_mg=100.0)
    r.notes = "updated"
    assert r.notes == "updated"


# ---------------------------------------------------------------------------
# Input validation — simulate_fed_fasted_pk
# ---------------------------------------------------------------------------


def test_empty_drug_name_raises():
    with pytest.raises(ValueError, match="drug_name"):
        simulate_fed_fasted_pk(**_default_kwargs(drug_name=""))


def test_invalid_bcs_class_raises():
    with pytest.raises(ValueError, match="bcs_class"):
        simulate_fed_fasted_pk(**_default_kwargs(bcs_class=5))


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_fed_fasted_pk(**_default_kwargs(dose_mg=0.0))


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_fed_fasted_pk(**_default_kwargs(cl_L_per_h=0.0))


def test_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_fed_fasted_pk(**_default_kwargs(vd_L=0.0))


def test_zero_fasted_solubility_raises():
    with pytest.raises(ValueError, match="fasted_solubility"):
        simulate_fed_fasted_pk(**_default_kwargs(fasted_solubility_mg_mL=0.0))


def test_zero_fed_solubility_raises():
    with pytest.raises(ValueError, match="fed_solubility"):
        simulate_fed_fasted_pk(**_default_kwargs(fed_solubility_mg_mL=0.0))


def test_fasted_f_out_of_range_raises():
    with pytest.raises(ValueError, match="fasted_f"):
        simulate_fed_fasted_pk(**_default_kwargs(fasted_f=0.0))


def test_fed_f_out_of_range_raises():
    with pytest.raises(ValueError, match="fed_f"):
        simulate_fed_fasted_pk(**_default_kwargs(fed_f=1.5))


def test_dt_ge_tend_raises():
    with pytest.raises(ValueError, match="dt_h"):
        simulate_fed_fasted_pk(**_default_kwargs(t_end_h=1.0, dt_h=2.0))


# ---------------------------------------------------------------------------
# simulate_fed_fasted_pk — basic outputs
# ---------------------------------------------------------------------------


def test_simulation_returns_fed_fasted_result():
    res = simulate_fed_fasted_pk(**_default_kwargs())
    assert isinstance(res, FedFastedResult)


def test_simulation_profile_lengths_equal():
    res = simulate_fed_fasted_pk(**_default_kwargs())
    assert len(res.times_h) == len(res.c_fasted_mg_L) == len(res.c_fed_mg_L)


def test_simulation_profile_non_empty():
    res = simulate_fed_fasted_pk(**_default_kwargs())
    assert len(res.times_h) > 0


def test_simulation_auc_positive():
    res = simulate_fed_fasted_pk(**_default_kwargs())
    assert res.auc_fasted > 0.0
    assert res.auc_fed > 0.0


def test_simulation_cmax_positive():
    res = simulate_fed_fasted_pk(**_default_kwargs())
    assert res.cmax_fasted > 0.0
    assert res.cmax_fed > 0.0


def test_simulation_concentrations_non_negative():
    res = simulate_fed_fasted_pk(**_default_kwargs())
    assert all(c >= 0.0 for c in res.c_fasted_mg_L)
    assert all(c >= 0.0 for c in res.c_fed_mg_L)


def test_simulation_drug_name_stored():
    res = simulate_fed_fasted_pk(**_default_kwargs(drug_name="Ibuprofen"))
    assert res.drug_name == "Ibuprofen"


def test_simulation_bcs_class_stored():
    res = simulate_fed_fasted_pk(**_default_kwargs(bcs_class=2))
    assert res.bcs_class == 2


# ---------------------------------------------------------------------------
# Food effect classification
# ---------------------------------------------------------------------------


def test_positive_food_effect_classification():
    """High fed_f vs low fasted_f should give positive food effect."""
    res = simulate_fed_fasted_pk(**_default_kwargs(fasted_f=0.2, fed_f=0.9))
    assert res.auc_ratio_fed_fasted > 1.25
    assert res.food_effect_classification == "positive"


def test_negative_food_effect_classification():
    """Very low fed_f should give negative food effect."""
    res = simulate_fed_fasted_pk(**_default_kwargs(fasted_f=0.9, fed_f=0.15))
    assert res.food_effect_classification == "negative"


def test_no_effect_food_classification():
    """Similar F values within ±25% should give no_effect."""
    res = simulate_fed_fasted_pk(**_default_kwargs(fasted_f=0.6, fed_f=0.65))
    assert res.food_effect_classification == "no_effect"


def test_auc_ratio_stored():
    res = simulate_fed_fasted_pk(**_default_kwargs())
    expected = res.auc_fed / res.auc_fasted
    assert res.auc_ratio_fed_fasted == pytest.approx(expected, rel=1e-6)


def test_cmax_ratio_stored():
    res = simulate_fed_fasted_pk(**_default_kwargs())
    expected = res.cmax_fed / res.cmax_fasted
    assert res.cmax_ratio == pytest.approx(expected, rel=1e-6)


def test_tmax_fed_greater_than_fasted_when_ka_similar():
    """Fed state has a longer lag time → Tmax should be later."""
    res = simulate_fed_fasted_pk(
        **_default_kwargs(fasted_ka_per_h=1.0, fed_ka_per_h=1.0, fasted_f=0.6, fed_f=0.6)
    )
    # Tmax_fed should be >= Tmax_fasted due to 1.5 h vs 0.25 h lag
    assert res.tmax_fed_h >= res.tmax_fasted_h


# ---------------------------------------------------------------------------
# bcs_food_effect_prediction
# ---------------------------------------------------------------------------


def test_bcs_prediction_returns_dict():
    result = bcs_food_effect_prediction(bcs_class=2, logP=4.0, solubility_mg_mL=0.05, dose_mg=100.0)
    assert isinstance(result, dict)


def test_bcs_i_minimal_effect():
    result = bcs_food_effect_prediction(bcs_class=1, logP=1.0, solubility_mg_mL=5.0, dose_mg=100.0)
    assert result["predicted_direction"] == "minimal"


def test_bcs_ii_lipophilic_positive():
    result = bcs_food_effect_prediction(bcs_class=2, logP=4.5, solubility_mg_mL=0.02, dose_mg=100.0)
    assert result["predicted_direction"] == "positive"


def test_bcs_ii_low_logp_variable():
    result = bcs_food_effect_prediction(bcs_class=2, logP=1.5, solubility_mg_mL=0.1, dose_mg=100.0)
    assert result["predicted_direction"] == "variable"


def test_bcs_iii_minimal():
    result = bcs_food_effect_prediction(bcs_class=3, logP=0.5, solubility_mg_mL=10.0, dose_mg=100.0)
    assert result["predicted_direction"] in ("minimal", "negative")


def test_bcs_iv_variable():
    result = bcs_food_effect_prediction(bcs_class=4, logP=1.0, solubility_mg_mL=0.01, dose_mg=100.0)
    assert result["predicted_direction"] == "variable"


def test_bcs_prediction_invalid_class_raises():
    with pytest.raises(ValueError, match="bcs_class"):
        bcs_food_effect_prediction(bcs_class=5, logP=2.0, solubility_mg_mL=1.0, dose_mg=100.0)


def test_bcs_prediction_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        bcs_food_effect_prediction(bcs_class=1, logP=1.0, solubility_mg_mL=1.0, dose_mg=0.0)


def test_bcs_prediction_zero_solubility_raises():
    with pytest.raises(ValueError, match="solubility_mg_mL"):
        bcs_food_effect_prediction(bcs_class=2, logP=3.0, solubility_mg_mL=0.0, dose_mg=100.0)


def test_bcs_prediction_contains_expected_keys():
    result = bcs_food_effect_prediction(bcs_class=2, logP=4.0, solubility_mg_mL=0.05, dose_mg=100.0)
    for key in ("bcs_class", "predicted_direction", "mechanism", "confidence", "recommendation"):
        assert key in result


def test_bcs_prediction_dose_number_in_result():
    result = bcs_food_effect_prediction(bcs_class=2, logP=4.0, solubility_mg_mL=0.1, dose_mg=100.0)
    assert "dose_number" in result
    expected_dn = 100.0 / (0.1 * 250.0)
    assert result["dose_number"] == pytest.approx(expected_dn, rel=1e-4)


def test_bcs_i_high_confidence():
    result = bcs_food_effect_prediction(bcs_class=1, logP=1.0, solubility_mg_mL=5.0, dose_mg=50.0)
    assert result["confidence"] == "high"


def test_bcs_ii_lipophilic_high_confidence():
    result = bcs_food_effect_prediction(bcs_class=2, logP=5.0, solubility_mg_mL=0.01, dose_mg=100.0)
    assert result["confidence"] == "high"
