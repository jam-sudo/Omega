"""Tests for Phase 683 — OAT PK sensitivity (compute_ota_sensitivity, etc.)."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.pk_parameter_sensitivity import (
    PKSensitivityResult683,
    analyze_pk_sensitivity,
    compute_ota_sensitivity,
    rank_pk_parameters,
    sensitivity_tornado_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auc_fn_1cpt(params: dict) -> float:
    """Simple analytical AUC = F * dose / CL for a 1-cpt model."""
    return params["f"] * params["dose"] / params["CL"]


def _cmax_fn_1cpt(params: dict) -> float:
    """Approximate Cmax ~ f * dose / Vd for a 1-cpt oral model."""
    return params["f"] * params["dose"] / params["Vd"]


BASE_PARAMS_AUC = {"f": 0.8, "dose": 100.0, "CL": 5.0, "Vd": 50.0, "ka": 1.0}


# ---------------------------------------------------------------------------
# compute_ota_sensitivity
# ---------------------------------------------------------------------------


def test_compute_ota_sensitivity_returns_dict():
    result = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL", "Vd", "ka"], _auc_fn_1cpt)
    assert isinstance(result, dict)


def test_compute_ota_sensitivity_correct_keys():
    result = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL", "Vd", "ka"], _auc_fn_1cpt)
    assert set(result.keys()) == {"CL", "Vd", "ka"}


def test_cl_sensitivity_for_auc_is_negative():
    """AUC = F*D/CL -> S(CL) < 0 (higher CL -> lower AUC)."""
    result = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL"], _auc_fn_1cpt)
    assert result["CL"] < 0.0


def test_cl_sensitivity_for_auc_approx_neg_one():
    """AUC = F*D/CL is inversely proportional to CL.

    Central-difference with delta=0.1 gives S = -1/(1-delta^2) ≈ -1.0101.
    The magnitude must be close to 1.0 (within 2%).
    """
    result = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL"], _auc_fn_1cpt)
    assert abs(abs(result["CL"]) - 1.0) < 0.02


def test_vd_sensitivity_for_auc_is_near_zero():
    """AUC = F*D/CL is independent of Vd -> S(Vd) ≈ 0."""
    result = compute_ota_sensitivity(BASE_PARAMS_AUC, ["Vd"], _auc_fn_1cpt)
    assert abs(result["Vd"]) < 1e-6


def test_ka_sensitivity_for_auc_is_zero():
    """AUC = F*D/CL is independent of ka -> S(ka) ≈ 0."""
    result = compute_ota_sensitivity(BASE_PARAMS_AUC, ["ka"], _auc_fn_1cpt)
    assert abs(result["ka"]) < 1e-6


def test_cmax_sensitivity_vd_negative():
    """Cmax ≈ F*D/Vd -> S(Vd) < 0."""
    result = compute_ota_sensitivity(BASE_PARAMS_AUC, ["Vd"], _cmax_fn_1cpt)
    assert result["Vd"] < 0.0


def test_sorted_by_abs_descending():
    result = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL", "Vd", "ka"], _auc_fn_1cpt)
    values = list(result.values())
    abs_values = [abs(v) for v in values]
    assert abs_values == sorted(abs_values, reverse=True)


def test_delta_fraction_parameter():
    """Different delta_fraction should give CL sensitivity close to -1 in magnitude."""
    result_10 = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL"], _auc_fn_1cpt, delta_fraction=0.1)
    result_5 = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL"], _auc_fn_1cpt, delta_fraction=0.05)
    # Both should be negative and magnitude within 2% of 1.0
    assert result_10["CL"] < 0.0
    assert result_5["CL"] < 0.0
    assert abs(abs(result_10["CL"]) - 1.0) < 0.02
    assert abs(abs(result_5["CL"]) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# rank_pk_parameters
# ---------------------------------------------------------------------------


def test_rank_pk_parameters_returns_list_of_tuples():
    sens = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL", "Vd", "ka"], _auc_fn_1cpt)
    ranked = rank_pk_parameters(sens)
    assert isinstance(ranked, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)


def test_rank_pk_parameters_sorted_descending():
    sens = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL", "Vd", "ka"], _auc_fn_1cpt)
    ranked = rank_pk_parameters(sens)
    abs_vals = [abs(v) for _, v in ranked]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_rank_pk_parameters_cl_is_first():
    """CL has the highest |S| for AUC."""
    sens = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL", "Vd", "ka"], _auc_fn_1cpt)
    ranked = rank_pk_parameters(sens)
    assert ranked[0][0] == "CL"


# ---------------------------------------------------------------------------
# sensitivity_tornado_data
# ---------------------------------------------------------------------------


def test_sensitivity_tornado_data_returns_list():
    sens = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL", "Vd", "ka"], _auc_fn_1cpt)
    data = sensitivity_tornado_data(sens)
    assert isinstance(data, list)


def test_sensitivity_tornado_data_has_correct_keys():
    sens = compute_ota_sensitivity(BASE_PARAMS_AUC, ["CL", "Vd"], _auc_fn_1cpt)
    data = sensitivity_tornado_data(sens)
    for entry in data:
        assert "param" in entry
        assert "sensitivity" in entry
        assert "direction" in entry


def test_sensitivity_tornado_data_direction_negative():
    """CL sensitivity for AUC is negative -> direction='negative'."""
    sens = {"CL": -1.0, "Vd": 0.0}
    data = sensitivity_tornado_data(sens)
    cl_entry = next(e for e in data if e["param"] == "CL")
    assert cl_entry["direction"] == "negative"


def test_sensitivity_tornado_data_direction_positive():
    """F sensitivity for AUC is positive -> direction='positive'."""
    sens = {"f": 1.0}
    data = sensitivity_tornado_data(sens)
    assert data[0]["direction"] == "positive"


# ---------------------------------------------------------------------------
# analyze_pk_sensitivity
# ---------------------------------------------------------------------------


def test_analyze_pk_sensitivity_returns_result():
    result = analyze_pk_sensitivity("DrugA", 100.0, 5.0, 50.0, 1.5)
    assert isinstance(result, PKSensitivityResult683)


def test_analyze_pk_sensitivity_compound_name():
    result = analyze_pk_sensitivity("TestCompound", 100.0, 5.0, 50.0, 1.5)
    assert result.compound_name == "TestCompound"


def test_analyze_pk_sensitivity_cl_is_most_sensitive_for_auc():
    """For AUC from 1-cpt oral, CL dominates (S≈-1), Vd and ka ≈0."""
    result = analyze_pk_sensitivity("DrugB", 100.0, 5.0, 50.0, 1.5, t_end_h=48.0)
    assert result.most_sensitive_param == "cl_L_per_h"


def test_analyze_pk_sensitivity_cl_sensitivity_negative():
    result = analyze_pk_sensitivity("DrugC", 100.0, 5.0, 50.0, 1.5)
    assert result.cl_sensitivity < 0.0


def test_analyze_pk_sensitivity_base_auc_positive():
    result = analyze_pk_sensitivity("DrugD", 100.0, 5.0, 50.0, 1.5)
    assert result.base_auc > 0.0


def test_analyze_pk_sensitivity_base_cmax_positive():
    result = analyze_pk_sensitivity("DrugE", 100.0, 5.0, 50.0, 1.5)
    assert result.base_cmax > 0.0


def test_analyze_pk_sensitivity_notes_contains_compound():
    result = analyze_pk_sensitivity("Warfarin", 100.0, 5.0, 50.0, 1.5)
    assert "Warfarin" in result.notes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_analyze_pk_sensitivity_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        analyze_pk_sensitivity("Drug", 0.0, 5.0, 50.0, 1.5)


def test_analyze_pk_sensitivity_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        analyze_pk_sensitivity("Drug", -10.0, 5.0, 50.0, 1.5)


def test_analyze_pk_sensitivity_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        analyze_pk_sensitivity("Drug", 100.0, 0.0, 50.0, 1.5)


def test_analyze_pk_sensitivity_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        analyze_pk_sensitivity("Drug", 100.0, 5.0, 0.0, 1.5)


def test_analyze_pk_sensitivity_ka_zero_raises():
    with pytest.raises(ValueError, match="ka_per_h"):
        analyze_pk_sensitivity("Drug", 100.0, 5.0, 50.0, 0.0)


def test_analyze_pk_sensitivity_f_oral_zero_raises():
    with pytest.raises(ValueError, match="f_oral"):
        analyze_pk_sensitivity("Drug", 100.0, 5.0, 50.0, 1.5, f_oral=0.0)
