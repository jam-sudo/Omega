"""Tests for Phase 778 active metabolite predictor API (predict_active_metabolite).

Imports are all at module top level.
"""


import pytest

from omega_pbpk.prediction.active_metabolite_predictor import (
    ActiveMetaboliteResult,
    predict_active_metabolite,
    predict_metabolite_cascade,
)

# ── Return type ───────────────────────────────────────────────────────────────


def test_returns_active_metabolite_result():
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        parent_vd_L=50.0,
    )
    assert isinstance(result, ActiveMetaboliteResult)


# ── Field types ───────────────────────────────────────────────────────────────


def test_parent_name_stored():
    result = predict_active_metabolite(
        parent_name="MyDrug", parent_dose_mg=100.0, parent_cl_L_per_h=5.0, parent_vd_L=50.0
    )
    assert result.parent_name == "MyDrug"


def test_metabolite_name_stored():
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        parent_vd_L=50.0,
        metabolite_name="M2",
    )
    assert result.metabolite_name == "M2"


def test_pathway_stored():
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        parent_vd_L=50.0,
        pathway="CYP2D6",
    )
    assert result.pathway == "CYP2D6"


def test_formation_fraction_stored():
    result = predict_active_metabolite(
        parent_name="DrugA", parent_dose_mg=100.0, parent_cl_L_per_h=5.0, parent_vd_L=50.0, fm=0.4
    )
    assert result.formation_fraction == pytest.approx(0.4)


# ── Formula: t_half = 0.693 * Vd / CL ────────────────────────────────────────


def test_t_half_formula():
    met_cl = 8.0
    met_vd = 40.0
    expected = 0.693 * met_vd / met_cl
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        parent_vd_L=50.0,
        met_cl_L_per_h=met_cl,
        met_vd_L=met_vd,
    )
    assert result.metabolite_t_half_h == pytest.approx(expected, rel=1e-6)


# ── Formula: auc_ratio = fm * parent_CL / met_CL ─────────────────────────────


def test_auc_ratio_formula():
    fm = 0.4
    parent_cl = 6.0
    met_cl = 12.0
    expected = fm * parent_cl / met_cl
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=parent_cl,
        parent_vd_L=50.0,
        fm=fm,
        met_cl_L_per_h=met_cl,
    )
    assert result.auc_ratio_met_parent == pytest.approx(expected, rel=1e-6)


# ── mist_relevant flag ────────────────────────────────────────────────────────


def test_mist_relevant_true_when_auc_ratio_above_0_1():
    # fm=0.5, parent_CL=10, met_CL=5 → ratio=1.0 > 0.1
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=10.0,
        parent_vd_L=50.0,
        fm=0.5,
        met_cl_L_per_h=5.0,
    )
    assert result.mist_relevant is True


def test_mist_relevant_false_when_auc_ratio_below_0_1():
    # fm=0.01, parent_CL=5, met_CL=100 → ratio=0.0005 < 0.1
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        parent_vd_L=50.0,
        fm=0.01,
        met_cl_L_per_h=100.0,
    )
    assert result.mist_relevant is False


# ── pharmacological_activity field ────────────────────────────────────────────


def test_pharmacological_activity_active():
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        parent_vd_L=50.0,
        pharmacological_activity="active",
    )
    assert result.pharmacological_activity == "active"


def test_pharmacological_activity_inactive():
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        parent_vd_L=50.0,
        pharmacological_activity="inactive",
    )
    assert result.pharmacological_activity == "inactive"


def test_pharmacological_activity_toxic():
    result = predict_active_metabolite(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        parent_vd_L=50.0,
        pharmacological_activity="toxic",
    )
    assert result.pharmacological_activity == "toxic"


def test_pharmacological_activity_in_valid_set():
    result = predict_active_metabolite(
        parent_name="DrugA", parent_dose_mg=100.0, parent_cl_L_per_h=5.0, parent_vd_L=50.0
    )
    assert result.pharmacological_activity in {"active", "inactive", "toxic"}


# ── Validation ────────────────────────────────────────────────────────────────


def test_fm_above_1_raises_value_error():
    with pytest.raises(ValueError):
        predict_active_metabolite(
            parent_name="DrugA",
            parent_dose_mg=100.0,
            parent_cl_L_per_h=5.0,
            parent_vd_L=50.0,
            fm=1.5,
        )


def test_fm_negative_raises_value_error():
    with pytest.raises(ValueError):
        predict_active_metabolite(
            parent_name="DrugA",
            parent_dose_mg=100.0,
            parent_cl_L_per_h=5.0,
            parent_vd_L=50.0,
            fm=-0.1,
        )


def test_met_cl_zero_raises_value_error():
    with pytest.raises(ValueError):
        predict_active_metabolite(
            parent_name="DrugA",
            parent_dose_mg=100.0,
            parent_cl_L_per_h=5.0,
            parent_vd_L=50.0,
            met_cl_L_per_h=0.0,
        )


def test_met_cl_negative_raises_value_error():
    with pytest.raises(ValueError):
        predict_active_metabolite(
            parent_name="DrugA",
            parent_dose_mg=100.0,
            parent_cl_L_per_h=5.0,
            parent_vd_L=50.0,
            met_cl_L_per_h=-5.0,
        )


def test_invalid_pharmacological_activity_raises_value_error():
    with pytest.raises(ValueError):
        predict_active_metabolite(
            parent_name="DrugA",
            parent_dose_mg=100.0,
            parent_cl_L_per_h=5.0,
            parent_vd_L=50.0,
            pharmacological_activity="unknown",
        )


# ── predict_metabolite_cascade ────────────────────────────────────────────────


def test_cascade_returns_list():
    metabolites = [
        {
            "metabolite_name": "M1",
            "pathway": "CYP3A4",
            "fm": 0.3,
            "met_cl_L_per_h": 10.0,
            "met_vd_L": 30.0,
        },
        {
            "metabolite_name": "M2",
            "pathway": "CYP2D6",
            "fm": 0.2,
            "met_cl_L_per_h": 8.0,
            "met_vd_L": 25.0,
        },
    ]
    results = predict_metabolite_cascade(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        metabolites=metabolites,
    )
    assert isinstance(results, list)
    assert len(results) == 2


def test_cascade_returns_active_metabolite_results():
    metabolites = [
        {
            "metabolite_name": "M1",
            "pathway": "CYP3A4",
            "fm": 0.3,
            "met_cl_L_per_h": 10.0,
            "met_vd_L": 30.0,
        },
    ]
    results = predict_metabolite_cascade(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        metabolites=metabolites,
    )
    assert all(isinstance(r, ActiveMetaboliteResult) for r in results)


def test_cascade_empty_list():
    results = predict_metabolite_cascade(
        parent_name="DrugA",
        parent_dose_mg=100.0,
        parent_cl_L_per_h=5.0,
        metabolites=[],
    )
    assert results == []
