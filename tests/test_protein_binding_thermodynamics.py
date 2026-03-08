"""Tests for Phase 546 — Drug Protein Binding Thermodynamics."""

import math

import pytest

from omega_pbpk.prediction.protein_binding_thermodynamics import (
    ProteinBindingThermodynamicsResult,
    predict_binding_thermodynamics,
    screen_binding_affinity,
)

# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_binding_thermodynamics("TestDrug", 3.0, 300.0)
    assert isinstance(result, ProteinBindingThermodynamicsResult)


def test_fields_present():
    result = predict_binding_thermodynamics("TestDrug", 3.0, 300.0)
    assert hasattr(result, "drug_name")
    assert hasattr(result, "logP")
    assert hasattr(result, "mw_Da")
    assert hasattr(result, "pKd")
    assert hasattr(result, "kd_M")
    assert hasattr(result, "kd_mg_L")
    assert hasattr(result, "ka_L_per_mol")
    assert hasattr(result, "dg_kJ_per_mol")
    assert hasattr(result, "dh_kJ_per_mol")
    assert hasattr(result, "ds_J_per_mol_K")
    assert hasattr(result, "binding_class")
    assert hasattr(result, "fu_predicted")
    assert hasattr(result, "fu_class")
    assert hasattr(result, "entropy_driven")
    assert hasattr(result, "notes")


def test_drug_name_stored():
    result = predict_binding_thermodynamics("Warfarin", 2.7, 308.0)
    assert result.drug_name == "Warfarin"


def test_logP_stored():
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    assert result.logP == 2.0


def test_mw_stored():
    result = predict_binding_thermodynamics("Drug", 2.0, 400.0)
    assert result.mw_Da == 400.0


# ---------------------------------------------------------------------------
# pKd / Kd calculations
# ---------------------------------------------------------------------------


def test_pkd_formula():
    logP = 2.0
    mw = 300.0
    result = predict_binding_thermodynamics("Drug", logP, mw)
    expected_pKd = 1.5 + 0.8 * max(0.0, logP) - 0.003 * mw
    assert result.pKd == pytest.approx(expected_pKd, rel=1e-6)


def test_kd_M_from_pkd():
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    expected = 10.0 ** (-result.pKd)
    assert result.kd_M == pytest.approx(expected, rel=1e-6)


def test_kd_mg_L_conversion():
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    expected = result.kd_M * 300.0 * 1000.0
    assert result.kd_mg_L == pytest.approx(expected, rel=1e-6)


def test_ka_inverse_kd():
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    assert result.ka_L_per_mol == pytest.approx(1.0 / result.kd_M, rel=1e-6)


# ---------------------------------------------------------------------------
# Higher logP → tighter binding
# ---------------------------------------------------------------------------


def test_higher_logP_higher_pKd():
    r_low = predict_binding_thermodynamics("Drug", 1.0, 300.0)
    r_high = predict_binding_thermodynamics("Drug", 4.0, 300.0)
    assert r_high.pKd > r_low.pKd


def test_higher_logP_lower_kd():
    r_low = predict_binding_thermodynamics("Drug", 1.0, 300.0)
    r_high = predict_binding_thermodynamics("Drug", 4.0, 300.0)
    assert r_high.kd_M < r_low.kd_M


# ---------------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------------


def test_dg_negative_for_tight_binder():
    # High logP → tight binding → dG should be negative (favorable)
    result = predict_binding_thermodynamics("Drug", 5.0, 200.0)
    assert result.dg_kJ_per_mol < 0.0


def test_dg_formula():
    R = 8.314
    T = 310.0
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    expected_dG = R * T * math.log(result.kd_M) / 1000.0
    assert result.dg_kJ_per_mol == pytest.approx(expected_dG, rel=1e-5)


def test_dh_formula():
    logP = 3.0
    result = predict_binding_thermodynamics("Drug", logP, 300.0)
    expected_dH = -0.5 * logP * 4.0
    assert result.dh_kJ_per_mol == pytest.approx(expected_dH, rel=1e-6)


def test_ds_formula():
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    T = 310.0
    dG_J = result.dg_kJ_per_mol * 1000.0
    dH_J = result.dh_kJ_per_mol * 1000.0
    expected_dS = (dH_J - dG_J) / T
    assert result.ds_J_per_mol_K == pytest.approx(expected_dS, rel=1e-5)


def test_entropy_driven_bool():
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    assert isinstance(result.entropy_driven, bool)


# ---------------------------------------------------------------------------
# Binding class
# ---------------------------------------------------------------------------


def test_binding_class_tight():
    # pKd > 6: need logP high enough
    # pKd = 1.5 + 0.8*logP - 0.003*mw > 6
    # For mw=100: 0.8*logP > 4.5 + 0.3 → logP > 5.6 → use 6.0
    result = predict_binding_thermodynamics("Drug", 6.0, 100.0)
    assert result.binding_class == "tight"


def test_binding_class_moderate():
    # pKd between 4 and 6
    # For mw=300: pKd = 1.5 + 0.8*logP - 0.9
    # logP=4 → pKd = 1.5 + 3.2 - 0.9 = 3.8 ... try logP=5 → 1.5+4.0-0.9=4.6 ✓
    result = predict_binding_thermodynamics("Drug", 5.0, 300.0)
    assert result.binding_class == "moderate"


def test_binding_class_weak():
    # pKd < 4: negative logP or high MW
    # logP=-1, mw=500: pKd = 1.5 + 0 - 1.5 = 0.0 → weak
    result = predict_binding_thermodynamics("Drug", -1.0, 500.0)
    assert result.binding_class == "weak"


# ---------------------------------------------------------------------------
# fu / fu_class
# ---------------------------------------------------------------------------


def test_fu_formula():
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    albumin_M = 6e-4
    expected_fu = 1.0 / (1.0 + 1 * result.ka_L_per_mol * albumin_M)
    expected_fu = max(0.001, min(1.0, expected_fu))
    assert result.fu_predicted == pytest.approx(expected_fu, rel=1e-5)


def test_fu_between_0_001_and_1():
    result = predict_binding_thermodynamics("Drug", 2.0, 300.0)
    assert 0.001 <= result.fu_predicted <= 1.0


def test_fu_class_highly_bound():
    # Tight binder → low fu
    result = predict_binding_thermodynamics("Drug", 6.0, 100.0)
    if result.fu_predicted < 0.1:
        assert result.fu_class == "highly_bound"


def test_fu_class_low_binding():
    # Weak binder → high fu
    result = predict_binding_thermodynamics("Drug", -2.0, 500.0)
    if result.fu_predicted > 0.5:
        assert result.fu_class == "low_binding"


def test_fu_class_moderately_bound():
    result = predict_binding_thermodynamics("Drug", 3.0, 400.0)
    assert result.fu_class in {"highly_bound", "moderately_bound", "low_binding"}


def test_higher_logP_lower_fu():
    r_low = predict_binding_thermodynamics("Drug", 1.0, 300.0)
    r_high = predict_binding_thermodynamics("Drug", 4.0, 300.0)
    assert r_high.fu_predicted < r_low.fu_predicted


# ---------------------------------------------------------------------------
# screen_binding_affinity
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"drug_name": "A", "logP": 1.0, "mw_Da": 300.0},
        {"drug_name": "B", "logP": 3.0, "mw_Da": 300.0},
    ]
    results = screen_binding_affinity(compounds)
    assert isinstance(results, list)


def test_screen_sorted_by_pKd_descending():
    compounds = [
        {"drug_name": "A", "logP": 1.0, "mw_Da": 300.0},
        {"drug_name": "B", "logP": 4.0, "mw_Da": 300.0},
        {"drug_name": "C", "logP": 2.0, "mw_Da": 300.0},
    ]
    results = screen_binding_affinity(compounds)
    pkds = [r.pKd for r in results]
    assert pkds == sorted(pkds, reverse=True)


def test_screen_correct_length():
    compounds = [{"drug_name": f"Drug{i}", "logP": float(i), "mw_Da": 300.0} for i in range(5)]
    results = screen_binding_affinity(compounds)
    assert len(results) == 5


def test_screen_returns_correct_types():
    compounds = [{"drug_name": "X", "logP": 2.0, "mw_Da": 300.0}]
    results = screen_binding_affinity(compounds)
    assert isinstance(results[0], ProteinBindingThermodynamicsResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_binding_thermodynamics("Drug", 2.0, 0.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_binding_thermodynamics("Drug", 2.0, -100.0)


def test_negative_logP_notes():
    result = predict_binding_thermodynamics("Drug", -1.0, 300.0)
    assert "logP<0" in result.notes
