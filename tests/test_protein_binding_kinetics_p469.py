"""Tests for Phase 469 — Plasma Protein Binding Kinetics."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.protein_binding_kinetics_p469 import (
    ProteinBindingKineticsResult,
    compare_proteins,
    predict_protein_binding_kinetics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROTEINS = ["albumin", "AAG", "lipoprotein"]
_PROTEIN_CONC_UM = {"albumin": 600.0, "AAG": 1.0, "lipoprotein": 0.5}


def _make(protein="albumin", kon=1e6, koff=1.0, drug="Drug"):
    return predict_protein_binding_kinetics(drug, protein, kon, koff)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _make()
    assert isinstance(result, ProteinBindingKineticsResult)


def test_result_is_frozen():
    result = _make()
    with pytest.raises((AttributeError, TypeError)):
        result.kd_uM = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# kd_uM correctness
# ---------------------------------------------------------------------------


def test_kd_uM_exact():
    kon = 2e6
    koff = 4.0
    result = _make(kon=kon, koff=koff)
    expected = koff / kon * 1e6  # 2.0 µM
    assert abs(result.kd_uM - expected) < 1e-10


def test_kd_uM_small():
    kon = 1e8
    koff = 0.01
    result = _make(kon=kon, koff=koff)
    expected = 0.01 / 1e8 * 1e6  # 1e-4 µM
    assert abs(result.kd_uM - expected) < 1e-15


# ---------------------------------------------------------------------------
# fu_equilibrium
# ---------------------------------------------------------------------------


def test_fu_in_range_albumin():
    result = _make(protein="albumin")
    assert 0.0 < result.fu_equilibrium < 1.0


def test_fu_in_range_aag():
    result = _make(protein="AAG")
    assert 0.0 < result.fu_equilibrium < 1.0


def test_fu_in_range_lipoprotein():
    result = _make(protein="lipoprotein")
    assert 0.0 < result.fu_equilibrium < 1.0


def test_fu_formula_albumin():
    """fu = 1/(1 + [P]/Kd) should match manual calculation."""
    kon = 1e6
    koff = 1.0
    kd_uM = koff / kon * 1e6  # 1.0 µM
    p_uM = 600.0
    expected_fu = 1.0 / (1.0 + p_uM / kd_uM)
    result = predict_protein_binding_kinetics("Drug", "albumin", kon, koff)
    assert abs(result.fu_equilibrium - expected_fu) < 1e-10


def test_fu_formula_aag():
    kon = 1e5
    koff = 0.5
    kd_uM = koff / kon * 1e6
    p_uM = 1.0
    expected_fu = 1.0 / (1.0 + p_uM / kd_uM)
    result = predict_protein_binding_kinetics("Drug", "AAG", kon, koff)
    assert abs(result.fu_equilibrium - expected_fu) < 1e-10


def test_higher_kon_lower_fu():
    """Tighter binding (higher kon, same koff) → lower fu."""
    r_low = _make(kon=1e5, koff=1.0)
    r_high = _make(kon=1e7, koff=1.0)
    assert r_high.fu_equilibrium < r_low.fu_equilibrium


def test_higher_koff_higher_fu():
    """Faster dissociation (higher koff, same kon) → higher Kd → higher fu."""
    r_slow = _make(kon=1e6, koff=0.01)
    r_fast = _make(kon=1e6, koff=10.0)
    assert r_fast.fu_equilibrium > r_slow.fu_equilibrium


# ---------------------------------------------------------------------------
# Kinetic time constants
# ---------------------------------------------------------------------------


def test_residence_time_exact():
    koff = 2.0
    result = _make(koff=koff)
    assert abs(result.residence_time_s - 1.0 / koff) < 1e-12


def test_time_to_equilibrium_exact():
    koff = 0.5
    result = _make(koff=koff)
    assert abs(result.time_to_equilibrium_s - 5.0 / koff) < 1e-12


def test_time_to_equilibrium_is_5x_residence():
    result = _make(koff=3.0)
    assert abs(result.time_to_equilibrium_s - 5.0 * result.residence_time_s) < 1e-12


# ---------------------------------------------------------------------------
# Binding classification
# ---------------------------------------------------------------------------


def test_binding_class_very_strong():
    # kd_uM < 0.01 µM → need koff/kon *1e6 < 0.01 → koff/kon < 1e-8
    # e.g. kon=1e9, koff=0.001 → kd_uM = 0.001/1e9*1e6 = 1e-6
    result = predict_protein_binding_kinetics("Drug", "albumin", 1e9, 0.001)
    assert result.binding_class == "very_strong"


def test_binding_class_strong():
    # 0.01 <= kd_uM < 1 → kd=0.1 µM → koff/kon*1e6=0.1 → kon=1e6, koff=0.1
    result = predict_protein_binding_kinetics("Drug", "albumin", 1e6, 0.1)
    kd = result.kd_uM
    assert 0.01 <= kd < 1.0
    assert result.binding_class == "strong"


def test_binding_class_moderate():
    # 1 <= kd_uM <= 100 → e.g. kd=10 µM → kon=1e6, koff=10 → kd=10/1e6*1e6=10
    result = predict_protein_binding_kinetics("Drug", "albumin", 1e6, 10.0)
    assert result.kd_uM == pytest.approx(10.0)
    assert result.binding_class == "moderate"


def test_binding_class_weak():
    # kd_uM > 100 → e.g. kon=1e4, koff=100 → kd=100/1e4*1e6=1e4
    result = predict_protein_binding_kinetics("Drug", "albumin", 1e4, 100.0)
    assert result.kd_uM > 100.0
    assert result.binding_class == "weak"


# ---------------------------------------------------------------------------
# Displacement risk
# ---------------------------------------------------------------------------


def test_displacement_risk_true_when_fu_low():
    # Tight albumin binding: high kon, low koff → very low fu
    result = predict_protein_binding_kinetics("Drug", "albumin", 1e9, 0.001)
    assert result.displacement_risk is True
    assert result.fu_equilibrium < 0.1


def test_displacement_risk_false_when_fu_high():
    # Weak binding: low kon → high Kd → high fu
    result = predict_protein_binding_kinetics("Drug", "albumin", 1e3, 1000.0)
    assert result.fu_equilibrium > 0.9
    assert result.displacement_risk is False


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = _make()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# compare_proteins
# ---------------------------------------------------------------------------


def test_compare_proteins_returns_three():
    results = compare_proteins("Drug", 1e6, 1.0)
    assert len(results) == 3


def test_compare_proteins_sorted_by_fu_ascending():
    results = compare_proteins("Drug", 1e6, 1.0)
    fus = [r.fu_equilibrium for r in results]
    assert fus == sorted(fus)


def test_compare_proteins_all_proteins_present():
    results = compare_proteins("Drug", 1e6, 1.0)
    proteins = {r.protein for r in results}
    assert proteins == {"albumin", "AAG", "lipoprotein"}


def test_compare_proteins_albumin_tightest():
    """Albumin has highest conc → lowest fu (tightest effective binding)."""
    results = compare_proteins("Drug", 1e6, 1.0)
    # albumin [P]=600 µM → smallest fu
    assert results[0].protein == "albumin"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_kon_zero():
    with pytest.raises(ValueError, match="kon_M_per_s"):
        predict_protein_binding_kinetics("Drug", "albumin", 0.0, 1.0)


def test_invalid_kon_negative():
    with pytest.raises(ValueError, match="kon_M_per_s"):
        predict_protein_binding_kinetics("Drug", "albumin", -1.0, 1.0)


def test_invalid_koff_zero():
    with pytest.raises(ValueError, match="koff_per_s"):
        predict_protein_binding_kinetics("Drug", "albumin", 1e6, 0.0)


def test_invalid_koff_negative():
    with pytest.raises(ValueError, match="koff_per_s"):
        predict_protein_binding_kinetics("Drug", "albumin", 1e6, -0.5)


def test_invalid_protein():
    with pytest.raises(ValueError, match="protein"):
        predict_protein_binding_kinetics("Drug", "globulin", 1e6, 1.0)
