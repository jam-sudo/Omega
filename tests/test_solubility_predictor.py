"""Tests for prediction/solubility_predictor.py — Phase 169."""

import math
import pytest

from omega_pbpk.prediction.solubility_predictor import (
    SolubilityResult,
    predict_solubility,
    screen_solubility,
)


# ---------------------------------------------------------------------------
# Basic prediction
# ---------------------------------------------------------------------------

def test_basic_prediction_returns_result():
    r = predict_solubility("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", mw=180.16, logP=1.2)
    assert isinstance(r, SolubilityResult)
    assert r.compound_name == "Aspirin"
    assert r.smiles == "CC(=O)Oc1ccccc1C(=O)O"


def test_solubility_mg_mL_matches_log10():
    r = predict_solubility("DrugA", "C", mw=100, logP=1.0)
    assert pytest.approx(r.solubility_mg_mL, rel=1e-6) == 10.0 ** r.log10_solubility_mg_mL


def test_solubility_ug_mL_is_1000x_mg():
    r = predict_solubility("DrugA", "C", mw=200, logP=2.0)
    assert pytest.approx(r.solubility_ug_mL, rel=1e-6) == r.solubility_mg_mL * 1000.0


def test_solubility_uM_uses_mw():
    r = predict_solubility("DrugA", "C", mw=200, logP=1.0)
    expected_uM = (r.solubility_mg_mL / 200.0) * 1e6
    assert pytest.approx(r.solubility_uM, rel=1e-6) == expected_uM


# ---------------------------------------------------------------------------
# ESOL model behaviour
# ---------------------------------------------------------------------------

def test_higher_logP_lower_solubility():
    r_low = predict_solubility("Low", "C", mw=200, logP=1.0, n_aromatic_rings=0)
    r_high = predict_solubility("High", "C", mw=200, logP=4.0, n_aromatic_rings=0)
    assert r_high.solubility_mg_mL < r_low.solubility_mg_mL


def test_higher_mw_slightly_lower_solubility():
    r_light = predict_solubility("Light", "C", mw=100, logP=2.0)
    r_heavy = predict_solubility("Heavy", "C", mw=500, logP=2.0)
    assert r_heavy.solubility_mg_mL < r_light.solubility_mg_mL


def test_more_aromatic_rings_lower_solubility():
    r0 = predict_solubility("A0", "C", mw=200, logP=2.0, n_aromatic_rings=0)
    r3 = predict_solubility("A3", "C", mw=200, logP=2.0, n_aromatic_rings=3)
    assert r3.solubility_mg_mL < r0.solubility_mg_mL


def test_more_rotatable_bonds_higher_solubility():
    r0 = predict_solubility("R0", "C", mw=200, logP=2.0, n_rotatable_bonds=0)
    r10 = predict_solubility("R10", "C", mw=200, logP=2.0, n_rotatable_bonds=10)
    assert r10.solubility_mg_mL > r0.solubility_mg_mL


def test_esol_regression_formula():
    """Verify the exact ESOL formula output."""
    log10_S = 0.16 - 0.63 * 2.0 - 0.0062 * 300.0 + 0.066 * 5 - 0.74 * 1
    r = predict_solubility("X", "C", mw=300, logP=2.0, n_rotatable_bonds=5, n_aromatic_rings=1)
    assert pytest.approx(r.log10_solubility_mg_mL, abs=1e-8) == log10_S


# ---------------------------------------------------------------------------
# pH correction (Henderson-Hasselbalch)
# ---------------------------------------------------------------------------

def test_ph_correction_increases_solubility_for_acid():
    r_no = predict_solubility("Acid", "C", mw=200, logP=2.0)
    r_ph = predict_solubility("Acid", "C", mw=200, logP=2.0, pka_acid=4.0, ph=7.4)
    assert r_ph.solubility_mg_mL > r_no.solubility_mg_mL


def test_ph_correction_flag():
    r_no = predict_solubility("A", "C", mw=200, logP=2.0)
    r_ph = predict_solubility("A", "C", mw=200, logP=2.0, pka_acid=4.5, ph=7.0)
    assert r_no.ph_correction_applied is False
    assert r_ph.ph_correction_applied is True


def test_ph_correction_factor_value():
    """S_pH = S_intrinsic * (1 + 10^(pH - pKa))."""
    r_no = predict_solubility("A", "C", mw=200, logP=2.0)
    pka = 4.0
    ph = 7.4
    r_ph = predict_solubility("A", "C", mw=200, logP=2.0, pka_acid=pka, ph=ph)
    expected = r_no.solubility_mg_mL * (1.0 + 10.0 ** (ph - pka))
    assert pytest.approx(r_ph.solubility_mg_mL, rel=1e-6) == expected


def test_acid_at_low_ph_minimal_correction():
    r_no = predict_solubility("A", "C", mw=200, logP=2.0)
    r_low = predict_solubility("A", "C", mw=200, logP=2.0, pka_acid=4.0, ph=2.0)
    # At pH 2.0, pKa 4.0: factor = 1 + 10^(-2) ≈ 1.01
    assert r_low.solubility_mg_mL < r_no.solubility_mg_mL * 1.05


# ---------------------------------------------------------------------------
# Solubility classification
# ---------------------------------------------------------------------------

def test_solubility_class_practically_insoluble():
    # Very high logP => low solubility
    r = predict_solubility("Insol", "C", mw=500, logP=6.0, n_aromatic_rings=3)
    assert r.solubility_class == "practically_insoluble"


def test_solubility_class_values_are_valid():
    for logP in [0.0, 1.0, 2.0, 3.0, 5.0]:
        r = predict_solubility("X", "C", mw=200, logP=logP)
        assert r.solubility_class in (
            "practically_insoluble",
            "slightly_soluble",
            "sparingly_soluble",
            "soluble",
            "freely_soluble",
        )


def test_bcs_flag_low_for_insoluble():
    r = predict_solubility("Insol", "C", mw=500, logP=6.0, n_aromatic_rings=3)
    assert r.bcs_solubility_flag == "low"


def test_bcs_flag_high_for_soluble():
    r = predict_solubility("Sol", "C", mw=100, logP=0.0, n_aromatic_rings=0)
    assert r.bcs_solubility_flag == "high"


def test_bcs_flag_values():
    for logP in [0.0, 2.0, 5.0, 7.0]:
        r = predict_solubility("X", "C", mw=200, logP=logP)
        assert r.bcs_solubility_flag in ("low", "high")


# ---------------------------------------------------------------------------
# screen_solubility
# ---------------------------------------------------------------------------

def test_screen_returns_sorted_ascending():
    compounds = [
        {"compound_name": "A", "smiles": "C", "mw": 200, "logP": 1.0},
        {"compound_name": "B", "smiles": "C", "mw": 200, "logP": 5.0},
        {"compound_name": "C", "smiles": "C", "mw": 200, "logP": 0.0},
    ]
    results = screen_solubility(compounds)
    sols = [r.solubility_mg_mL for r in results]
    assert sols == sorted(sols)


def test_screen_returns_correct_count():
    compounds = [
        {"compound_name": f"D{i}", "smiles": "C", "mw": 200, "logP": float(i)}
        for i in range(5)
    ]
    results = screen_solubility(compounds)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_solubility("X", "C", mw=0, logP=1.0)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_solubility("X", "C", mw=-10, logP=1.0)


def test_empty_compound_name():
    with pytest.raises(ValueError, match="compound_name"):
        predict_solubility("", "C", mw=200, logP=1.0)


def test_empty_smiles():
    with pytest.raises(ValueError, match="smiles"):
        predict_solubility("X", "", mw=200, logP=1.0)


def test_negative_rotatable_bonds():
    with pytest.raises(ValueError, match="n_rotatable_bonds"):
        predict_solubility("X", "C", mw=200, logP=1.0, n_rotatable_bonds=-1)


def test_negative_aromatic_rings():
    with pytest.raises(ValueError, match="n_aromatic_rings"):
        predict_solubility("X", "C", mw=200, logP=1.0, n_aromatic_rings=-1)


def test_screen_empty_list():
    with pytest.raises(ValueError, match="compounds"):
        screen_solubility([])


def test_result_is_frozen():
    r = predict_solubility("X", "C", mw=200, logP=1.0)
    with pytest.raises(AttributeError):
        r.solubility_mg_mL = 999.0  # type: ignore[misc]
