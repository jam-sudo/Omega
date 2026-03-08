"""Tests for Phase 456 — Pharmacophore Properties (pharmacophore_properties_p456)."""

import pytest

from omega_pbpk.prediction.pharmacophore_properties_p456 import (
    PharmacophoreResult,
    predict_pharmacophore,
    screen_pharmacophore,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_pharmacophore("DrugA", mw=300.0, logp=2.0, hbd=2, hba=5)
    assert isinstance(result, PharmacophoreResult)


def test_drug_name_stored():
    result = predict_pharmacophore("Aspirin", mw=180.0, logp=1.2, hbd=1, hba=4)
    assert result.drug_name == "Aspirin"


# ---------------------------------------------------------------------------
# Drug-like classification
# ---------------------------------------------------------------------------


def test_drug_like_molecule():
    """MW=400, logP=2, hbd=2, hba=5 should be drug_like."""
    result = predict_pharmacophore(
        "DrugLike", mw=400.0, logp=2.0, hbd=2, hba=5, rotatable_bonds=5, psa=80.0, charge=0
    )
    assert result.drug_likeness_class == "drug_like"


def test_fragment_like_molecule():
    """MW=200, logP=1 should be fragment_like."""
    result = predict_pharmacophore(
        "Fragment", mw=200.0, logp=1.0, hbd=1, hba=3, rotatable_bonds=3, psa=60.0, charge=0
    )
    assert result.drug_likeness_class == "fragment_like"


def test_lead_like_molecule():
    """MW=340, logP=2.5 should be lead_like (not fragment_like)."""
    result = predict_pharmacophore(
        "Lead", mw=340.0, logp=2.5, hbd=2, hba=4, rotatable_bonds=5, psa=80.0, charge=0
    )
    assert result.drug_likeness_class == "lead_like"


def test_beyond_rule_of_5():
    """MW=700, logP=6 should be beyond_rule_of_5."""
    result = predict_pharmacophore(
        "bRo5", mw=700.0, logp=6.0, hbd=8, hba=12, rotatable_bonds=15, psa=180.0, charge=0
    )
    assert result.drug_likeness_class == "beyond_rule_of_5"


# ---------------------------------------------------------------------------
# Lipinski violations
# ---------------------------------------------------------------------------


def test_no_lipinski_violations():
    result = predict_pharmacophore("Clean", mw=300.0, logp=2.0, hbd=2, hba=5)
    assert result.lipinski_violations == 0


def test_one_lipinski_violation_mw():
    result = predict_pharmacophore("BigMW", mw=550.0, logp=2.0, hbd=2, hba=5)
    assert result.lipinski_violations == 1


def test_two_lipinski_violations():
    result = predict_pharmacophore("TwoViol", mw=550.0, logp=6.0, hbd=2, hba=5)
    assert result.lipinski_violations == 2


def test_four_lipinski_violations():
    result = predict_pharmacophore("AllViol", mw=600.0, logp=6.0, hbd=6, hba=11)
    assert result.lipinski_violations == 4


# ---------------------------------------------------------------------------
# Veber compliance
# ---------------------------------------------------------------------------


def test_veber_compliant():
    result = predict_pharmacophore(
        "Veber", mw=300.0, logp=2.0, hbd=2, hba=5, rotatable_bonds=8, psa=100.0
    )
    assert result.veber_compliant is True


def test_not_veber_compliant_rotbonds():
    result = predict_pharmacophore(
        "Flexible", mw=300.0, logp=2.0, hbd=2, hba=5, rotatable_bonds=12, psa=80.0
    )
    assert result.veber_compliant is False


def test_not_veber_compliant_psa():
    result = predict_pharmacophore(
        "HighPSA", mw=300.0, logp=2.0, hbd=2, hba=5, rotatable_bonds=5, psa=150.0
    )
    assert result.veber_compliant is False


# ---------------------------------------------------------------------------
# Drug-likeness score
# ---------------------------------------------------------------------------


def test_score_in_range():
    result = predict_pharmacophore("DrugA", mw=300.0, logp=2.0, hbd=2, hba=5)
    assert 0.0 <= result.drug_likeness_score <= 100.0


def test_perfect_score_drug_like():
    result = predict_pharmacophore(
        "Perfect", mw=300.0, logp=2.0, hbd=2, hba=5, rotatable_bonds=5, psa=80.0
    )
    assert result.drug_likeness_score == pytest.approx(100.0)


def test_score_reduced_by_violations():
    r_clean = predict_pharmacophore("Clean", mw=300.0, logp=2.0, hbd=2, hba=5)
    r_viol = predict_pharmacophore("Viol", mw=600.0, logp=6.0, hbd=2, hba=5)
    assert r_viol.drug_likeness_score < r_clean.drug_likeness_score


def test_score_reduced_by_all_violations():
    """Four Lipinski violations + Veber non-compliant => score = 100 - 40 - 10 = 50."""
    result = predict_pharmacophore(
        "Worst", mw=700.0, logp=7.0, hbd=8, hba=12, rotatable_bonds=15, psa=200.0
    )
    assert result.drug_likeness_score == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Pharmacophore features
# ---------------------------------------------------------------------------


def test_hbond_donor_features_equals_hbd():
    result = predict_pharmacophore("DrugA", mw=300.0, logp=2.0, hbd=3, hba=5)
    assert result.hbond_donor_features == 3


def test_hbond_acceptor_features_equals_hba():
    result = predict_pharmacophore("DrugA", mw=300.0, logp=2.0, hbd=2, hba=7)
    assert result.hbond_acceptor_features == 7


def test_charged_features_when_charged():
    result = predict_pharmacophore("Charged", mw=300.0, logp=1.0, hbd=1, hba=3, charge=1)
    assert result.charged_features == 1


def test_no_charged_features_neutral():
    result = predict_pharmacophore("Neutral", mw=300.0, logp=1.0, hbd=1, hba=3, charge=0)
    assert result.charged_features == 0


def test_total_features_sum():
    result = predict_pharmacophore(
        "Check", mw=300.0, logp=2.0, hbd=2, hba=5, rotatable_bonds=5, psa=80.0, charge=1
    )
    expected = (
        result.hydrophobic_centers
        + result.aromatic_rings_approx
        + result.hbond_donor_features
        + result.hbond_acceptor_features
        + result.charged_features
    )
    assert result.total_features == expected


# ---------------------------------------------------------------------------
# screen_pharmacophore
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    drugs = [
        {"drug_name": "A", "mw": 300.0, "logp": 2.0, "hbd": 2, "hba": 5},
        {"drug_name": "B", "mw": 700.0, "logp": 6.0, "hbd": 8, "hba": 12},
    ]
    results = screen_pharmacophore(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_descending():
    drugs = [
        {
            "drug_name": "Low",
            "mw": 700.0,
            "logp": 6.0,
            "hbd": 8,
            "hba": 12,
            "rotatable_bonds": 15,
            "psa": 200.0,
        },
        {"drug_name": "High", "mw": 300.0, "logp": 2.0, "hbd": 2, "hba": 5},
    ]
    results = screen_pharmacophore(drugs)
    assert results[0].drug_likeness_score >= results[1].drug_likeness_score


def test_screen_empty_list():
    results = screen_pharmacophore([])
    assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_pharmacophore("Drug", mw=0.0, logp=2.0, hbd=1, hba=2)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_pharmacophore("Drug", mw=-100.0, logp=2.0, hbd=1, hba=2)


def test_invalid_logp_too_low():
    with pytest.raises(ValueError, match="logp"):
        predict_pharmacophore("Drug", mw=300.0, logp=-6.0, hbd=1, hba=2)


def test_invalid_logp_too_high():
    with pytest.raises(ValueError, match="logp"):
        predict_pharmacophore("Drug", mw=300.0, logp=11.0, hbd=1, hba=2)


def test_invalid_hbd_negative():
    with pytest.raises(ValueError, match="hbd"):
        predict_pharmacophore("Drug", mw=300.0, logp=2.0, hbd=-1, hba=2)


def test_invalid_psa_negative():
    with pytest.raises(ValueError, match="psa"):
        predict_pharmacophore("Drug", mw=300.0, logp=2.0, hbd=1, hba=2, psa=-5.0)
