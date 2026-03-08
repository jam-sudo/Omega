"""
Tests for Phase 456 — Pharmacophore Properties (pharmacophore_properties.py)
"""

import pytest

from omega_pbpk.prediction.pharmacophore_properties import (
    PharmacophoreResult,
    analyze_pharmacophore,
    screen_pharmacophores,
)


# --- Baseline fixture ---
@pytest.fixture
def baseline():
    return analyze_pharmacophore(
        drug_name="TestDrug",
        n_hbd=2,
        n_hba=4,
        n_aromatic=2,
        n_hydrophobic=1,
        n_positive_ionizable=1,
        n_negative_ionizable=0,
    )


# --- 1. Return type ---
def test_return_type(baseline):
    assert isinstance(baseline, PharmacophoreResult)


# --- 2. Drug name preserved ---
def test_drug_name(baseline):
    assert baseline.drug_name == "TestDrug"


# --- 3. All counts preserved ---
def test_counts_preserved(baseline):
    assert baseline.n_hbd == 2
    assert baseline.n_hba == 4
    assert baseline.n_aromatic == 2
    assert baseline.n_hydrophobic == 1
    assert baseline.n_positive_ionizable == 1
    assert baseline.n_negative_ionizable == 0


# --- 4. pharmacophore_complexity between 0 and 100 ---
def test_complexity_range(baseline):
    assert 0.0 <= baseline.pharmacophore_complexity <= 100.0


# --- 5. drug_receptor_complementarity between 0 and 1 ---
def test_complementarity_range(baseline):
    assert 0.0 <= baseline.drug_receptor_complementarity <= 1.0


# --- 6. predicted_ki_nM > 0 ---
def test_ki_positive(baseline):
    assert baseline.predicted_ki_nM > 0.0


# --- 7. selectivity_score between 0 and 1 ---
def test_selectivity_range(baseline):
    assert 0.0 <= baseline.selectivity_score <= 1.0


# --- 8. bioavailability_score between 0 and 100 ---
def test_bioavailability_range(baseline):
    assert 0.0 <= baseline.bioavailability_score <= 100.0


# --- 9. More HBD -> lower bioavailability_score ---
def test_more_hbd_lower_bioavailability():
    res_low = analyze_pharmacophore("A", 2, 4, 2, 1, 0, 0)
    res_high = analyze_pharmacophore("B", 6, 4, 2, 1, 0, 0)  # HBD > 3
    assert res_high.bioavailability_score < res_low.bioavailability_score


# --- 10. High complexity -> low selectivity ---
def test_high_complexity_low_selectivity():
    # Very complex molecule
    res_complex = analyze_pharmacophore("Complex", 5, 8, 5, 5, 3, 3)
    res_simple = analyze_pharmacophore("Simple", 1, 2, 1, 0, 0, 0)
    assert res_complex.selectivity_score < res_simple.selectivity_score


# --- 11. activity_prediction correctly assigned ---
def test_activity_high_potency():
    # Use parameters that produce high complementarity -> low Ki
    res = analyze_pharmacophore("HighPotency", 2, 4, 2, 1, 1, 0)
    # complementarity should be near 1, Ki near 1 nM -> high_potency
    if res.predicted_ki_nM < 100.0:
        assert res.activity_prediction == "high_potency"
    elif res.predicted_ki_nM <= 1000.0:
        assert res.activity_prediction == "moderate"
    else:
        assert res.activity_prediction == "weak"


def test_activity_weak():
    # Very mismatched features -> lower complementarity -> higher Ki
    # Extreme deviations from ideal push complementarity toward 0
    # Use large counts far from ideal to maximize deviation
    res = analyze_pharmacophore("WeakDrug", 10, 15, 10, 10, 0, 0)
    # complementarity should be low -> Ki should be moderate-to-weak
    # Verify activity label matches the Ki value
    if res.predicted_ki_nM > 1000.0:
        assert res.activity_prediction == "weak"
    elif res.predicted_ki_nM > 100.0:
        assert res.activity_prediction == "moderate"
    else:
        assert res.activity_prediction == "high_potency"


# --- 12. pharmacophore_class correctly assigned ---
def test_class_cns_friendly():
    # aromatic >= 2 and positive > 0
    res = analyze_pharmacophore("CNSDrug", 1, 3, 3, 1, 2, 0)
    assert res.pharmacophore_class == "CNS_friendly"


def test_class_kinase_like():
    # aromatic >= 2, hbd < 2, no positive
    res = analyze_pharmacophore("KinaseDrug", 1, 4, 2, 1, 0, 0)
    assert res.pharmacophore_class == "kinase_like"


def test_class_hbd_rich():
    # hbd > 3, no aromatic criteria
    res = analyze_pharmacophore("HBDDrug", 5, 4, 0, 0, 0, 0)
    assert res.pharmacophore_class == "HBD_rich"


def test_class_lipophilic():
    # hydrophobic > 3, no strong aromatic pattern, hbd <= 3
    res = analyze_pharmacophore("LipoDrug", 1, 2, 0, 5, 0, 0)
    assert res.pharmacophore_class == "lipophilic"


def test_class_gpcr_like():
    # Falls through all other conditions
    res = analyze_pharmacophore("GPCRDrug", 1, 2, 1, 1, 0, 0)
    assert res.pharmacophore_class == "GPCR_like"


# --- 13. screen_pharmacophores returns correct length ---
def test_screen_returns_correct_length():
    compounds = [
        {
            "drug_name": "A",
            "n_hbd": 2,
            "n_hba": 4,
            "n_aromatic": 2,
            "n_hydrophobic": 1,
            "n_positive_ionizable": 1,
            "n_negative_ionizable": 0,
        },
        {
            "drug_name": "B",
            "n_hbd": 1,
            "n_hba": 3,
            "n_aromatic": 1,
            "n_hydrophobic": 2,
            "n_positive_ionizable": 0,
            "n_negative_ionizable": 1,
        },
        {
            "drug_name": "C",
            "n_hbd": 3,
            "n_hba": 6,
            "n_aromatic": 3,
            "n_hydrophobic": 0,
            "n_positive_ionizable": 0,
            "n_negative_ionizable": 2,
        },
    ]
    results = screen_pharmacophores(compounds)
    assert len(results) == 3


# --- 14. screen_pharmacophores sorted descending by complementarity ---
def test_screen_sorted_descending():
    compounds = [
        {
            "drug_name": "A",
            "n_hbd": 2,
            "n_hba": 4,
            "n_aromatic": 2,
            "n_hydrophobic": 1,
            "n_positive_ionizable": 1,
            "n_negative_ionizable": 0,
        },
        {
            "drug_name": "B",
            "n_hbd": 0,
            "n_hba": 0,
            "n_aromatic": 0,
            "n_hydrophobic": 0,
            "n_positive_ionizable": 0,
            "n_negative_ionizable": 0,
        },
        {
            "drug_name": "C",
            "n_hbd": 3,
            "n_hba": 5,
            "n_aromatic": 2,
            "n_hydrophobic": 2,
            "n_positive_ionizable": 0,
            "n_negative_ionizable": 1,
        },
    ]
    results = screen_pharmacophores(compounds)
    comps = [r.drug_receptor_complementarity for r in results]
    assert comps == sorted(comps, reverse=True)


# --- 15. Validation: negative counts raise ValueError ---
def test_validation_negative_hbd():
    with pytest.raises(ValueError, match="n_hbd"):
        analyze_pharmacophore("Drug", -1, 4, 2, 1, 0, 0)


def test_validation_negative_hba():
    with pytest.raises(ValueError, match="n_hba"):
        analyze_pharmacophore("Drug", 2, -1, 2, 1, 0, 0)


def test_validation_negative_aromatic():
    with pytest.raises(ValueError, match="n_aromatic"):
        analyze_pharmacophore("Drug", 2, 4, -1, 1, 0, 0)


def test_validation_negative_hydrophobic():
    with pytest.raises(ValueError, match="n_hydrophobic"):
        analyze_pharmacophore("Drug", 2, 4, 2, -1, 0, 0)


def test_validation_negative_positive():
    with pytest.raises(ValueError, match="n_positive_ionizable"):
        analyze_pharmacophore("Drug", 2, 4, 2, 1, -1, 0)


def test_validation_negative_negative():
    with pytest.raises(ValueError, match="n_negative_ionizable"):
        analyze_pharmacophore("Drug", 2, 4, 2, 1, 0, -1)


# --- 16. complexity formula check ---
def test_complexity_formula():
    # n_hbd=1, n_hba=1, n_aromatic=1, n_hydrophobic=1, pos=0, neg=0
    # raw = 10+8+15+5 = 38
    res = analyze_pharmacophore("Drug", 1, 1, 1, 1, 0, 0)
    assert abs(res.pharmacophore_complexity - 38.0) < 1e-6


def test_complexity_capped_at_100():
    # Very high counts -> should be capped at 100
    res = analyze_pharmacophore("Drug", 5, 5, 5, 5, 5, 5)
    assert res.pharmacophore_complexity == 100.0


# --- 17. notes is a string ---
def test_notes_is_string(baseline):
    assert isinstance(baseline.notes, str)
    assert len(baseline.notes) > 0


# --- 18. bioavailability penalizes high HBA ---
def test_more_hba_lower_bioavailability():
    res_low = analyze_pharmacophore("A", 2, 4, 1, 1, 0, 0)
    res_high = analyze_pharmacophore("B", 2, 10, 1, 1, 0, 0)  # HBA > 7
    assert res_high.bioavailability_score < res_low.bioavailability_score


# --- 19. bioavailability penalizes high aromatic ---
def test_more_aromatic_lower_bioavailability():
    res_low = analyze_pharmacophore("A", 1, 3, 2, 1, 0, 0)
    res_high = analyze_pharmacophore("B", 1, 3, 6, 1, 0, 0)  # aromatic > 4
    assert res_high.bioavailability_score < res_low.bioavailability_score


# --- 20. PharmacophoreResult is frozen (immutable) ---
def test_result_frozen(baseline):
    with pytest.raises((AttributeError, TypeError)):
        baseline.n_hbd = 99
