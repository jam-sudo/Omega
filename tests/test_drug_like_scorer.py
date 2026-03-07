"""Tests for drug-likeness scoring (QED, MPO, Lipinski, Veber)."""

import pytest

from omega_pbpk.prediction.drug_like_scorer import (
    DrugLikenessResult,
    score_drug_likeness,
    screen_drug_likeness,
)

# ---------------------------------------------------------------------------
# Basic return-type and frozen tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = score_drug_likeness("Aspirin", mw_da=180.0, logp=1.2, hbd=1, hba=4, psa_A2=63.6)
    assert isinstance(result, DrugLikenessResult)


def test_result_is_frozen():
    result = score_drug_likeness("Aspirin", mw_da=180.0, logp=1.2, hbd=1, hba=4, psa_A2=63.6)
    with pytest.raises((AttributeError, TypeError)):
        result.compound_name = "changed"  # type: ignore[misc]


def test_compound_name_preserved():
    result = score_drug_likeness("Ibuprofen", mw_da=206.0, logp=3.5, hbd=1, hba=2, psa_A2=37.3)
    assert result.compound_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# QED range
# ---------------------------------------------------------------------------


def test_qed_in_range_typical_drug():
    result = score_drug_likeness("DrugA", mw_da=400.0, logp=2.0, hbd=1, hba=5, psa_A2=70.0)
    assert 0.0 <= result.qed_score <= 1.0


def test_qed_in_range_poor_drug():
    result = score_drug_likeness("HeavyDrug", mw_da=900.0, logp=8.0, hbd=10, hba=15, psa_A2=300.0)
    assert 0.0 <= result.qed_score <= 1.0


def test_qed_in_range_zero():
    # Extremely poor profile
    result = score_drug_likeness("Terrible", mw_da=1200.0, logp=12.0, hbd=20, hba=25, psa_A2=500.0)
    assert 0.0 <= result.qed_score <= 1.0


# ---------------------------------------------------------------------------
# MPO range
# ---------------------------------------------------------------------------


def test_mpo_in_range():
    result = score_drug_likeness("DrugA", mw_da=400.0, logp=2.0, hbd=1, hba=5, psa_A2=70.0)
    assert 0.0 <= result.mpo_score <= 6.0


def test_mpo_in_range_poor():
    result = score_drug_likeness("HeavyDrug", mw_da=900.0, logp=8.0, hbd=10, hba=15, psa_A2=300.0)
    assert 0.0 <= result.mpo_score <= 6.0


# ---------------------------------------------------------------------------
# High QED for drug-like molecule
# ---------------------------------------------------------------------------


def test_ideal_drug_high_qed():
    """MW=400, logP=2, HBD=1 should give high QED (>0.5)."""
    result = score_drug_likeness(
        "IdealDrug",
        mw_da=400.0,
        logp=2.0,
        hbd=1,
        hba=4,
        psa_A2=60.0,
        n_rotatable_bonds=3,
        n_aromatic_rings=2,
    )
    assert result.qed_score > 0.5


def test_ideal_drug_excellent_class():
    result = score_drug_likeness(
        "IdealDrug",
        mw_da=339.0,
        logp=2.4,
        hbd=0,
        hba=3,
        psa_A2=57.0,
        n_rotatable_bonds=3,
        n_aromatic_rings=1,
    )
    assert result.drug_likeness_class in ("excellent", "good")


# ---------------------------------------------------------------------------
# Lipinski violations
# ---------------------------------------------------------------------------


def test_lipinski_passes_drug_like():
    result = score_drug_likeness("Aspirin", mw_da=180.0, logp=1.2, hbd=1, hba=4, psa_A2=63.6)
    assert result.lipinski_passes is True
    assert result.lipinski_violations == 0


def test_lipinski_violation_mw():
    result = score_drug_likeness("HeavyDrug", mw_da=600.0, logp=2.0, hbd=1, hba=4, psa_A2=60.0)
    assert result.lipinski_violations >= 1
    assert result.lipinski_passes is False


def test_lipinski_violation_logp():
    result = score_drug_likeness("LipophilicDrug", mw_da=400.0, logp=6.0, hbd=1, hba=4, psa_A2=60.0)
    assert result.lipinski_violations >= 1


def test_lipinski_violation_hbd():
    result = score_drug_likeness("HBDDrug", mw_da=400.0, logp=2.0, hbd=7, hba=4, psa_A2=60.0)
    assert result.lipinski_violations >= 1


def test_lipinski_violation_hba():
    result = score_drug_likeness("HBADrug", mw_da=400.0, logp=2.0, hbd=1, hba=12, psa_A2=60.0)
    assert result.lipinski_violations >= 1


def test_lipinski_four_violations():
    result = score_drug_likeness("Terrible", mw_da=600.0, logp=6.0, hbd=7, hba=12, psa_A2=60.0)
    assert result.lipinski_violations == 4
    assert result.lipinski_passes is False


# ---------------------------------------------------------------------------
# Veber rules
# ---------------------------------------------------------------------------


def test_veber_passes():
    result = score_drug_likeness(
        "DrugA", mw_da=300.0, logp=2.0, hbd=1, hba=4, psa_A2=60.0, n_rotatable_bonds=5
    )
    assert result.veber_passes is True


def test_veber_fails_high_psa():
    result = score_drug_likeness(
        "DrugB", mw_da=300.0, logp=2.0, hbd=1, hba=4, psa_A2=150.0, n_rotatable_bonds=5
    )
    assert result.veber_passes is False


def test_veber_fails_high_rotbonds():
    result = score_drug_likeness(
        "DrugC", mw_da=300.0, logp=2.0, hbd=1, hba=4, psa_A2=60.0, n_rotatable_bonds=12
    )
    assert result.veber_passes is False


# ---------------------------------------------------------------------------
# Drug-likeness class
# ---------------------------------------------------------------------------


def test_drug_likeness_class_excellent():
    # Use the ideal QED parameters (gaussian peaks) → very high QED
    result = score_drug_likeness(
        "Ideal",
        mw_da=339.0,
        logp=2.4,
        hbd=0,
        hba=3,
        psa_A2=57.0,
        n_rotatable_bonds=3,
        n_aromatic_rings=1,
    )
    assert result.drug_likeness_class == "excellent"


def test_drug_likeness_class_poor():
    result = score_drug_likeness("Poor", mw_da=900.0, logp=9.0, hbd=10, hba=15, psa_A2=300.0)
    assert result.drug_likeness_class == "poor"


# ---------------------------------------------------------------------------
# Fragment complexity
# ---------------------------------------------------------------------------


def test_fragment_complexity_calculation():
    result = score_drug_likeness(
        "DrugA", mw_da=500.0, logp=2.0, hbd=1, hba=5, psa_A2=70.0, n_heavy_atoms=25
    )
    assert abs(result.fragment_complexity - 500.0 / 25.0) < 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_da"):
        score_drug_likeness("Bad", mw_da=0.0, logp=2.0, hbd=1, hba=4, psa_A2=60.0)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_da"):
        score_drug_likeness("Bad", mw_da=-100.0, logp=2.0, hbd=1, hba=4, psa_A2=60.0)


def test_hbd_negative_raises():
    with pytest.raises(ValueError, match="hbd"):
        score_drug_likeness("Bad", mw_da=300.0, logp=2.0, hbd=-1, hba=4, psa_A2=60.0)


# ---------------------------------------------------------------------------
# screen_drug_likeness
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "mw_da": 350.0, "logp": 2.0, "hbd": 1, "hba": 4, "psa_A2": 60.0},
        {"compound_name": "B", "mw_da": 500.0, "logp": 4.0, "hbd": 2, "hba": 6, "psa_A2": 90.0},
    ]
    results = screen_drug_likeness(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_qed_descending():
    compounds = [
        {
            "compound_name": "Poor",
            "mw_da": 900.0,
            "logp": 8.0,
            "hbd": 8,
            "hba": 12,
            "psa_A2": 250.0,
        },
        {
            "compound_name": "Ideal",
            "mw_da": 339.0,
            "logp": 2.4,
            "hbd": 0,
            "hba": 3,
            "psa_A2": 57.0,
            "n_rotatable_bonds": 3,
            "n_aromatic_rings": 1,
        },
    ]
    results = screen_drug_likeness(compounds)
    assert results[0].compound_name == "Ideal"
    assert results[0].qed_score >= results[1].qed_score
