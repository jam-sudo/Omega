"""Tests for Phase 215: chemical_space_explorer."""

import pytest

from omega_pbpk.prediction.chemical_space_explorer import (
    DrugLikenessResult,
    evaluate_drug_likeness,
    screen_compound_library,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASPIRIN_KWARGS = dict(
    drug_name="Aspirin",
    mw_Da=180.0,
    logP=1.2,
    hbd=1,
    hba=3,
    psa_A2=57.0,
    rotatable_bonds=3,
)


def aspirin_result() -> DrugLikenessResult:
    return evaluate_drug_likeness(**ASPIRIN_KWARGS)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = aspirin_result()
    assert isinstance(result, DrugLikenessResult)


def test_frozen_dataclass():
    result = aspirin_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = aspirin_result()
    assert result.drug_name == "Aspirin"


def test_mw_preserved():
    result = aspirin_result()
    assert result.mw_Da == 180.0


def test_logP_preserved():
    result = aspirin_result()
    assert result.logP == 1.2


def test_hbd_preserved():
    result = aspirin_result()
    assert result.hbd == 1


def test_hba_preserved():
    result = aspirin_result()
    assert result.hba == 3


def test_psa_preserved():
    result = aspirin_result()
    assert result.psa_A2 == 57.0


def test_rotatable_bonds_preserved():
    result = aspirin_result()
    assert result.rotatable_bonds == 3


# ---------------------------------------------------------------------------
# Lipinski Ro5 violations
# ---------------------------------------------------------------------------


def test_aspirin_zero_ro5_violations():
    result = aspirin_result()
    assert result.ro5_violations == 0


def test_large_mw_violation():
    result = evaluate_drug_likeness(
        drug_name="BigDrug",
        mw_Da=600.0,
        logP=1.0,
        hbd=1,
        hba=3,
        psa_A2=90.0,
        rotatable_bonds=5,
    )
    assert result.ro5_violations >= 1


def test_high_logP_violation():
    result = evaluate_drug_likeness(
        drug_name="HighLogP",
        mw_Da=300.0,
        logP=6.0,
        hbd=1,
        hba=3,
        psa_A2=90.0,
        rotatable_bonds=5,
    )
    assert result.ro5_violations >= 1


def test_high_hbd_violation():
    result = evaluate_drug_likeness(
        drug_name="HighHBD",
        mw_Da=300.0,
        logP=1.0,
        hbd=6,
        hba=3,
        psa_A2=90.0,
        rotatable_bonds=5,
    )
    assert result.ro5_violations >= 1


def test_high_hba_violation():
    result = evaluate_drug_likeness(
        drug_name="HighHBA",
        mw_Da=300.0,
        logP=1.0,
        hbd=1,
        hba=11,
        psa_A2=90.0,
        rotatable_bonds=5,
    )
    assert result.ro5_violations >= 1


def test_multiple_ro5_violations():
    result = evaluate_drug_likeness(
        drug_name="BadDrug",
        mw_Da=700.0,
        logP=7.0,
        hbd=8,
        hba=12,
        psa_A2=200.0,
        rotatable_bonds=15,
    )
    assert result.ro5_violations == 4


# ---------------------------------------------------------------------------
# Veber rules
# ---------------------------------------------------------------------------


def test_aspirin_veber_pass():
    result = aspirin_result()
    assert result.veber_pass is True


def test_veber_fail_high_psa():
    result = evaluate_drug_likeness(
        drug_name="HighPSA",
        mw_Da=300.0,
        logP=1.0,
        hbd=1,
        hba=3,
        psa_A2=150.0,
        rotatable_bonds=5,
    )
    assert result.veber_pass is False


def test_veber_fail_high_rot_bonds():
    result = evaluate_drug_likeness(
        drug_name="FlexDrug",
        mw_Da=300.0,
        logP=1.0,
        hbd=1,
        hba=3,
        psa_A2=90.0,
        rotatable_bonds=12,
    )
    assert result.veber_pass is False


# ---------------------------------------------------------------------------
# Overall score and class
# ---------------------------------------------------------------------------


def test_overall_score_in_range():
    result = aspirin_result()
    assert 0.0 <= result.overall_score <= 100.0


def test_drug_likeness_class_valid():
    for mw in [180, 600]:
        result = evaluate_drug_likeness(
            drug_name="Test",
            mw_Da=float(mw),
            logP=1.0,
            hbd=1,
            hba=3,
            psa_A2=57.0,
            rotatable_bonds=3,
        )
        assert result.drug_likeness_class in ("excellent", "good", "moderate", "poor")


def test_bad_compound_lower_score_than_aspirin():
    good = aspirin_result()
    bad = evaluate_drug_likeness(
        drug_name="BadDrug",
        mw_Da=700.0,
        logP=7.0,
        hbd=8,
        hba=12,
        psa_A2=200.0,
        rotatable_bonds=15,
    )
    assert bad.overall_score < good.overall_score


# ---------------------------------------------------------------------------
# Screen library
# ---------------------------------------------------------------------------


def test_screen_compound_library_sorted():
    compounds = [
        dict(
            drug_name="Bad", mw_Da=700.0, logP=7.0, hbd=8, hba=12, psa_A2=200.0, rotatable_bonds=15
        ),
        dict(drug_name="Good", mw_Da=180.0, logP=1.2, hbd=1, hba=3, psa_A2=57.0, rotatable_bonds=3),
        dict(
            drug_name="Medium", mw_Da=450.0, logP=4.0, hbd=2, hba=5, psa_A2=80.0, rotatable_bonds=7
        ),
    ]
    results = screen_compound_library(compounds)
    assert results[0].overall_score >= results[1].overall_score >= results[2].overall_score


def test_screen_compound_library_length():
    compounds = [
        dict(
            drug_name=f"Drug{i}",
            mw_Da=200.0 + i * 50,
            logP=float(i),
            hbd=1,
            hba=3,
            psa_A2=60.0,
            rotatable_bonds=3,
        )
        for i in range(5)
    ]
    results = screen_compound_library(compounds)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError):
        evaluate_drug_likeness(
            "X", mw_Da=0.0, logP=1.0, hbd=1, hba=3, psa_A2=57.0, rotatable_bonds=3
        )


def test_validation_mw_negative():
    with pytest.raises(ValueError):
        evaluate_drug_likeness(
            "X", mw_Da=-100.0, logP=1.0, hbd=1, hba=3, psa_A2=57.0, rotatable_bonds=3
        )


def test_validation_psa_negative():
    with pytest.raises(ValueError):
        evaluate_drug_likeness(
            "X", mw_Da=300.0, logP=1.0, hbd=1, hba=3, psa_A2=-10.0, rotatable_bonds=3
        )
