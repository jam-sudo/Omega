"""Tests for drug_likeness_scorer module."""

from omega_pbpk.prediction.drug_likeness_scorer import (
    DrugLikenessResult,
    RuleSetResult,
    apply_egan,
    apply_ghose,
    apply_lipinski,
    apply_muegge,
    apply_veber,
    score_drug_likeness,
)


def make_clean_drug():
    return score_drug_likeness(
        drug_name="CleanDrug",
        logP=2.0,
        mw=350.0,
        psa=60.0,
        n_hbd=2,
        n_hba=4,
        n_rotatable_bonds=5,
        n_aromatic_rings=1,
    )


def test_returns_result():
    result = make_clean_drug()
    assert isinstance(result, DrugLikenessResult)


def test_clean_drug_drug_like():
    result = make_clean_drug()
    assert result.drug_likeness_class == "drug_like"


def test_lipinski_passes_for_clean_drug():
    result = apply_lipinski(mw=350.0, logP=2.0, n_hbd=2, n_hba=4)
    assert result.passed is True
    assert len(result.violations) == 0


def test_lipinski_fails_for_large_drug():
    result = apply_lipinski(mw=700.0, logP=6.0, n_hbd=6, n_hba=11)
    assert result.passed is False
    assert len(result.violations) >= 2


def test_veber_passes():
    result = apply_veber(n_rotatable=8, psa=100.0)
    assert result.passed is True
    assert len(result.violations) == 0


def test_veber_fails():
    result = apply_veber(n_rotatable=15, psa=160.0)
    assert result.passed is False
    assert len(result.violations) == 2


def test_ghose_passes():
    result = apply_ghose(logP=2.0, mw=300.0)
    assert result.passed is True


def test_ghose_fails():
    # logP=6 violates [-0.4, 5.6], mw=500 violates [160, 480]
    result = apply_ghose(logP=6.0, mw=500.0)
    assert result.passed is False
    assert len(result.violations) >= 1


def test_egan_passes():
    result = apply_egan(logP=3.0, psa=80.0)
    assert result.passed is True


def test_egan_fails():
    result = apply_egan(logP=6.0, psa=140.0)
    assert result.passed is False


def test_muegge_passes():
    result = apply_muegge(mw=350.0, logP=2.0, psa=60.0, n_hbd=2, n_hba=4, n_rotatable=5, n_rings=2)
    assert result.passed is True


def test_muegge_fails_high_mw():
    result = apply_muegge(mw=650.0, logP=2.0, psa=60.0, n_hbd=2, n_hba=4, n_rotatable=5, n_rings=2)
    assert result.passed is False
    assert any("MW" in v for v in result.violations)


def test_overall_score_in_0_100():
    result = make_clean_drug()
    assert 0.0 <= result.overall_score <= 100.0


def test_drug_likeness_class_valid():
    result = make_clean_drug()
    assert result.drug_likeness_class in {"drug_like", "borderline", "non_drug_like"}


def test_lead_like_small_lipophilic():
    result = score_drug_likeness(
        drug_name="LeadDrug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        n_hbd=2,
        n_hba=4,
        n_rotatable_bonds=5,
        n_aromatic_rings=1,
    )
    assert result.lead_like is True


def test_lead_like_false_large():
    result = score_drug_likeness(
        drug_name="LargeDrug",
        logP=4.0,
        mw=400.0,
        psa=80.0,
        n_hbd=3,
        n_hba=7,
        n_rotatable_bonds=5,
        n_aromatic_rings=2,
    )
    assert result.lead_like is False


def test_fragment_like():
    result = score_drug_likeness(
        drug_name="Fragment",
        logP=1.0,
        mw=200.0,
        psa=50.0,
        n_hbd=2,
        n_hba=2,
        n_rotatable_bonds=3,
        n_aromatic_rings=1,
    )
    assert result.fragment_like is True


def test_rule_set_result_score_in_0_1():
    result = apply_lipinski(mw=350.0, logP=2.0, n_hbd=2, n_hba=4)
    assert 0.0 <= result.score <= 1.0


def test_violations_tuple():
    result = apply_lipinski(mw=700.0, logP=6.0, n_hbd=6, n_hba=11)
    assert isinstance(result.violations, tuple)


def test_logD_proxy_logP():
    result = score_drug_likeness(
        drug_name="TestDrug",
        logP=2.5,
        mw=300.0,
        psa=60.0,
        n_hbd=2,
        n_hba=4,
    )
    assert result.logD == 2.5


def test_high_score_for_drug_like():
    result = make_clean_drug()
    assert result.overall_score >= 70.0


def test_low_score_for_non_drug_like():
    result = score_drug_likeness(
        drug_name="BadDrug",
        logP=7.0,
        mw=800.0,
        psa=200.0,
        n_hbd=8,
        n_hba=15,
        n_rotatable_bonds=20,
        n_aromatic_rings=10,
    )
    assert result.overall_score < 40.0
    assert result.drug_likeness_class == "non_drug_like"


def test_notes_nonempty():
    result = make_clean_drug()
    assert len(result.notes) > 0


def test_all_rule_sets_stored():
    result = make_clean_drug()
    assert isinstance(result.lipinski, RuleSetResult)
    assert isinstance(result.veber, RuleSetResult)
    assert isinstance(result.ghose, RuleSetResult)
    assert isinstance(result.egan, RuleSetResult)
    assert isinstance(result.muegge, RuleSetResult)
