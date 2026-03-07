"""Tests for prediction/drug_metabolism_prediction.py — Phase 830."""

import pytest

from omega_pbpk.prediction.drug_metabolism_prediction import (
    MetabolismPredictionResult,
    predict_metabolism,
    predict_metabolite_library,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_VALID_PATHWAYS = {"CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "UGT", "non_CYP"}
_VALID_RISK = {"low", "moderate", "high"}


def _make_cyp2d6_drug():
    """High pKa basic amine drug → CYP2D6 expected."""
    return predict_metabolism(
        drug_name="cyp2d6_drug",
        logp=2.0,
        mw_da=300.0,
        pka=9.5,  # > 8
        num_aromatic_rings=1,
        num_rotatable_bonds=3,
        has_hydroxyl=False,
        has_amine=True,  # amine + high pKa → CYP2D6 score +3
        has_ester=False,
        has_amide=False,
    )


def _make_cyp3a4_drug():
    """Large lipophilic drug with 2 aromatic rings → CYP3A4 expected."""
    return predict_metabolism(
        drug_name="cyp3a4_drug",
        logp=4.0,
        mw_da=500.0,  # > 400
        pka=7.0,
        num_aromatic_rings=2,
        num_rotatable_bonds=4,
        has_hydroxyl=False,
        has_amine=False,
        has_ester=False,
        has_amide=False,
    )


def _make_ugt_drug():
    """Polar hydroxyl + amide compound with low logP → UGT expected.

    UGT score: +3 (hydroxyl + logP<2) + 2 (amide) = 5
    CYP2C9: pka=6.5 >= 5, so no +3 bonus; mw<400 → +1 only = 1
    All other pathways score 0-1 → UGT wins clearly.
    """
    return predict_metabolism(
        drug_name="ugt_drug",
        logp=0.5,  # < 2
        mw_da=250.0,
        pka=6.5,  # NOT < 5, so CYP2C9 does not get +3
        num_aromatic_rings=0,
        num_rotatable_bonds=2,
        has_hydroxyl=True,  # hydroxyl + low logP → UGT +3
        has_amine=False,
        has_ester=False,
        has_amide=True,  # amide → UGT +2 (total UGT=5)
    )


def _make_ester_drug():
    """Ester-containing drug → non_CYP pathway scored."""
    return predict_metabolism(
        drug_name="ester_drug",
        logp=1.0,
        mw_da=200.0,
        pka=7.0,
        num_aromatic_rings=0,
        num_rotatable_bonds=2,
        has_hydroxyl=False,
        has_amine=False,
        has_ester=True,  # ester → non_CYP +2
        has_amide=False,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _make_cyp3a4_drug()
    assert isinstance(result, MetabolismPredictionResult)


def test_result_is_frozen():
    result = _make_cyp3a4_drug()
    with pytest.raises(Exception):
        result.primary_pathway = "CYP2D6"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Primary pathway validity
# ---------------------------------------------------------------------------


def test_primary_pathway_is_valid():
    for drug_fn in [_make_cyp2d6_drug, _make_cyp3a4_drug, _make_ugt_drug, _make_ester_drug]:
        result = drug_fn()
        assert result.primary_pathway in _VALID_PATHWAYS


def test_cyp2d6_primary_for_high_pka_amine():
    result = _make_cyp2d6_drug()
    # CYP2D6 score=4 (amine+pKa>8: +3, mw<400: +1)
    # CYP3A4 score=1 (logP>1 but <=3: 0, no 2 aro rings: 0, mw<=400: 0) → actually 0
    # CYP2D6 should be primary
    assert result.primary_pathway == "CYP2D6"


def test_cyp3a4_primary_for_large_lipophilic_drug():
    result = _make_cyp3a4_drug()
    assert result.primary_pathway == "CYP3A4"


def test_ugt_primary_for_polar_hydroxyl():
    result = _make_ugt_drug()
    assert result.primary_pathway == "UGT"


def test_ester_contributes_non_cyp_score():
    result = _make_ester_drug()
    # non_CYP gets +2 for ester; should appear as primary or secondary if score >= 2
    # With logP=1.0 and simple molecule, no other pathway should dominate
    # CYP2C9: pka=7 not <5, so 0; CYP2D6: no amine, so 0; UGT: no hydroxyl, so 0
    # non_CYP score=2 should be primary
    assert result.primary_pathway == "non_CYP"


# ---------------------------------------------------------------------------
# Secondary pathway
# ---------------------------------------------------------------------------


def test_secondary_pathway_is_valid_or_empty():
    result = _make_cyp3a4_drug()
    assert result.secondary_pathway in _VALID_PATHWAYS or result.secondary_pathway == ""


def test_secondary_pathway_empty_when_score_below_2():
    # Isolated CYP3A4 dominant drug with no secondary pathway scoring >= 2
    result = predict_metabolism(
        drug_name="simple_3a4",
        logp=4.0,
        mw_da=450.0,
        pka=7.0,
        num_aromatic_rings=2,
        num_rotatable_bonds=3,
        has_hydroxyl=False,
        has_amine=False,
        has_ester=False,
        has_amide=False,
    )
    # CYP3A4 score=4; all others <=1 → secondary should be ""
    assert result.secondary_pathway == ""


# ---------------------------------------------------------------------------
# fm_primary range
# ---------------------------------------------------------------------------


def test_fm_primary_in_range():
    for drug_fn in [_make_cyp2d6_drug, _make_cyp3a4_drug, _make_ugt_drug]:
        result = drug_fn()
        assert 0.4 <= result.fm_primary <= 0.9, f"fm_primary={result.fm_primary} out of range"


def test_fm_secondary_zero_when_no_secondary():
    result = predict_metabolism(
        drug_name="no_secondary",
        logp=4.0,
        mw_da=450.0,
        pka=7.0,
        num_aromatic_rings=2,
        num_rotatable_bonds=3,
        has_hydroxyl=False,
        has_amine=False,
        has_ester=False,
        has_amide=False,
    )
    assert result.secondary_pathway == ""
    assert result.fm_secondary == 0.0


def test_fm_secondary_in_range_when_present():
    result = predict_metabolism(
        drug_name="dual_pathway",
        logp=2.5,
        mw_da=350.0,
        pka=9.0,  # high pKa amine → CYP2D6 +3
        num_aromatic_rings=2,  # → CYP3A4 +1
        num_rotatable_bonds=6,  # → CYP2C19 +1
        has_hydroxyl=False,
        has_amine=True,
        has_ester=False,
        has_amide=False,
    )
    if result.secondary_pathway:
        assert 0.05 <= result.fm_secondary <= 0.5


# ---------------------------------------------------------------------------
# Hepatic first-pass and bioavailability
# ---------------------------------------------------------------------------


def test_hepatic_first_pass_in_range():
    for drug_fn in [_make_cyp2d6_drug, _make_cyp3a4_drug, _make_ugt_drug]:
        result = drug_fn()
        assert 0.0 <= result.hepatic_first_pass_fraction <= 1.0


def test_bioavailability_in_range():
    for drug_fn in [_make_cyp2d6_drug, _make_cyp3a4_drug, _make_ugt_drug]:
        result = drug_fn()
        assert 0.0 <= result.bioavailability_fraction <= 1.0


def test_high_clint_reduces_bioavailability():
    """Higher logP → higher CLint → higher hepatic extraction → lower F."""
    low_logp = predict_metabolism(
        "low_logp",
        logp=0.0,
        mw_da=300.0,
        pka=7.0,
        num_aromatic_rings=0,
        num_rotatable_bonds=2,
        has_hydroxyl=False,
        has_amine=False,
        has_ester=False,
        has_amide=False,
    )
    high_logp = predict_metabolism(
        "high_logp",
        logp=5.0,
        mw_da=300.0,
        pka=7.0,
        num_aromatic_rings=0,
        num_rotatable_bonds=2,
        has_hydroxyl=False,
        has_amine=False,
        has_ester=False,
        has_amide=False,
    )
    assert low_logp.bioavailability_fraction > high_logp.bioavailability_fraction


# ---------------------------------------------------------------------------
# CYP inhibition/induction risk
# ---------------------------------------------------------------------------


def test_cyp_inhibition_risk_valid():
    for drug_fn in [_make_cyp2d6_drug, _make_cyp3a4_drug, _make_ugt_drug]:
        result = drug_fn()
        assert result.cyp_inhibition_risk in _VALID_RISK


def test_cyp_induction_risk_valid():
    for drug_fn in [_make_cyp2d6_drug, _make_cyp3a4_drug, _make_ugt_drug]:
        result = drug_fn()
        assert result.cyp_induction_risk in _VALID_RISK


def test_high_logp_high_pka_gives_high_inhibition_risk():
    result = predict_metabolism(
        "risky",
        logp=5.0,
        mw_da=400.0,
        pka=8.0,
        num_aromatic_rings=1,
        num_rotatable_bonds=3,
        has_hydroxyl=False,
        has_amine=False,
        has_ester=False,
        has_amide=False,
    )
    assert result.cyp_inhibition_risk == "high"


def test_large_lipophilic_gives_high_induction_risk():
    result = predict_metabolism(
        "inducer",
        logp=5.0,
        mw_da=600.0,
        pka=7.0,
        num_aromatic_rings=2,
        num_rotatable_bonds=4,
        has_hydroxyl=False,
        has_amine=False,
        has_ester=False,
        has_amide=False,
    )
    assert result.cyp_induction_risk == "high"


# ---------------------------------------------------------------------------
# DDI victim risk
# ---------------------------------------------------------------------------


def test_ddi_victim_risk_valid():
    for drug_fn in [_make_cyp2d6_drug, _make_cyp3a4_drug, _make_ugt_drug]:
        result = drug_fn()
        assert result.ddi_victim_risk in _VALID_RISK


def test_high_fm_cyp_gives_high_ddi_victim_risk():
    result = _make_cyp2d6_drug()
    # CYP2D6 primary with score 4 → fm = 0.4 + 0.1*4 = 0.8 > 0.7 → high
    assert result.ddi_victim_risk == "high"


# ---------------------------------------------------------------------------
# predict_metabolite_library
# ---------------------------------------------------------------------------


def test_library_returns_list():
    data = [
        {
            "drug_name": "drugA",
            "logp": 3.0,
            "mw_da": 350.0,
            "pka": 9.0,
            "num_aromatic_rings": 1,
            "num_rotatable_bonds": 3,
            "has_hydroxyl": False,
            "has_amine": True,
            "has_ester": False,
            "has_amide": False,
        },
        {
            "drug_name": "drugB",
            "logp": 1.0,
            "mw_da": 200.0,
            "pka": 4.0,
            "num_aromatic_rings": 0,
            "num_rotatable_bonds": 2,
            "has_hydroxyl": True,
            "has_amine": False,
            "has_ester": False,
            "has_amide": False,
        },
    ]
    results = predict_metabolite_library(data)
    assert isinstance(results, list)
    assert len(results) == 2


def test_library_all_correct_type():
    data = [
        {
            "drug_name": f"drug_{i}",
            "logp": float(i),
            "mw_da": 300.0,
            "pka": 7.0,
            "num_aromatic_rings": 1,
            "num_rotatable_bonds": 3,
            "has_hydroxyl": False,
            "has_amine": False,
            "has_ester": False,
            "has_amide": False,
        }
        for i in range(3)
    ]
    results = predict_metabolite_library(data)
    for r in results:
        assert isinstance(r, MetabolismPredictionResult)


def test_library_preserves_drug_names():
    data = [
        {
            "drug_name": "alpha",
            "logp": 2.0,
            "mw_da": 300.0,
            "pka": 7.0,
            "num_aromatic_rings": 1,
            "num_rotatable_bonds": 3,
            "has_hydroxyl": False,
            "has_amine": False,
            "has_ester": False,
            "has_amide": False,
        },
        {
            "drug_name": "beta",
            "logp": 3.0,
            "mw_da": 400.0,
            "pka": 7.0,
            "num_aromatic_rings": 2,
            "num_rotatable_bonds": 4,
            "has_hydroxyl": False,
            "has_amine": False,
            "has_ester": False,
            "has_amide": False,
        },
    ]
    results = predict_metabolite_library(data)
    names = [r.drug_name for r in results]
    assert names == ["alpha", "beta"]


def test_empty_library_returns_empty_list():
    results = predict_metabolite_library([])
    assert results == []
