"""Tests for Phase 1146 — Drug-polymer interaction prediction."""

import pytest

from omega_pbpk.prediction.drug_polymer_interaction import (
    predict_drug_polymer_interaction,
    rank_polymers_for_drug,
    DrugPolymerInteractionResult,
    _POLYMERS,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = predict_drug_polymer_interaction(
        "itraconazole", logP=5.66, ionization_type="base",
        pka=3.7, has_h_bond_donor=False, has_h_bond_acceptor=True,
        polymer_name="HPMC-AS",
    )
    assert isinstance(result, DrugPolymerInteractionResult)


def test_all_polymers_available():
    assert len(_POLYMERS) == 6
    for name in ["HPMC", "PVP", "HPMC-AS", "Eudragit-E", "Eudragit-L", "PEG"]:
        assert name in _POLYMERS


# ---------------------------------------------------------------------------
# Ionic interaction tests
# ---------------------------------------------------------------------------


def test_base_drug_hpmc_as_ionic_score_30():
    """Base drug + HPMC-AS (anionic) → ionic_score=30."""
    result = predict_drug_polymer_interaction(
        "base_drug", logP=3.0, ionization_type="base",
        pka=8.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="HPMC-AS",
    )
    assert result.ionic_interaction_score == pytest.approx(30.0)


def test_base_drug_eudragit_L_ionic_score_30():
    """Base drug + Eudragit-L (anionic) → ionic_score=30."""
    result = predict_drug_polymer_interaction(
        "base_drug", logP=3.0, ionization_type="base",
        pka=8.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="Eudragit-L",
    )
    assert result.ionic_interaction_score == pytest.approx(30.0)


def test_acid_drug_eudragit_E_ionic_score_25():
    """Acid drug + Eudragit-E (cationic) → ionic_score=25."""
    result = predict_drug_polymer_interaction(
        "acid_drug", logP=2.0, ionization_type="acid",
        pka=4.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
        polymer_name="Eudragit-E",
    )
    assert result.ionic_interaction_score == pytest.approx(25.0)


def test_neutral_drug_ionic_polymer_score_5():
    """Neutral drug + ionic polymer → ionic_score=5 (no match)."""
    result = predict_drug_polymer_interaction(
        "neutral_drug", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="HPMC-AS",
    )
    assert result.ionic_interaction_score == pytest.approx(5.0)


def test_neutral_drug_non_ionic_polymer_score_0():
    """Drug + non-ionic polymer → ionic_score=0."""
    result = predict_drug_polymer_interaction(
        "neutral_drug", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="HPMC",
    )
    assert result.ionic_interaction_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# H-bond tests
# ---------------------------------------------------------------------------


def test_hbd_drug_hpmc_higher_than_pvp():
    """H-bond donor drug + HPMC (hbd=3) → higher h-bond score than PVP (hbd=0)."""
    hpmc = predict_drug_polymer_interaction(
        "hbd_drug", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=True, has_h_bond_acceptor=False,
        polymer_name="HPMC",
    )
    pvp = predict_drug_polymer_interaction(
        "hbd_drug", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=True, has_h_bond_acceptor=False,
        polymer_name="PVP",
    )
    # HPMC has hba=10 vs PVP hba=1 → hbd_score: min(40,10*4=40) vs min(40,1*4=4)
    assert hpmc.h_bond_score > pvp.h_bond_score


def test_no_hbd_no_hba_drug_h_bond_score_zero():
    result = predict_drug_polymer_interaction(
        "bare_drug", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="HPMC",
    )
    assert result.h_bond_score == pytest.approx(0.0)


def test_hbd_hba_drug_higher_than_hbd_only():
    """Drug with both HBD and HBA should score >= drug with only HBD."""
    both = predict_drug_polymer_interaction(
        "d", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
        polymer_name="HPMC",
    )
    hbd_only = predict_drug_polymer_interaction(
        "d", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=True, has_h_bond_acceptor=False,
        polymer_name="HPMC",
    )
    assert both.h_bond_score >= hbd_only.h_bond_score


def test_h_bond_score_capped_at_40():
    """H-bond score must be capped at 40."""
    result = predict_drug_polymer_interaction(
        "d", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
        polymer_name="HPMC-AS",  # hba=12, hbd=3
    )
    assert result.h_bond_score <= 40.0


# ---------------------------------------------------------------------------
# logP compatibility score
# ---------------------------------------------------------------------------


def test_logp_score_30_when_within_1():
    """logP matching polymer optimum within 1 → score=30."""
    # HPMC logP_optimum=1.5
    result = predict_drug_polymer_interaction(
        "d", logP=1.8, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="HPMC",
    )
    assert result.logp_compatibility_score == pytest.approx(30.0)


def test_logp_score_20_when_within_2():
    result = predict_drug_polymer_interaction(
        "d", logP=3.4, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="HPMC",  # optimum=1.5, diff=1.9 → <2 → 20
    )
    assert result.logp_compatibility_score == pytest.approx(20.0)


def test_logp_score_5_when_ge_3():
    result = predict_drug_polymer_interaction(
        "d", logP=8.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="HPMC",  # optimum=1.5, diff=6.5 → >=3 → 5
    )
    assert result.logp_compatibility_score == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Interaction class thresholds
# ---------------------------------------------------------------------------


def test_interaction_class_strong():
    # Force a strong score: base + HPMC-AS (30 ionic) + logP match (30) + HBD (40) = 100 → capped at 100
    result = predict_drug_polymer_interaction(
        "d", logP=2.5, ionization_type="base",
        pka=9.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
        polymer_name="HPMC-AS",
    )
    assert result.interaction_score >= 70
    assert result.interaction_class == "strong"


def test_interaction_class_weak():
    # No HBD, no HBA, logP=8 (score=5), neutral drug, non-ionic polymer
    result = predict_drug_polymer_interaction(
        "d", logP=8.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="PVP",
    )
    assert result.interaction_class == "weak"


def test_interaction_score_capped_at_100():
    result = predict_drug_polymer_interaction(
        "d", logP=2.5, ionization_type="base",
        pka=9.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
        polymer_name="HPMC-AS",
    )
    assert result.interaction_score <= 100.0


# ---------------------------------------------------------------------------
# Recommended use
# ---------------------------------------------------------------------------


def test_recommended_use_asd_for_strong():
    result = predict_drug_polymer_interaction(
        "d", logP=2.5, ionization_type="base",
        pka=9.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
        polymer_name="HPMC-AS",
    )
    assert result.recommended_use == "ASD"


def test_recommended_use_not_recommended_for_weak():
    result = predict_drug_polymer_interaction(
        "d", logP=8.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
        polymer_name="PVP",
    )
    assert result.recommended_use == "not_recommended"


def test_recommended_use_values_valid():
    for polymer in _POLYMERS:
        r = predict_drug_polymer_interaction(
            "d", logP=2.0, ionization_type="neutral",
            pka=7.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
            polymer_name=polymer,
        )
        assert r.recommended_use in {"ASD", "matrix", "coating", "not_recommended"}


# ---------------------------------------------------------------------------
# rank_polymers_for_drug
# ---------------------------------------------------------------------------


def test_rank_returns_6_results():
    ranked = rank_polymers_for_drug(
        "d", logP=2.5, ionization_type="base",
        pka=9.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
    )
    assert len(ranked) == 6


def test_rank_sorted_descending():
    ranked = rank_polymers_for_drug(
        "d", logP=2.5, ionization_type="base",
        pka=9.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
    )
    scores = [r.interaction_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_all_polymer_names_present():
    ranked = rank_polymers_for_drug(
        "d", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
    )
    names = {r.polymer_name for r in ranked}
    assert names == set(_POLYMERS.keys())


def test_rank_base_drug_hpmc_as_near_top():
    """For a base drug, HPMC-AS (anionic) should rank highly."""
    ranked = rank_polymers_for_drug(
        "base_d", logP=2.5, ionization_type="base",
        pka=9.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
    )
    top_2_names = {r.polymer_name for r in ranked[:2]}
    assert "HPMC-AS" in top_2_names


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_logP_out_of_range():
    with pytest.raises(ValueError, match="logP"):
        predict_drug_polymer_interaction(
            "d", logP=15.0, ionization_type="neutral",
            pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
            polymer_name="HPMC",
        )


def test_validation_logP_too_low():
    with pytest.raises(ValueError, match="logP"):
        predict_drug_polymer_interaction(
            "d", logP=-10.0, ionization_type="neutral",
            pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
            polymer_name="HPMC",
        )


def test_validation_ionization_type_invalid():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_drug_polymer_interaction(
            "d", logP=2.0, ionization_type="zwitterion",
            pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
            polymer_name="HPMC",
        )


def test_validation_pka_out_of_range():
    with pytest.raises(ValueError, match="pka"):
        predict_drug_polymer_interaction(
            "d", logP=2.0, ionization_type="neutral",
            pka=15.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
            polymer_name="HPMC",
        )


def test_validation_polymer_name_invalid():
    with pytest.raises(ValueError, match="polymer_name"):
        predict_drug_polymer_interaction(
            "d", logP=2.0, ionization_type="neutral",
            pka=7.0, has_h_bond_donor=False, has_h_bond_acceptor=False,
            polymer_name="Kollidon",
        )


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = predict_drug_polymer_interaction(
        "d", logP=2.0, ionization_type="neutral",
        pka=7.0, has_h_bond_donor=True, has_h_bond_acceptor=True,
        polymer_name="HPMC",
    )
    assert len(result.notes) > 0


def test_result_fields_preserved():
    result = predict_drug_polymer_interaction(
        "itraconazole", logP=5.66, ionization_type="base",
        pka=3.7, has_h_bond_donor=False, has_h_bond_acceptor=True,
        polymer_name="HPMC-AS",
    )
    assert result.drug_name == "itraconazole"
    assert result.polymer_name == "HPMC-AS"
    assert result.logP == pytest.approx(5.66)
    assert result.ionization_type == "base"
    assert result.has_h_bond_donor is False
    assert result.has_h_bond_acceptor is True
