"""Tests for Phase 420 — excipient_interaction_predictor_p420.py"""

import pytest

from omega_pbpk.prediction.excipient_interaction_predictor_p420 import (
    ExcipientInteractionResult,
    predict_excipient_interaction,
    screen_excipient_compatibility,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def pred(
    drug_name="TestDrug",
    excipient_name="MCC",
    drug_pka=7.0,
    drug_logp=2.0,
    ionization_type="neutral",
    drug_mw_Da=300.0,
    drug_primary_amine=False,
):
    return predict_excipient_interaction(
        drug_name=drug_name,
        excipient_name=excipient_name,
        drug_pka=drug_pka,
        drug_logp=drug_logp,
        ionization_type=ionization_type,
        drug_mw_Da=drug_mw_Da,
        drug_primary_amine=drug_primary_amine,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = pred()
    assert isinstance(r, ExcipientInteractionResult)


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    r = pred(drug_name="Metformin")
    assert r.drug_name == "Metformin"


def test_excipient_name_preserved():
    r = pred(excipient_name="MCC")
    assert r.excipient_name == "MCC"


# ---------------------------------------------------------------------------
# MCC — no interaction for neutral drug
# ---------------------------------------------------------------------------


def test_mcc_no_interaction_neutral():
    r = pred(excipient_name="MCC", ionization_type="neutral", drug_primary_amine=False)
    assert r.interaction_type == "none"
    assert r.severity == "none"
    assert r.severity_score == pytest.approx(0.0)


def test_mcc_compatible():
    r = pred(excipient_name="MCC", ionization_type="neutral")
    assert r.compatibility == "compatible"


# ---------------------------------------------------------------------------
# Maillard reaction: lactose + primary amine
# ---------------------------------------------------------------------------


def test_lactose_primary_amine_maillard():
    r = pred(
        excipient_name="lactose",
        ionization_type="base",
        drug_primary_amine=True,
    )
    assert r.interaction_type == "complexation"
    assert r.severity == "high"


def test_lactose_primary_amine_score():
    r = pred(
        excipient_name="lactose",
        ionization_type="base",
        drug_primary_amine=True,
    )
    assert r.severity_score == pytest.approx(75.0)


def test_lactose_primary_amine_stability_impact():
    r = pred(
        excipient_name="lactose",
        ionization_type="base",
        drug_primary_amine=True,
    )
    assert r.stability_impact_pct == pytest.approx(20.0)


def test_lactose_primary_amine_incompatible():
    r = pred(
        excipient_name="lactose",
        ionization_type="base",
        drug_primary_amine=True,
    )
    assert r.compatibility == "incompatible"


# ---------------------------------------------------------------------------
# Citric acid + base drug → pH_shift
# ---------------------------------------------------------------------------


def test_citric_acid_base_high_delta_ph():
    # citric_acid ph=3.0, drug_pka=9.0 → delta=6 > 3 → high
    r = pred(excipient_name="citric_acid", ionization_type="base", drug_pka=9.0)
    assert r.interaction_type == "pH_shift"
    assert r.severity == "high"
    assert r.severity_score == pytest.approx(70.0)


def test_citric_acid_base_moderate_delta_ph():
    # citric_acid ph=3.0, drug_pka=5.0 → delta=2 (1-3) → moderate
    r = pred(excipient_name="citric_acid", ionization_type="base", drug_pka=5.0)
    assert r.severity == "moderate"


def test_citric_acid_base_low_delta_ph():
    # citric_acid ph=3.0, drug_pka=3.5 → delta=0.5 < 1 → low
    r = pred(excipient_name="citric_acid", ionization_type="base", drug_pka=3.5)
    assert r.severity == "low"


# ---------------------------------------------------------------------------
# Na_bicarbonate + acid drug → pH_shift
# ---------------------------------------------------------------------------


def test_na_bicarbonate_acid_drug():
    # Na_bicarbonate ph=8.5, acid drug
    r = pred(excipient_name="Na_bicarbonate", ionization_type="acid", drug_pka=4.0)
    assert r.interaction_type == "pH_shift"


def test_na_bicarbonate_acid_drug_high_severity():
    # delta = |4.0 - 8.5| = 4.5 > 3 → high
    r = pred(excipient_name="Na_bicarbonate", ionization_type="acid", drug_pka=4.0)
    assert r.severity == "high"


# ---------------------------------------------------------------------------
# Severity score bounds
# ---------------------------------------------------------------------------


def test_severity_score_in_range():
    for exc in ["MCC", "lactose", "citric_acid", "Na_bicarbonate", "SDS"]:
        r = predict_excipient_interaction(
            drug_name="Drug",
            excipient_name=exc,
            drug_pka=7.0,
            drug_logp=2.5,
            ionization_type="base",
            drug_mw_Da=350.0,
            drug_primary_amine=(exc == "lactose"),
        )
        assert 0.0 <= r.severity_score <= 100.0, f"Score out of range for {exc}"


# ---------------------------------------------------------------------------
# stability_impact_pct
# ---------------------------------------------------------------------------


def test_stability_impact_pct_nonneg():
    r = pred(excipient_name="MCC")
    assert r.stability_impact_pct >= 0.0


def test_mcc_stability_impact_zero():
    r = pred(excipient_name="MCC", ionization_type="neutral")
    assert r.stability_impact_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Compatibility assignment
# ---------------------------------------------------------------------------


def test_compatibility_incompatible_when_score_above_60():
    # lactose + primary amine score=75 → incompatible
    r = pred(excipient_name="lactose", drug_primary_amine=True, ionization_type="base")
    assert r.compatibility == "incompatible"


def test_compatibility_conditionally_when_score_30_60():
    # citric_acid + base drug moderate score=40 → conditionally_compatible
    r = pred(excipient_name="citric_acid", ionization_type="base", drug_pka=5.0)
    assert r.compatibility == "conditionally_compatible"


# ---------------------------------------------------------------------------
# Solubility change factor
# ---------------------------------------------------------------------------


def test_solubility_change_factor_reasonable():
    r = pred(excipient_name="Na_bicarbonate", ionization_type="acid", drug_pka=4.0)
    assert 0.1 <= r.solubility_change_factor <= 10.0


def test_solubility_change_factor_one_for_no_interaction():
    r = pred(excipient_name="MCC", ionization_type="neutral")
    assert r.solubility_change_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# screen_excipient_compatibility
# ---------------------------------------------------------------------------


def test_screen_correct_length():
    excipients = ["MCC", "lactose", "citric_acid", "Na_bicarbonate"]
    results = screen_excipient_compatibility(
        drug_name="TestDrug",
        drug_pka=7.0,
        drug_logp=2.0,
        ionization_type="neutral",
        drug_mw_Da=300.0,
        excipient_names=excipients,
    )
    assert len(results) == len(excipients)


def test_screen_sorted_ascending_by_score():
    excipients = ["MCC", "lactose", "citric_acid", "Na_bicarbonate", "SDS"]
    results = screen_excipient_compatibility(
        drug_name="TestDrug",
        drug_pka=7.0,
        drug_logp=2.0,
        ionization_type="base",
        drug_mw_Da=300.0,
        excipient_names=excipients,
        drug_primary_amine=False,
    )
    scores = [r.severity_score for r in results]
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# Invalid excipient → ValueError
# ---------------------------------------------------------------------------


def test_invalid_excipient_raises():
    with pytest.raises(ValueError, match="Unknown excipient"):
        pred(excipient_name="NonExistentFiller123")
