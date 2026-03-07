"""Tests for Phase 1036 — Skin Sensitization Prediction."""

import pytest

from omega_pbpk.prediction.skin_sensitization import (
    SkinSensitizationResult,
    predict_skin_sensitization,
)

_VALID_CLASSES = {"non-sensitizer", "weak", "moderate", "strong", "extreme"}
_VALID_RISKS = {"low", "moderate", "high"}


# ---------- Helper -----------------------------------------------------------


def make_result(
    drug_name="TestDrug",
    mw_Da=300.0,
    logp=2.0,
    has_electrophile=False,
    has_snh=False,
    has_quinone=False,
    has_epoxide=False,
    has_isocyanate=False,
    topical_use=False,
    concentration_pct=1.0,
):
    return predict_skin_sensitization(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        has_electrophile=has_electrophile,
        has_snh=has_snh,
        has_quinone=has_quinone,
        has_epoxide=has_epoxide,
        has_isocyanate=has_isocyanate,
        topical_use=topical_use,
        concentration_pct=concentration_pct,
    )


# ---------- Return-type tests ------------------------------------------------


def test_returns_frozen_dataclass():
    r = make_result()
    assert isinstance(r, SkinSensitizationResult)


def test_frozen_immutable():
    r = make_result()
    with pytest.raises((AttributeError, TypeError)):
        r.drug_name = "other"  # type: ignore[misc]


# ---------- Score range tests ------------------------------------------------


def test_protein_reactivity_score_in_range():
    r = make_result(has_electrophile=True, has_epoxide=True, has_isocyanate=True)
    assert 0.0 <= r.protein_reactivity_score <= 100.0


def test_skin_penetration_score_in_range():
    for logp in [-3.0, 0.0, 2.5, 5.0, 8.0]:
        r = make_result(logp=logp)
        assert 0.0 <= r.skin_penetration_score <= 100.0, f"Out of range at logP={logp}"


def test_metabolism_activation_score_in_range():
    r = make_result(has_quinone=True, has_snh=True, topical_use=True)
    assert 0.0 <= r.metabolism_activation_score <= 100.0


def test_overall_sensitization_score_in_range():
    r = make_result(has_isocyanate=True, has_electrophile=True, has_epoxide=True)
    assert 0.0 <= r.overall_sensitization_score <= 100.0


# ---------- Class / EC3 / risk tests -----------------------------------------


def test_sensitization_class_valid():
    for logp in [-2.0, 0.0, 2.0, 4.0]:
        r = make_result(logp=logp)
        assert r.sensitization_class in _VALID_CLASSES


def test_ec3_in_range():
    r = make_result(has_isocyanate=True, has_electrophile=True)
    assert 0.1 <= r.EC3_estimate_pct <= 100.0


def test_therapeutic_risk_valid():
    r = make_result()
    assert r.therapeutic_risk in _VALID_RISKS


# ---------- Biological logic -------------------------------------------------


def test_isocyanate_protein_reactivity_gte_60():
    r = make_result(has_isocyanate=True)
    assert r.protein_reactivity_score >= 60.0


def test_epoxide_protein_reactivity_gte_45():
    r = make_result(has_epoxide=True)
    assert r.protein_reactivity_score >= 45.0


def test_no_alerts_returns_no_alert_message():
    r = make_result()
    assert r.structural_alerts == ["No structural alerts identified"]


def test_epoxide_in_structural_alerts():
    r = make_result(has_epoxide=True)
    assert "Epoxide" in r.structural_alerts


def test_higher_logp_higher_skin_penetration_than_very_low():
    r_good = make_result(logp=2.0)
    r_low = make_result(logp=-2.0)
    assert r_good.skin_penetration_score > r_low.skin_penetration_score


def test_topical_use_increases_metabolism_activation():
    r_topical = make_result(topical_use=True)
    r_no_topical = make_result(topical_use=False)
    assert r_topical.metabolism_activation_score > r_no_topical.metabolism_activation_score


def test_all_false_drug_is_non_sensitizer():
    # Use very small logP with heavy MW to suppress sps enough to get non-sensitizer
    # sps at logP=0: 70*(0+1)/2 = 35, mw_factor = max(0.3, 1-(900-300)*0.001) = 0.4
    # sps = 35*0.4 = 14, overall = 0*0.5 + 14*0.3 + 10*0.2 = 4.2 + 2 = 6.2 → non-sensitizer
    r3 = predict_skin_sensitization("clean", mw_Da=900.0, logp=0.0)
    # sps at logP=0: 70*(0+1)/2 = 35, mw_factor = max(0.3, 1-(900-300)*0.001)=max(0.3,0.4)=0.4
    # sps = 35*0.4 = 14, overall = 0*0.5 + 14*0.3 + 10*0.2 = 4.2 + 2 = 6.2 → non-sensitizer
    assert r3.sensitization_class == "non-sensitizer"


def test_strong_sensitizer_is_extreme():
    r = make_result(has_isocyanate=True, has_electrophile=True, has_epoxide=True)
    # prs = 60+40+45 = 145 → 100 (clamped)
    # overall = 100*0.5 + sps*0.3 + mas*0.2 >= 50 → "strong" or "extreme"
    assert r.sensitization_class in ("strong", "extreme")


def test_notes_nonempty():
    r = make_result()
    assert len(r.notes) > 0


# ---------- Validation tests -------------------------------------------------


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_skin_sensitization("Drug", mw_Da=0.0, logp=2.0)


def test_validation_concentration_zero_raises():
    with pytest.raises(ValueError, match="concentration_pct"):
        predict_skin_sensitization("Drug", mw_Da=300.0, logp=2.0, concentration_pct=0.0)


def test_validation_logp_too_high_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_skin_sensitization("Drug", mw_Da=300.0, logp=15.0)


def test_validation_logp_too_low_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_skin_sensitization("Drug", mw_Da=300.0, logp=-6.0)
