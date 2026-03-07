"""Tests for Phase 604 — DILI Risk Prediction."""

from __future__ import annotations

from omega_pbpk.risk.drug_induced_liver_injury import (
    DILIResult,
    dili_score_from_features,
    predict_dili_risk,
)

VALID_KWARGS = dict(
    drug_name="TestDrug",
    daily_dose_mg=50.0,
    logP=2.0,
    mw=300.0,
    n_aromatic_rings=1,
    has_reactive_metabolite=False,
    has_mitochondrial_liability=False,
    bsep_ic50_uM=None,
    fu_plasma=0.5,
    f_oral=0.8,
    psa=80.0,
    n_hbd=2,
)


def default_result() -> DILIResult:
    return predict_dili_risk(**VALID_KWARGS)


# --- basic return type ---


def test_returns_result():
    r = default_result()
    assert isinstance(r, DILIResult)


def test_score_in_0_100():
    r = default_result()
    assert 0.0 <= r.dili_risk_score <= 100.0


def test_concern_level_valid():
    r = default_result()
    assert r.dili_concern_level in {"no_concern", "low", "moderate", "high", "most_concerned"}


# --- scoring directional tests ---


def test_reactive_metabolite_increases_score():
    r_no = predict_dili_risk(**{**VALID_KWARGS, "has_reactive_metabolite": False})
    r_rm = predict_dili_risk(**{**VALID_KWARGS, "has_reactive_metabolite": True})
    assert r_rm.dili_risk_score > r_no.dili_risk_score


def test_mitochondrial_increases_score():
    r_no = predict_dili_risk(**{**VALID_KWARGS, "has_mitochondrial_liability": False})
    r_mt = predict_dili_risk(**{**VALID_KWARGS, "has_mitochondrial_liability": True})
    assert r_mt.dili_risk_score > r_no.dili_risk_score


def test_bsep_inhibition_increases_score():
    r_no = predict_dili_risk(**{**VALID_KWARGS, "bsep_ic50_uM": None})
    r_bsep = predict_dili_risk(**{**VALID_KWARGS, "bsep_ic50_uM": 5.0})
    assert r_bsep.dili_risk_score > r_no.dili_risk_score
    assert r_bsep.bile_salt_transporter_risk is True


def test_high_dose_increases_score():
    r_low = predict_dili_risk(**{**VALID_KWARGS, "daily_dose_mg": 10.0})
    r_high = predict_dili_risk(**{**VALID_KWARGS, "daily_dose_mg": 500.0})
    assert r_high.dili_risk_score > r_low.dili_risk_score


# --- structural alerts ---


def test_structural_alerts_list():
    r = default_result()
    assert isinstance(r.structural_alerts, list)


def test_no_alerts_for_safe_drug():
    """Simple drug with no structural alerts."""
    r = predict_dili_risk(
        drug_name="SafeDrug",
        daily_dose_mg=10.0,
        logP=1.0,
        mw=300.0,
        n_aromatic_rings=0,
        psa=80.0,
        n_hbd=2,
    )
    # no aromatic rings → no halogenated, epoxide, quinone; logP=1>0 but no aromatic → no aniline/nitro
    assert "halogenated_aromatic" not in r.structural_alerts
    assert "epoxide_risk" not in r.structural_alerts


def test_halogenated_aromatic_alert():
    r = predict_dili_risk(
        drug_name="HaloArom",
        daily_dose_mg=50.0,
        logP=4.0,
        mw=350.0,
        n_aromatic_rings=2,
        psa=80.0,
        n_hbd=2,
    )
    assert "halogenated_aromatic" in r.structural_alerts


def test_epoxide_risk_alert():
    r = predict_dili_risk(
        drug_name="EpoxRisk",
        daily_dose_mg=50.0,
        logP=2.0,
        mw=300.0,
        n_aromatic_rings=3,
        psa=80.0,
        n_hbd=2,
    )
    assert "epoxide_risk" in r.structural_alerts


# --- derived flags ---


def test_oxidative_stress_from_rm():
    r = predict_dili_risk(**{**VALID_KWARGS, "has_reactive_metabolite": True})
    assert r.oxidative_stress_risk is True


def test_oxidative_stress_from_rings():
    r = predict_dili_risk(
        **{**VALID_KWARGS, "n_aromatic_rings": 2, "has_reactive_metabolite": False}
    )
    assert r.oxidative_stress_risk is True


def test_dose_dependent_flag():
    r = predict_dili_risk(**{**VALID_KWARGS, "daily_dose_mg": 200.0})
    assert r.dose_dependent_risk is True


def test_hepatic_auc_proxy_positive():
    r = default_result()
    assert r.hepatic_auc_proxy > 0


# --- confidence ---


def test_confidence_high_with_rm():
    r = predict_dili_risk(**{**VALID_KWARGS, "has_reactive_metabolite": True})
    assert r.confidence == "high"


def test_confidence_low_without_info():
    r = predict_dili_risk(
        **{
            **VALID_KWARGS,
            "has_reactive_metabolite": False,
            "has_mitochondrial_liability": False,
            "bsep_ic50_uM": None,
        }
    )
    assert r.confidence == "low"


# --- concern levels ---


def test_no_concern_for_clean_drug():
    r = predict_dili_risk(
        drug_name="Clean",
        daily_dose_mg=5.0,
        logP=1.0,
        mw=200.0,
        n_aromatic_rings=0,
        has_reactive_metabolite=False,
        has_mitochondrial_liability=False,
        bsep_ic50_uM=None,
        fu_plasma=0.5,
        f_oral=0.8,
        psa=80.0,
        n_hbd=2,
    )
    assert r.dili_concern_level == "no_concern" or r.dili_risk_score < 20


def test_most_concerned_for_dangerous():
    r = predict_dili_risk(
        drug_name="Dangerous",
        daily_dose_mg=600.0,
        logP=4.0,
        mw=400.0,
        n_aromatic_rings=3,
        has_reactive_metabolite=True,
        has_mitochondrial_liability=True,
        bsep_ic50_uM=1.0,
        fu_plasma=0.5,
        f_oral=0.9,
        psa=50.0,
        n_hbd=0,
    )
    assert r.dili_concern_level == "most_concerned"


# --- dili_score_from_features ---


def test_dili_score_from_features_returns_float():
    features = {
        "reactive_metabolite": False,
        "mitochondrial": False,
        "bsep_ic50": None,
        "n_alerts": 0,
        "daily_dose": 50.0,
        "logP": 2.0,
    }
    result = dili_score_from_features(features)
    assert isinstance(result, float)


def test_dili_score_from_features_in_range():
    features = {
        "reactive_metabolite": True,
        "mitochondrial": True,
        "bsep_ic50": 5.0,
        "n_alerts": 3,
        "daily_dose": 600.0,
        "logP": 4.0,
    }
    result = dili_score_from_features(features)
    assert 0.0 <= result <= 100.0


# --- notes ---


def test_notes_nonempty():
    r = default_result()
    assert len(r.notes) > 0
