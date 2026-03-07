"""Tests for risk/carcinogenicity_predictor.py"""

from omega_pbpk.risk.carcinogenicity_predictor import (
    CarcinogenicityResult,
    estimate_td50,
    predict_carcinogenicity,
)

# --- Common parameters ---
DRUG = "TestCompound"
DOSE = 100.0  # mg/day
LOGP = 2.0
MW = 300.0


def test_returns_result():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW)
    assert isinstance(result, CarcinogenicityResult)


def test_clean_drug_negative():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW)
    assert result.carcinogenicity_risk == "negative"
    assert result.regulatory_flag == "clean"
    assert result.genotoxic_potential is False


def test_nitro_group_high_risk():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, has_nitro_group=True)
    assert result.genotoxic_potential is True
    assert result.risk_score >= 30.0


def test_azo_linkage_risk():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, has_azo_linkage=True)
    assert result.genotoxic_potential is True
    assert "azo_reduction" in result.structural_alerts


def test_alkylating_high_risk():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, has_alkylating_group=True)
    assert result.risk_score >= 30.0
    assert "alkylation" in result.structural_alerts


def test_epoxide_risk():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, has_epoxide=True)
    assert result.genotoxic_potential is True
    assert "epoxide_reactivity" in result.structural_alerts


def test_michael_acceptor_risk():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, has_michael_acceptor=True)
    assert result.genotoxic_potential is True
    assert "michael_addition" in result.structural_alerts


def test_polycyclic_aromatic():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, n_aromatic_rings=3)
    assert result.epigenetic_concern is True
    assert "polycyclic_aromatic" in result.structural_alerts


def test_hormonal_drug():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, is_hormonal=True)
    assert result.hormesis_concern is True
    assert "hormonal_disruption" in result.structural_alerts


def test_risk_score_in_0_100():
    # Trigger many alerts
    result = predict_carcinogenicity(
        DRUG,
        DOSE,
        logP=6.0,
        mw=600.0,
        has_nitro_group=True,
        has_azo_linkage=True,
        has_alkylating_group=True,
        has_epoxide=True,
        has_michael_acceptor=True,
        has_intercalator=True,
        n_aromatic_rings=4,
        is_hormonal=True,
    )
    assert 0.0 <= result.risk_score <= 100.0


def test_risk_class_valid():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW)
    valid_classes = {"negative", "low_concern", "moderate_concern", "high_concern"}
    assert result.carcinogenicity_risk in valid_classes


def test_genotoxic_flag_regulatory():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, has_nitro_group=True)
    assert result.regulatory_flag == "likely_carcinogen"


def test_clean_regulatory_flag():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW)
    assert result.regulatory_flag == "clean"


def test_td50_positive():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW)
    assert result.td50_mg_per_kg_day > 0.0


def test_safety_margin_positive():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW)
    assert result.safety_margin_fold > 0.0


def test_high_dose_lower_safety_margin():
    result_low_dose = predict_carcinogenicity(DRUG, 10.0, LOGP, MW)
    result_high_dose = predict_carcinogenicity(DRUG, 1000.0, LOGP, MW)
    assert result_high_dose.safety_margin_fold < result_low_dose.safety_margin_fold


def test_genotoxic_lower_td50():
    td50_genotoxic = estimate_td50(LOGP, MW, n_alerts=1, genotoxic=True)
    td50_clean = estimate_td50(LOGP, MW, n_alerts=1, genotoxic=False)
    assert td50_genotoxic < td50_clean


def test_multiple_alerts_high_score():
    result = predict_carcinogenicity(
        DRUG,
        DOSE,
        LOGP,
        MW,
        has_nitro_group=True,
        has_epoxide=True,
        has_alkylating_group=True,
    )
    assert result.carcinogenicity_risk == "high_concern"


def test_structural_alerts_tuple():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, has_nitro_group=True)
    assert isinstance(result.structural_alerts, tuple)


def test_confidence_high_for_genotoxic():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW, has_nitro_group=True)
    assert result.confidence == "high"


def test_notes_nonempty():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW)
    assert len(result.notes) > 0


def test_threshold_dose_positive():
    result = predict_carcinogenicity(DRUG, DOSE, LOGP, MW)
    assert result.threshold_dose_mg > 0.0
