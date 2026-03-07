"""Tests for Phase 737 — Drug Crystallization Risk Predictor."""

from omega_pbpk.prediction.crystallization_risk import (
    CrystallizationRiskResult,
    predict_crystallization_risk,
    screen_crystallization_risk,
)


def test_returns_crystallization_risk_result():
    result = predict_crystallization_risk("TestDrug")
    assert isinstance(result, CrystallizationRiskResult)


def test_compound_name_stored():
    result = predict_crystallization_risk("Aspirin")
    assert result.compound_name == "Aspirin"


def test_fields_echoed_back():
    result = predict_crystallization_risk(
        "Drug A", logp=3.5, mw_da=450.0, tm_celsius=180.0, logs_mg_L=5.0, target_conc_mg_L=2.0
    )
    assert result.logp == 3.5
    assert result.mw_da == 450.0
    assert result.tm_celsius == 180.0
    assert result.logs_mg_L == 5.0


def test_risk_score_between_0_and_100():
    result = predict_crystallization_risk(
        "X", logp=10.0, mw_da=1000.0, tm_celsius=300.0, logs_mg_L=0.001
    )
    assert 0.0 <= result.risk_score <= 100.0


def test_risk_score_nonnegative_for_benign_drug():
    result = predict_crystallization_risk(
        "Benign", logp=0.0, mw_da=150.0, tm_celsius=50.0, logs_mg_L=500.0
    )
    assert result.risk_score >= 0.0


def test_risk_level_values():
    valid = {"low", "moderate", "high"}
    for logp in [0.0, 3.0, 6.0]:
        result = predict_crystallization_risk("Drug", logp=logp)
        assert result.risk_level in valid


def test_very_low_solubility_gives_high_risk():
    result = predict_crystallization_risk("InsolubleX", logp=5.0, tm_celsius=220.0, logs_mg_L=0.05)
    assert result.risk_level == "high"


def test_high_solubility_gives_low_risk():
    result = predict_crystallization_risk(
        "Soluble", logp=0.5, mw_da=200.0, tm_celsius=80.0, logs_mg_L=100.0
    )
    assert result.risk_level == "low"


def test_high_logp_increases_risk_vs_low_logp():
    low = predict_crystallization_risk("LowLogP", logp=0.0, logs_mg_L=5.0)
    high = predict_crystallization_risk("HighLogP", logp=5.0, logs_mg_L=5.0)
    assert high.risk_score > low.risk_score


def test_high_tm_increases_risk_vs_low_tm():
    low = predict_crystallization_risk("LowTm", tm_celsius=100.0, logs_mg_L=5.0)
    high = predict_crystallization_risk("HighTm", tm_celsius=250.0, logs_mg_L=5.0)
    assert high.risk_score > low.risk_score


def test_very_low_solubility_adds_40_points():
    result = predict_crystallization_risk(
        "VeryLow", logp=0.0, mw_da=200.0, tm_celsius=50.0, logs_mg_L=0.05
    )
    assert result.risk_score >= 40.0


def test_supersaturation_ratio_formula():
    result = predict_crystallization_risk("SR", logs_mg_L=5.0, target_conc_mg_L=10.0)
    assert abs(result.supersaturation_ratio - 2.0) < 1e-9


def test_supersaturation_ratio_below_one_when_conc_low():
    result = predict_crystallization_risk("SR", logs_mg_L=100.0, target_conc_mg_L=10.0)
    assert result.supersaturation_ratio < 1.0


def test_crystallization_likely_true_when_supersaturated_and_not_low_risk():
    result = predict_crystallization_risk(
        "CrystDrug", logp=5.0, tm_celsius=220.0, logs_mg_L=0.05, target_conc_mg_L=1.0
    )
    assert result.crystallization_likely is True


def test_crystallization_likely_false_when_low_risk():
    result = predict_crystallization_risk(
        "SafeDrug", logp=0.5, mw_da=200.0, tm_celsius=80.0, logs_mg_L=100.0, target_conc_mg_L=500.0
    )
    assert result.crystallization_likely is False


def test_crystallization_likely_false_when_not_supersaturated():
    result = predict_crystallization_risk(
        "NotSuper", logp=5.0, tm_celsius=220.0, logs_mg_L=10.0, target_conc_mg_L=0.5
    )
    assert result.crystallization_likely is False


def test_excipients_none_for_low_risk():
    result = predict_crystallization_risk(
        "Safe", logp=0.5, mw_da=150.0, tm_celsius=80.0, logs_mg_L=200.0
    )
    assert result.recommended_excipients == "none"


def test_excipients_pvp_for_moderate_risk():
    result = predict_crystallization_risk(
        "Mod", logp=3.0, mw_da=200.0, tm_celsius=100.0, logs_mg_L=0.5
    )
    assert result.recommended_excipients == "PVP K30"


def test_excipients_hpc_sds_for_high_risk():
    result = predict_crystallization_risk("HighRisk", logp=5.0, tm_celsius=220.0, logs_mg_L=0.05)
    assert result.recommended_excipients == "HPC+SDS"


def test_notes_nonempty():
    result = predict_crystallization_risk("AnyDrug")
    assert len(result.notes) > 0


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "logp": 1.0, "logs_mg_L": 100.0},
        {"compound_name": "B", "logp": 5.0, "logs_mg_L": 0.05, "tm_celsius": 250.0},
    ]
    results = screen_crystallization_risk(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_risk_score_descending():
    compounds = [
        {"compound_name": "Low", "logp": 0.5, "logs_mg_L": 200.0, "tm_celsius": 50.0},
        {"compound_name": "High", "logp": 6.0, "logs_mg_L": 0.01, "tm_celsius": 250.0},
        {"compound_name": "Mid", "logp": 3.0, "logs_mg_L": 0.5, "tm_celsius": 160.0},
    ]
    results = screen_crystallization_risk(compounds)
    scores = [r.risk_score for r in results]
    assert scores == sorted(scores, reverse=True)
