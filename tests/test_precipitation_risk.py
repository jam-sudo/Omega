"""Tests for Phase 861 — Drug Precipitation Risk."""

import pytest

from omega_pbpk.prediction.precipitation_risk import (
    PrecipitationRiskResult,
    predict_precipitation_risk,
    screen_formulation_concentrations,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def basic_result(ionization_type="base", pka=8.0, logp=2.0, iv_conc=1.0):
    return predict_precipitation_risk(
        drug_name="TestDrug",
        pka=pka,
        logp=logp,
        ionization_type=ionization_type,
        iv_conc_mg_mL=iv_conc,
    )


# ---------------------------------------------------------------------------
# Type and field tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    r = basic_result()
    assert isinstance(r, PrecipitationRiskResult)


def test_all_fields_present():
    r = basic_result()
    assert r.drug_name == "TestDrug"
    assert r.ionization_type == "base"
    assert r.pka == 8.0
    assert r.logp == 2.0
    assert r.s0_intrinsic_mg_mL > 0.0
    assert r.cs_gastric_mg_mL > 0.0
    assert r.cs_intestinal_mg_mL > 0.0
    assert r.cs_plasma_mg_mL > 0.0
    assert 0.0 <= r.iv_precipitation_risk_score <= 100.0
    assert 0.0 <= r.gi_precipitation_risk_score <= 100.0
    assert r.overall_risk in {"low", "moderate", "high"}
    assert r.supersaturation_ratio_gi > 0.0
    assert isinstance(r.recommended_action, str)
    assert isinstance(r.notes, str)


def test_drug_name_preserved():
    r = predict_precipitation_risk("Aspirin", 3.5, 1.2, "acid")
    assert r.drug_name == "Aspirin"


# ---------------------------------------------------------------------------
# Physicochemical behaviour
# ---------------------------------------------------------------------------


def test_base_drug_cs_gastric_greater_than_intestinal():
    """Base drugs are more soluble at low gastric pH."""
    r = basic_result(ionization_type="base", pka=8.0)
    assert r.cs_gastric_mg_mL > r.cs_intestinal_mg_mL


def test_acid_drug_cs_intestinal_greater_than_gastric():
    """Acid drugs are more soluble at higher intestinal pH."""
    r = predict_precipitation_risk("AcidDrug", pka=4.0, logp=1.5, ionization_type="acid")
    assert r.cs_intestinal_mg_mL > r.cs_gastric_mg_mL


def test_neutral_drug_similar_gastric_intestinal():
    """Neutral drugs: solubility independent of pH (both equal S0)."""
    r = predict_precipitation_risk("NeutralDrug", pka=7.0, logp=1.0, ionization_type="neutral")
    assert abs(r.cs_gastric_mg_mL - r.cs_intestinal_mg_mL) < 1e-9


def test_neutral_gi_risk_zero():
    r = predict_precipitation_risk("NeutralDrug", pka=7.0, logp=1.0, ionization_type="neutral")
    assert r.gi_precipitation_risk_score == 0.0


def test_s0_formula():
    """S0 = 10^(-logP + 0.5)."""
    logp = 2.0
    r = basic_result(logp=logp)
    expected_s0 = 10.0 ** (-logp + 0.5)
    assert abs(r.s0_intrinsic_mg_mL - expected_s0) < 1e-9


# ---------------------------------------------------------------------------
# IV risk scoring
# ---------------------------------------------------------------------------


def test_high_iv_conc_high_risk_score():
    """High IV concentration well above plasma solubility → high risk."""
    r = predict_precipitation_risk(
        "HighConc", pka=7.0, logp=4.0, ionization_type="neutral", iv_conc_mg_mL=100.0
    )
    assert r.iv_precipitation_risk_score > 50.0


def test_low_iv_conc_low_risk_score():
    """Very low IV concentration below plasma solubility → zero risk."""
    # logp=-2 → S0 = 10^(2+0.5)=316 mg/mL → large plasma solubility
    r = predict_precipitation_risk(
        "LowConc", pka=7.0, logp=-2.0, ionization_type="neutral", iv_conc_mg_mL=0.001
    )
    assert r.iv_precipitation_risk_score == 0.0


def test_iv_risk_at_si_exactly_1():
    """When concentration exactly equals plasma solubility, risk = 0."""
    # neutral drug: cs_plasma = S0 = 10^(-logp+0.5)
    logp = 2.0
    s0 = 10.0 ** (-logp + 0.5)
    r = predict_precipitation_risk(
        "Threshold", pka=7.0, logp=logp, ionization_type="neutral", iv_conc_mg_mL=s0
    )
    assert r.iv_precipitation_risk_score == 0.0


# ---------------------------------------------------------------------------
# GI risk scoring for base
# ---------------------------------------------------------------------------


def test_base_high_pka_high_gi_risk():
    """Base with very high pKa: large gastric/intestinal solubility ratio → gi risk > 0."""
    r = predict_precipitation_risk("StrongBase", pka=10.0, logp=2.0, ionization_type="base")
    assert r.gi_precipitation_risk_score > 0.0


def test_base_supersaturation_ratio_gt_1():
    r = predict_precipitation_risk("Base", pka=8.0, logp=2.0, ionization_type="base")
    assert r.supersaturation_ratio_gi > 1.0


def test_acid_supersaturation_ratio_lt_1():
    """For acids, cs_gastric < cs_intestinal → ratio < 1."""
    r = predict_precipitation_risk("Acid", pka=4.0, logp=1.5, ionization_type="acid")
    assert r.supersaturation_ratio_gi < 1.0


# ---------------------------------------------------------------------------
# Overall risk classification
# ---------------------------------------------------------------------------


def test_overall_risk_valid_values():
    for it in ("acid", "base", "neutral"):
        r = predict_precipitation_risk("Drug", pka=7.0, logp=2.0, ionization_type=it)
        assert r.overall_risk in {"low", "moderate", "high"}


def test_high_overall_risk():
    """Force both iv and gi risk high."""
    # logp=5 → low S0 → low cs_plasma; high IV conc
    r = predict_precipitation_risk(
        "HighRisk", pka=10.0, logp=5.0, ionization_type="base", iv_conc_mg_mL=50.0
    )
    assert r.overall_risk in {"moderate", "high"}


def test_low_overall_risk():
    r = predict_precipitation_risk(
        "LowRisk", pka=7.0, logp=-2.0, ionization_type="neutral", iv_conc_mg_mL=0.001
    )
    assert r.overall_risk == "low"


# ---------------------------------------------------------------------------
# screen_formulation_concentrations
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_formulation_concentrations(
        "Drug",
        pka=7.0,
        logp=2.0,
        ionization_type="neutral",
        iv_concentrations_mg_mL=[1.0, 5.0, 0.1],
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_ascending_by_iv_risk():
    results = screen_formulation_concentrations(
        "Drug",
        pka=7.0,
        logp=2.0,
        ionization_type="neutral",
        iv_concentrations_mg_mL=[10.0, 0.01, 5.0, 0.1],
    )
    scores = [r.iv_precipitation_risk_score for r in results]
    assert scores == sorted(scores)


def test_screen_result_type():
    results = screen_formulation_concentrations(
        "Drug", pka=7.0, logp=2.0, ionization_type="neutral", iv_concentrations_mg_mL=[1.0, 2.0]
    )
    for r in results:
        assert isinstance(r, PrecipitationRiskResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_pka_below_range():
    with pytest.raises(ValueError, match="pka must be in"):
        predict_precipitation_risk("D", pka=-1.0, logp=2.0, ionization_type="base")


def test_pka_above_range():
    with pytest.raises(ValueError, match="pka must be in"):
        predict_precipitation_risk("D", pka=15.0, logp=2.0, ionization_type="base")


def test_iv_conc_zero():
    with pytest.raises(ValueError, match="iv_conc_mg_mL must be > 0"):
        predict_precipitation_risk(
            "D", pka=7.0, logp=2.0, ionization_type="base", iv_conc_mg_mL=0.0
        )


def test_iv_conc_negative():
    with pytest.raises(ValueError, match="iv_conc_mg_mL must be > 0"):
        predict_precipitation_risk(
            "D", pka=7.0, logp=2.0, ionization_type="base", iv_conc_mg_mL=-1.0
        )


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type must be"):
        predict_precipitation_risk("D", pka=7.0, logp=2.0, ionization_type="zwitterion")
