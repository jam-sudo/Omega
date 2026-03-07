"""Tests for BCS dose number calculator (Phase 1126)."""

import math

import pytest

from omega_pbpk.prediction.dose_number_calculator import (
    DissolutionRiskResult,
    DoseNumberResult,
    assess_dissolution_risk,
    calculate_dose_number,
    classify_bcs_solubility,
)


# ---------------------------------------------------------------------------
# classify_bcs_solubility
# ---------------------------------------------------------------------------


def test_classify_bcs_high_solubility():
    assert classify_bcs_solubility(0.5) == "high"


def test_classify_bcs_low_solubility():
    assert classify_bcs_solubility(1.0) == "low"


def test_classify_bcs_boundary_exactly_one():
    assert classify_bcs_solubility(1.0) == "low"


def test_classify_bcs_low_high_do():
    assert classify_bcs_solubility(10.0) == "low"


# ---------------------------------------------------------------------------
# calculate_dose_number — result types
# ---------------------------------------------------------------------------


def test_result_type():
    r = calculate_dose_number("Drug", 10.0, 1.0, 4.0, "acid")
    assert isinstance(r, DoseNumberResult)


def test_result_is_frozen():
    r = calculate_dose_number("Drug", 10.0, 1.0, 4.0, "acid")
    with pytest.raises(Exception):
        r.dose_number = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Do < 1 => high BCS solubility
# ---------------------------------------------------------------------------


def test_do_less_than_one_gives_high_class():
    # dose=10mg, solubility=1 mg/mL neutral, Do = 10/(1*250) = 0.04
    r = calculate_dose_number("Drug", 10.0, 1.0, 7.0, "neutral")
    assert r.dose_number < 1.0
    assert r.bcs_solubility_class == "high"
    assert r.dissolution_limited is False
    assert r.absorption_risk == "low"


def test_do_ge_one_gives_low_class():
    # dose=500 mg, solubility=0.5 mg/mL neutral, Do = 500/(0.5*250) = 4.0
    r = calculate_dose_number("Drug", 500.0, 0.5, 7.0, "neutral")
    assert r.dose_number >= 1.0
    assert r.bcs_solubility_class == "low"
    assert r.dissolution_limited is True


def test_do_moderate_risk():
    # dose=200 mg, S=0.5 mg/mL neutral, Do=200/(0.5*250)=1.6
    r = calculate_dose_number("Drug", 200.0, 0.5, 7.0, "neutral")
    assert 1.0 <= r.dose_number < 5.0
    assert r.absorption_risk == "moderate"


def test_do_high_risk():
    # dose=5000 mg, S=0.1 mg/mL neutral, Do=200
    r = calculate_dose_number("Drug", 5000.0, 0.1, 7.0, "neutral")
    assert r.dose_number >= 5.0
    assert r.absorption_risk == "high"


# ---------------------------------------------------------------------------
# Henderson-Hasselbalch pH effects
# ---------------------------------------------------------------------------


def test_acid_drug_higher_ph_gives_higher_solubility_lower_do():
    """Acid drug: at pH > pKa, ionized => higher solubility => lower Do."""
    r_low_ph = calculate_dose_number("AcidDrug", 100.0, 0.1, 4.0, "acid", ph_gi=3.0)
    r_high_ph = calculate_dose_number("AcidDrug", 100.0, 0.1, 4.0, "acid", ph_gi=7.0)
    assert r_high_ph.solubility_at_ph_mg_mL > r_low_ph.solubility_at_ph_mg_mL
    assert r_high_ph.dose_number < r_low_ph.dose_number


def test_base_drug_lower_ph_gives_higher_solubility_lower_do():
    """Base drug: at pH < pKa, protonated => higher solubility => lower Do."""
    r_low_ph = calculate_dose_number("BaseDrug", 100.0, 0.1, 9.0, "base", ph_gi=2.0)
    r_high_ph = calculate_dose_number("BaseDrug", 100.0, 0.1, 9.0, "base", ph_gi=7.0)
    assert r_low_ph.solubility_at_ph_mg_mL > r_high_ph.solubility_at_ph_mg_mL
    assert r_low_ph.dose_number < r_high_ph.dose_number


def test_neutral_drug_solubility_unchanged_by_ph():
    """Neutral drug: S_ph = S0 regardless of pH."""
    r1 = calculate_dose_number("NeutralDrug", 50.0, 0.5, 7.0, "neutral", ph_gi=2.0)
    r2 = calculate_dose_number("NeutralDrug", 50.0, 0.5, 7.0, "neutral", ph_gi=8.0)
    assert abs(r1.solubility_at_ph_mg_mL - r2.solubility_at_ph_mg_mL) < 1e-9


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_do_proportional_to_dose():
    r1 = calculate_dose_number("Drug", 100.0, 0.5, 7.0, "neutral")
    r2 = calculate_dose_number("Drug", 200.0, 0.5, 7.0, "neutral")
    assert abs(r2.dose_number / r1.dose_number - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# assess_dissolution_risk
# ---------------------------------------------------------------------------


def test_dissolution_risk_result_type():
    r = assess_dissolution_risk("Drug", 100.0, 0.5, 4.0, "acid")
    assert isinstance(r, DissolutionRiskResult)


def test_dissolution_risk_worst_case_is_max():
    r = assess_dissolution_risk("Drug", 100.0, 0.5, 4.0, "acid")
    computed_max = max(r.do_gastric, r.do_duodenum, r.do_jejunum, r.do_fasted_si)
    assert abs(r.worst_case_do - computed_max) < 1e-9


def test_dissolution_risk_primary_location_is_argmax():
    r = assess_dissolution_risk("Drug", 100.0, 0.5, 4.0, "acid")
    do_values = {
        "gastric": r.do_gastric,
        "duodenum": r.do_duodenum,
        "jejunum": r.do_jejunum,
        "fasted_si": r.do_fasted_si,
    }
    expected_location = max(do_values, key=lambda k: do_values[k])
    assert r.primary_risk_location == expected_location


def test_dissolution_risk_acid_drug_worst_at_high_ph():
    """Acid drug: least soluble at low pH => gastric typically worst."""
    r = assess_dissolution_risk("AcidDrug", 100.0, 0.01, 5.0, "acid")
    assert r.primary_risk_location == "gastric"


def test_dissolution_risk_base_drug_worst_at_high_ph():
    """Base drug: least soluble at high pH => jejunum likely worst."""
    r = assess_dissolution_risk("BaseDrug", 100.0, 0.01, 5.0, "base")
    # Jejunum pH=6.8 > fasted_si pH=6.5 > duodenum pH=5.0 >> gastric pH=1.5
    # Highest Do at highest pH for base => jejunum
    assert r.primary_risk_location == "jejunum"


def test_dissolution_risk_overall_risk_low():
    # High solubility drug: Do < 1 everywhere
    r = assess_dissolution_risk("HighSol", 10.0, 5.0, 7.0, "neutral")
    assert r.overall_dissolution_risk == "low"


def test_dissolution_risk_overall_risk_high():
    # Very low solubility
    r = assess_dissolution_risk("PoorSol", 1000.0, 0.001, 7.0, "neutral")
    assert r.overall_dissolution_risk == "high"


def test_dissolution_risk_do_all_positive():
    r = assess_dissolution_risk("Drug", 50.0, 0.1, 6.0, "acid")
    assert r.do_gastric > 0
    assert r.do_duodenum > 0
    assert r.do_jejunum > 0
    assert r.do_fasted_si > 0


def test_dissolution_risk_notes_non_empty():
    r = assess_dissolution_risk("Drug", 50.0, 0.1, 6.0, "acid")
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors — calculate_dose_number
# ---------------------------------------------------------------------------


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        calculate_dose_number("Drug", 0.0, 1.0, 4.0, "acid")


def test_invalid_solubility_zero_raises():
    with pytest.raises(ValueError, match="solubility"):
        calculate_dose_number("Drug", 10.0, 0.0, 4.0, "acid")


def test_invalid_pka_below_zero_raises():
    with pytest.raises(ValueError, match="pka"):
        calculate_dose_number("Drug", 10.0, 1.0, -1.0, "acid")


def test_invalid_pka_above_14_raises():
    with pytest.raises(ValueError, match="pka"):
        calculate_dose_number("Drug", 10.0, 1.0, 15.0, "acid")


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        calculate_dose_number("Drug", 10.0, 1.0, 4.0, "zwitterion")


def test_invalid_ph_gi_raises():
    with pytest.raises(ValueError, match="ph_gi"):
        calculate_dose_number("Drug", 10.0, 1.0, 4.0, "acid", ph_gi=15.0)


# ---------------------------------------------------------------------------
# Validation errors — assess_dissolution_risk
# ---------------------------------------------------------------------------


def test_dissolution_risk_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        assess_dissolution_risk("Drug", -10.0, 1.0, 4.0, "acid")


def test_dissolution_risk_invalid_solubility_raises():
    with pytest.raises(ValueError, match="solubility"):
        assess_dissolution_risk("Drug", 10.0, -1.0, 4.0, "acid")


def test_dissolution_risk_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        assess_dissolution_risk("Drug", 10.0, 1.0, 4.0, "amphoteric")


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_dose_number_notes_non_empty():
    r = calculate_dose_number("Drug", 10.0, 1.0, 4.0, "acid")
    assert isinstance(r.notes, str) and len(r.notes) > 0
