"""Tests for Phase 764 — Drug Biowaiver Assessment."""

import pytest

from omega_pbpk.biopharmaceutics.biowaiver_assessment import (
    BiowaiverAssessmentResult,
    assess_biowaiver,
    screen_biowaiver_candidates,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_biowaiver_assessment_result():
    result = assess_biowaiver(
        drug_name="TestDrug",
        dose_mg=10.0,
        solubility_pH4_5_mg_mL=5.0,
        fa_pct=95.0,
    )
    assert isinstance(result, BiowaiverAssessmentResult)


def test_drug_name_preserved():
    result = assess_biowaiver("Ibuprofen", dose_mg=400.0, solubility_pH4_5_mg_mL=0.1)
    assert result.drug_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# Dose number computation
# ---------------------------------------------------------------------------


def test_dose_number_computed_correctly():
    dose_mg = 100.0
    sol_mg_mL = 2.0
    result = assess_biowaiver("DnDrug", dose_mg=dose_mg, solubility_pH4_5_mg_mL=sol_mg_mL)
    expected_do = dose_mg / (sol_mg_mL * 250.0)
    assert abs(result.dose_number - expected_do) < 1e-9


def test_dose_number_positive():
    result = assess_biowaiver("PosDoNum", dose_mg=50.0, solubility_pH4_5_mg_mL=1.0)
    assert result.dose_number > 0.0


# ---------------------------------------------------------------------------
# BCS Class I: full criteria → eligible
# ---------------------------------------------------------------------------


def test_bcs_class_i_all_criteria_eligible():
    result = assess_biowaiver(
        drug_name="BCSClassI",
        dose_mg=10.0,
        solubility_pH4_5_mg_mL=5.0,  # Do = 10/(5*250) = 0.008 < 1 → high sol
        fa_pct=95.0,  # > 85% → high perm
        dissolution_85_in_30min=True,
        is_narrow_ti=False,
        is_prodrug=False,
        has_absorption_window=False,
        has_pgp_inhibitor_excipient=False,
        high_surfactant_load=False,
    )
    assert result.bcs_class == "I"
    assert result.biowaiver_eligible is True


# ---------------------------------------------------------------------------
# Disqualifying factors
# ---------------------------------------------------------------------------


def test_narrow_ti_disqualifies():
    result = assess_biowaiver(
        drug_name="NTIDrug",
        dose_mg=5.0,
        solubility_pH4_5_mg_mL=5.0,
        fa_pct=95.0,
        is_narrow_ti=True,
    )
    assert result.biowaiver_eligible is False


def test_prodrug_disqualifies():
    result = assess_biowaiver(
        drug_name="ProdrugX",
        dose_mg=5.0,
        solubility_pH4_5_mg_mL=5.0,
        fa_pct=95.0,
        is_prodrug=True,
    )
    assert result.biowaiver_eligible is False


def test_bcs_class_iv_not_eligible():
    result = assess_biowaiver(
        drug_name="BCSClassIV",
        dose_mg=500.0,  # Do = 500/(0.1*250) = 20 → low sol
        solubility_pH4_5_mg_mL=0.1,
        fa_pct=40.0,  # low perm
        dissolution_85_in_30min=True,
    )
    assert result.bcs_class == "IV"
    assert result.biowaiver_eligible is False


def test_no_rapid_dissolution_disqualifies():
    result = assess_biowaiver(
        drug_name="SlowDissDrug",
        dose_mg=10.0,
        solubility_pH4_5_mg_mL=5.0,
        fa_pct=95.0,
        dissolution_85_in_30min=False,
    )
    assert result.biowaiver_eligible is False


def test_absorption_window_disqualifies():
    result = assess_biowaiver(
        drug_name="AbsWinDrug",
        dose_mg=10.0,
        solubility_pH4_5_mg_mL=5.0,
        fa_pct=95.0,
        has_absorption_window=True,
    )
    assert result.biowaiver_eligible is False


def test_pgp_inhibitor_excipient_disqualifies():
    result = assess_biowaiver(
        drug_name="PgpDrug",
        dose_mg=10.0,
        solubility_pH4_5_mg_mL=5.0,
        fa_pct=95.0,
        has_pgp_inhibitor_excipient=True,
    )
    assert result.biowaiver_eligible is False


# ---------------------------------------------------------------------------
# BCS class valid values
# ---------------------------------------------------------------------------


def test_bcs_class_valid_values():
    # Class I
    r1 = assess_biowaiver("D1", dose_mg=10.0, solubility_pH4_5_mg_mL=5.0, fa_pct=95.0)
    assert r1.bcs_class == "I"
    # Class II: low sol, high perm
    r2 = assess_biowaiver("D2", dose_mg=500.0, solubility_pH4_5_mg_mL=0.1, fa_pct=92.0)
    assert r2.bcs_class == "II"
    # Class III: high sol, low perm
    r3 = assess_biowaiver("D3", dose_mg=10.0, solubility_pH4_5_mg_mL=5.0, fa_pct=50.0)
    assert r3.bcs_class == "III"
    # Class IV: low sol, low perm
    r4 = assess_biowaiver("D4", dose_mg=500.0, solubility_pH4_5_mg_mL=0.1, fa_pct=40.0)
    assert r4.bcs_class == "IV"


# ---------------------------------------------------------------------------
# BCS Class III: conditional eligibility
# ---------------------------------------------------------------------------


def test_bcs_class_iii_eligible_when_fa_ge_85():
    # Class III: high sol (Do<1) + low Peff → class III; fa>=85 → conditional biowaiver eligible
    result = assess_biowaiver(
        drug_name="ClassIII_OK",
        dose_mg=10.0,
        solubility_pH4_5_mg_mL=5.0,  # Do=10/(5*250)=0.008 < 1 → high sol
        peff_cm_s=1e-5,  # below 2e-4 threshold → low perm → Class III
        fa_pct=88.0,  # >=85 → conditional biowaiver OK
        dissolution_85_in_30min=True,
        is_narrow_ti=False,
        is_prodrug=False,
        has_absorption_window=False,
        has_pgp_inhibitor_excipient=False,
        high_surfactant_load=False,
    )
    assert result.bcs_class == "III"
    assert result.biowaiver_eligible is True


def test_bcs_class_iii_not_eligible_when_fa_lt_85():
    result = assess_biowaiver(
        drug_name="ClassIII_Fail",
        dose_mg=10.0,
        solubility_pH4_5_mg_mL=5.0,  # high sol
        fa_pct=70.0,  # fa < 85
    )
    assert result.bcs_class == "III"
    assert result.biowaiver_eligible is False


# ---------------------------------------------------------------------------
# screen_biowaiver_candidates sorted eligible first
# ---------------------------------------------------------------------------


def test_screen_biowaiver_candidates_returns_list():
    drugs = [
        {"drug_name": "A", "dose_mg": 10.0, "solubility_pH4_5_mg_mL": 5.0, "fa_pct": 95.0},
        {"drug_name": "B", "dose_mg": 500.0, "solubility_pH4_5_mg_mL": 0.1, "fa_pct": 40.0},
    ]
    results = screen_biowaiver_candidates(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_biowaiver_candidates_eligible_first():
    drugs = [
        {
            "drug_name": "Ineligible",
            "dose_mg": 500.0,
            "solubility_pH4_5_mg_mL": 0.1,
            "fa_pct": 40.0,
        },
        {
            "drug_name": "Eligible",
            "dose_mg": 10.0,
            "solubility_pH4_5_mg_mL": 5.0,
            "fa_pct": 95.0,
        },
    ]
    results = screen_biowaiver_candidates(drugs)
    assert results[0].biowaiver_eligible is True
    assert results[-1].biowaiver_eligible is False


# ---------------------------------------------------------------------------
# risk_factors string
# ---------------------------------------------------------------------------


def test_risk_factors_empty_when_no_risks():
    result = assess_biowaiver("CleanDrug", dose_mg=10.0, solubility_pH4_5_mg_mL=5.0, fa_pct=95.0)
    assert result.risk_factors == ""


def test_risk_factors_nonempty_when_risks_present():
    result = assess_biowaiver(
        "RiskyDrug",
        dose_mg=10.0,
        solubility_pH4_5_mg_mL=5.0,
        fa_pct=95.0,
        is_narrow_ti=True,
        is_prodrug=True,
    )
    assert len(result.risk_factors) > 0
    assert "narrow therapeutic index" in result.risk_factors


# ---------------------------------------------------------------------------
# Validation: dose_mg <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_negative_dose_raises_value_error():
    with pytest.raises(ValueError):
        assess_biowaiver("NegDose", dose_mg=-10.0, solubility_pH4_5_mg_mL=1.0)


def test_zero_dose_raises_value_error():
    with pytest.raises(ValueError):
        assess_biowaiver("ZeroDose", dose_mg=0.0, solubility_pH4_5_mg_mL=1.0)


def test_zero_solubility_raises_value_error():
    with pytest.raises(ValueError):
        assess_biowaiver("ZeroSol", dose_mg=100.0, solubility_pH4_5_mg_mL=0.0)


# ---------------------------------------------------------------------------
# notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = assess_biowaiver("NotesDrug", dose_mg=10.0, solubility_pH4_5_mg_mL=5.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
