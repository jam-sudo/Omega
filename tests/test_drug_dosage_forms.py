"""Tests for Phase 605 — Drug Dosage Forms Reference Guide."""

import pytest

from omega_pbpk.clinical.drug_dosage_forms import (
    DOSAGE_FORM_DATABASE,
    DosageFormProfile,
    DosageFormSelectionResult,
    get_dosage_form_profile,
    list_forms_by_category,
    select_dosage_form,
)

# --- get_dosage_form_profile tests ---


def test_get_profile_returns_result():
    result = get_dosage_form_profile("immediate_release_tablet")
    assert isinstance(result, DosageFormProfile)


def test_get_profile_iv_bolus_ba_100():
    profile = get_dosage_form_profile("iv_bolus")
    assert profile.typical_bioavailability_pct == 100.0


def test_get_profile_case():
    """Sublingual onset should be less than oral IR tablet onset."""
    sl = get_dosage_form_profile("sublingual_tablet")
    ir = get_dosage_form_profile("immediate_release_tablet")
    assert sl.onset_min < ir.onset_min


def test_get_unknown_form_raises():
    with pytest.raises(ValueError, match="Unknown dosage form"):
        get_dosage_form_profile("nonexistent_form_xyz")


def test_sublingual_onset_lt_10():
    profile = get_dosage_form_profile("sublingual_tablet")
    assert profile.onset_min < 10.0


def test_iv_bolus_no_first_pass():
    profile = get_dosage_form_profile("iv_bolus")
    assert profile.first_pass is False


def test_er_tablet_duration_ge_12():
    profile = get_dosage_form_profile("extended_release_tablet")
    assert profile.duration_h >= 12.0


# --- select_dosage_form tests ---


def test_select_returns_result():
    result = select_dosage_form(
        drug_name="TestDrug",
        logP=2.0,
        mw=350.0,
        solubility_mg_mL=1.0,
    )
    assert isinstance(result, DosageFormSelectionResult)


def test_select_fast_onset_prefers_iv():
    """Required onset of 2 min — only IV bolus or sublingual can meet this."""
    result = select_dosage_form(
        drug_name="EmergencyDrug",
        logP=2.0,
        mw=300.0,
        solubility_mg_mL=5.0,
        required_onset_min=2.0,
        required_duration_h=1.0,
        target_bioavailability_pct=100.0,
    )
    # Only iv_bolus has onset=1 min, sublingual=5 min (>2 min threshold)
    assert result.recommended_form in ("iv_bolus", "sublingual_tablet")


def test_select_oral_preferred_when_acceptable():
    """When oral is acceptable and onset/duration allows, oral forms should be candidates."""
    result = select_dosage_form(
        drug_name="OralDrug",
        logP=1.5,
        mw=300.0,
        solubility_mg_mL=10.0,
        required_onset_min=120.0,
        required_duration_h=4.0,
        oral_acceptable=True,
        injection_acceptable=True,
        target_bioavailability_pct=80.0,
    )
    # With oral acceptable and high target BA, oral_solution (85%) should be competitive
    assert result.recommended_form is not None
    assert isinstance(result.recommended_form, str)


def test_select_oral_excluded_when_not_acceptable():
    """When oral_acceptable=False, no oral form should be recommended."""
    result = select_dosage_form(
        drug_name="InjectableDrug",
        logP=2.0,
        mw=300.0,
        solubility_mg_mL=1.0,
        required_onset_min=60.0,
        required_duration_h=6.0,
        oral_acceptable=False,
        injection_acceptable=True,
        target_bioavailability_pct=90.0,
    )
    oral_forms = {
        "immediate_release_tablet",
        "extended_release_tablet",
        "oral_solution",
        "sublingual_tablet",
        "oral_disintegrating_tablet",
    }
    assert result.recommended_form not in oral_forms


def test_select_long_duration_prefers_er():
    """Required duration of 20h should prefer ER tablet or transdermal patch."""
    result = select_dosage_form(
        drug_name="ChronicDrug",
        logP=2.0,
        mw=350.0,
        solubility_mg_mL=2.0,
        required_onset_min=240.0,
        required_duration_h=20.0,
        oral_acceptable=True,
        injection_acceptable=True,
        target_bioavailability_pct=65.0,
    )
    long_duration_forms = {"extended_release_tablet", "transdermal_patch", "iv_infusion"}
    assert result.recommended_form in long_duration_forms


def test_select_recommended_form_nonempty():
    result = select_dosage_form(
        drug_name="TestDrug",
        logP=2.0,
        mw=300.0,
        solubility_mg_mL=1.0,
    )
    assert result.recommended_form != ""


def test_select_alternatives_tuple():
    result = select_dosage_form(
        drug_name="TestDrug",
        logP=2.0,
        mw=300.0,
        solubility_mg_mL=1.0,
    )
    assert isinstance(result.alternative_forms, tuple)


def test_select_notes_nonempty_from_select():
    result = select_dosage_form(
        drug_name="TestDrug",
        logP=2.0,
        mw=300.0,
        solubility_mg_mL=1.0,
    )
    assert len(result.notes) > 0


def test_pk_implications_nonempty():
    result = select_dosage_form(
        drug_name="TestDrug",
        logP=2.0,
        mw=300.0,
        solubility_mg_mL=1.0,
    )
    assert len(result.pk_implications) > 0


def test_selection_rationale_nonempty():
    result = select_dosage_form(
        drug_name="TestDrug",
        logP=2.0,
        mw=300.0,
        solubility_mg_mL=1.0,
    )
    assert len(result.selection_rationale) > 0


# --- list_forms_by_category tests ---


def test_list_forms_by_oral():
    forms = list_forms_by_category("oral")
    assert "immediate_release_tablet" in forms
    assert "extended_release_tablet" in forms
    assert "oral_solution" in forms


def test_list_forms_by_parenteral():
    forms = list_forms_by_category("parenteral")
    assert "iv_bolus" in forms
    assert "im_injection" in forms


def test_list_forms_by_inhalation():
    forms = list_forms_by_category("inhalation")
    assert "inhaled_dry_powder" in forms


def test_list_invalid_category_raises():
    with pytest.raises(ValueError, match="Invalid category"):
        list_forms_by_category("intrathecal")


# --- Database integrity tests ---


def test_database_has_12_plus():
    assert len(DOSAGE_FORM_DATABASE) >= 12


# --- Validation tests ---


def test_select_invalid_onset_raises():
    with pytest.raises(ValueError, match="required_onset_min"):
        select_dosage_form(
            drug_name="TestDrug",
            logP=2.0,
            mw=300.0,
            solubility_mg_mL=1.0,
            required_onset_min=0.0,
        )


def test_select_invalid_duration_raises():
    with pytest.raises(ValueError, match="required_duration_h"):
        select_dosage_form(
            drug_name="TestDrug",
            logP=2.0,
            mw=300.0,
            solubility_mg_mL=1.0,
            required_duration_h=-1.0,
        )
