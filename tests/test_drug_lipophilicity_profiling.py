"""Tests for Phase 962 — drug_lipophilicity_profiling module."""

import pytest

from omega_pbpk.prediction.drug_lipophilicity_profiling import (
    LipophilicityProfileResult,
    profile_lipophilicity,
    screen_lipophilicity,
)


def make_result(drug_name="TestDrug", logp=2.0, pka=7.0, ionization_type="neutral"):
    return profile_lipophilicity(drug_name, logp, pka, ionization_type)


def test_return_type():
    result = make_result()
    assert isinstance(result, LipophilicityProfileResult)


def test_result_is_frozen():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.logp = 99.0


def test_logd_at_gastric_ph_is_float():
    result = make_result()
    assert isinstance(result.logd_at_gastric_ph, float)


def test_logd_at_intestinal_ph_is_float():
    result = make_result()
    assert isinstance(result.logd_at_intestinal_ph, float)


def test_logd_plasma_ph_is_float():
    result = make_result()
    assert isinstance(result.logd_plasma_ph, float)


def test_membrane_partition_coeff_positive():
    result = make_result()
    assert result.membrane_partition_coeff > 0


def test_skin_permeability_positive():
    result = make_result()
    assert result.skin_permeability_cm_h > 0


def test_fu_predicted_in_range():
    result = make_result()
    assert 0.0 <= result.fu_predicted <= 1.0


def test_optimal_for_oral_is_bool():
    result = make_result()
    assert isinstance(result.optimal_for_oral, bool)


def test_optimal_for_cns_is_bool():
    result = make_result()
    assert isinstance(result.optimal_for_cns, bool)


def test_lipophilicity_class_valid():
    result = make_result()
    assert result.lipophilicity_class in {"hydrophilic", "lipophilic", "moderate"}


def test_efflux_risk_is_bool():
    result = make_result()
    assert isinstance(result.efflux_risk, bool)


def test_notes_non_empty():
    result = make_result()
    assert result.notes and len(result.notes) > 0


def test_drug_name_preserved():
    result = make_result(drug_name="Aspirin")
    assert result.drug_name == "Aspirin"


def test_neutral_logd_equals_logp():
    """For neutral compound, logD == logP at all pH."""
    logp = 2.5
    result = profile_lipophilicity("Neutral", logp=logp, pka=7.0, ionization_type="neutral")
    assert abs(result.logd_at_gastric_ph - logp) < 1e-9
    assert abs(result.logd_at_intestinal_ph - logp) < 1e-9
    assert abs(result.logd_plasma_ph - logp) < 1e-9


def test_acid_high_ph_logd_less_than_logp():
    """Acid at high pH is ionized → logD < logP."""
    # pKa=4 means strongly ionized at pH 7.4
    result = profile_lipophilicity("AcidDrug", logp=3.0, pka=4.0, ionization_type="acid")
    assert result.logd_plasma_ph < result.logp


def test_base_low_ph_logd_less_than_logp():
    """Base at low pH is ionized → logD < logP."""
    # pKa=9 means strongly ionized at pH 1.2
    result = profile_lipophilicity("BaseDrug", logp=3.0, pka=9.0, ionization_type="base")
    assert result.logd_at_gastric_ph < result.logp


def test_high_logp_lipophilic_class():
    result = profile_lipophilicity("Lipophilic", logp=4.0, pka=7.0, ionization_type="neutral")
    assert result.lipophilicity_class == "lipophilic"


def test_low_logp_hydrophilic_class():
    result = profile_lipophilicity("Hydrophilic", logp=-1.0, pka=7.0, ionization_type="neutral")
    assert result.lipophilicity_class == "hydrophilic"


def test_logp_1_to_3_optimal_for_cns():
    result = profile_lipophilicity("CNSDrug", logp=2.0, pka=7.0, ionization_type="neutral")
    assert result.optimal_for_cns is True


def test_logp_above_3_not_optimal_for_cns():
    result = profile_lipophilicity("Drug", logp=4.0, pka=7.0, ionization_type="neutral")
    assert result.optimal_for_cns is False


def test_efflux_risk_high_logp_basic():
    """logP > 4 and pKa > 7 → efflux_risk = True."""
    result = profile_lipophilicity("EfDrug", logp=5.0, pka=8.0, ionization_type="base")
    assert result.efflux_risk is True


def test_no_efflux_risk_low_logp():
    result = profile_lipophilicity("Safe", logp=2.0, pka=8.0, ionization_type="base")
    assert result.efflux_risk is False


def test_screen_sorted_by_logp_ascending():
    compounds = [
        {"drug_name": "High", "logp": 4.0, "pka": 7.0, "ionization_type": "neutral"},
        {"drug_name": "Low", "logp": -1.0, "pka": 7.0, "ionization_type": "neutral"},
        {"drug_name": "Mid", "logp": 2.0, "pka": 7.0, "ionization_type": "neutral"},
    ]
    results = screen_lipophilicity(compounds)
    logps = [r.logp for r in results]
    assert logps == sorted(logps)


def test_screen_returns_correct_length():
    compounds = [
        {"drug_name": f"Drug{i}", "logp": float(i), "pka": 7.0, "ionization_type": "neutral"}
        for i in range(5)
    ]
    results = screen_lipophilicity(compounds)
    assert len(results) == 5


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        profile_lipophilicity("Drug", logp=2.0, pka=7.0, ionization_type="zwitterion")
