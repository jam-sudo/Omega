"""Tests for pH-dependent solubility profiling (Phase 696)."""

import pytest

from omega_pbpk.prediction.solubility_ph_profile import (
    SolubilityPHProfile,
    generate_solubility_ph_profile,
    predict_gi_solubility_risk,
    predict_solubility_at_ph,
    screen_solubility_profiles,
)

# ---------------------------------------------------------------------------
# predict_solubility_at_ph
# ---------------------------------------------------------------------------


def test_acid_high_ph_solubility_much_greater_than_s0():
    # acid pKa=5, pH=7 → 10^(7-5)=100 → S = S0 * 101
    s = predict_solubility_at_ph(0.01, pka=5.0, compound_type="acid", ph=7.0)
    assert s > 0.01 * 50


def test_acid_low_ph_solubility_close_to_s0():
    # acid pKa=5, pH=3 → 10^(3-5)=0.01 → S ≈ S0 * 1.01
    s = predict_solubility_at_ph(0.01, pka=5.0, compound_type="acid", ph=3.0)
    assert abs(s - 0.01 * 1.01) < 0.001


def test_base_low_ph_solubility_greater_than_s0():
    # base pKa=9, pH=7 → 10^(9-7)=100 → S = S0 * 101
    s = predict_solubility_at_ph(0.01, pka=9.0, compound_type="base", ph=7.0)
    assert s > 0.01 * 50


def test_base_high_ph_close_to_s0():
    # base pKa=9, pH=11 → 10^(9-11)=0.01 → S ≈ S0 * 1.01
    s = predict_solubility_at_ph(0.01, pka=9.0, compound_type="base", ph=11.0)
    assert abs(s - 0.01 * 1.01) < 0.001


def test_neutral_solubility_equals_s0():
    s = predict_solubility_at_ph(0.5, pka=7.0, compound_type="neutral", ph=7.0)
    assert s == pytest.approx(0.5)


def test_amphoteric_solubility_equals_s0():
    s = predict_solubility_at_ph(0.5, pka=7.0, compound_type="amphoteric", ph=5.0)
    assert s == pytest.approx(0.5)


def test_predict_solubility_invalid_s0_raises():
    with pytest.raises(ValueError):
        predict_solubility_at_ph(0.0, pka=5.0, compound_type="acid", ph=7.0)


def test_predict_solubility_invalid_compound_type_raises():
    with pytest.raises(ValueError):
        predict_solubility_at_ph(0.1, pka=5.0, compound_type="zwitterion_bad", ph=7.0)


# ---------------------------------------------------------------------------
# generate_solubility_ph_profile
# ---------------------------------------------------------------------------


def test_generate_profile_returns_dataclass():
    profile = generate_solubility_ph_profile("TestCompound", 0.1, 5.0, "acid")
    assert isinstance(profile, SolubilityPHProfile)


def test_generate_profile_n_points():
    profile = generate_solubility_ph_profile("TestCompound", 0.1, 5.0, "acid", n_points=15)
    assert len(profile.ph_values) == 15
    assert len(profile.solubility_mg_per_mL) == 15


def test_generate_profile_s_jejunum_positive():
    profile = generate_solubility_ph_profile("TestCompound", 0.1, 5.0, "acid")
    assert profile.s_jejunum > 0.0


def test_acid_s_colon_greater_than_s_fasted_stomach():
    # Acid more soluble at higher pH → colon (7.4) > fasted stomach (1.7)
    profile = generate_solubility_ph_profile("AcidDrug", 0.01, 5.0, "acid")
    assert profile.s_colon > profile.s_fasted_stomach


def test_base_s_fasted_stomach_greater_than_s_colon():
    # Base more soluble at lower pH → fasted stomach (1.7) > colon (7.4)
    profile = generate_solubility_ph_profile("BaseDrug", 0.01, 9.0, "base")
    assert profile.s_fasted_stomach > profile.s_colon


def test_generate_profile_min_max_solubility():
    profile = generate_solubility_ph_profile("TestCompound", 0.1, 5.0, "acid")
    assert profile.min_solubility <= profile.max_solubility
    assert profile.min_solubility == pytest.approx(min(profile.solubility_mg_per_mL))
    assert profile.max_solubility == pytest.approx(max(profile.solubility_mg_per_mL))


def test_generate_profile_notes_nonempty():
    profile = generate_solubility_ph_profile("TestCompound", 0.1, 5.0, "acid")
    assert len(profile.notes) > 0


def test_generate_profile_invalid_s0_raises():
    with pytest.raises(ValueError):
        generate_solubility_ph_profile("Bad", -0.1, 5.0, "acid")


def test_generate_profile_invalid_compound_type_raises():
    with pytest.raises(ValueError):
        generate_solubility_ph_profile("Bad", 0.1, 5.0, "zwitterion_bad")


def test_generate_profile_invalid_ph_range_raises():
    with pytest.raises(ValueError):
        generate_solubility_ph_profile("Bad", 0.1, 5.0, "acid", ph_range=(8.0, 1.0))


# ---------------------------------------------------------------------------
# predict_gi_solubility_risk
# ---------------------------------------------------------------------------


def test_gi_risk_returns_dict():
    result = predict_gi_solubility_risk("TestDrug", 10.0, 1.0, 5.0, "acid")
    assert isinstance(result, dict)
    assert "risk_flags" in result
    assert "do_values" in result
    assert "overall_risk" in result


def test_gi_risk_high_dose_low_solubility_high_risk():
    # dose=1000 mg, S0=0.0001 mg/mL → Do >> 1 everywhere → high risk
    result = predict_gi_solubility_risk("PoorDrug", 1000.0, 0.0001, 5.0, "neutral")
    assert result["overall_risk"] == "high"
    assert len(result["risk_flags"]) > 0


def test_gi_risk_low_dose_high_solubility_low_risk():
    # dose=1 mg, S0=100 mg/mL → Do << 1 everywhere → low risk
    result = predict_gi_solubility_risk("SolubleDrug", 1.0, 100.0, 5.0, "neutral")
    assert result["overall_risk"] == "low"
    assert len(result["risk_flags"]) == 0


def test_gi_risk_do_values_are_floats():
    result = predict_gi_solubility_risk("TestDrug", 10.0, 1.0, 5.0, "acid")
    for _seg, do in result["do_values"].items():
        assert isinstance(do, float)
        assert do >= 0.0


# ---------------------------------------------------------------------------
# screen_solubility_profiles
# ---------------------------------------------------------------------------


def test_screen_solubility_profiles_sorted_by_s_jejunum():
    compounds = [
        {"compound_name": "A", "s0_mg_per_mL": 0.001, "pka": 5.0, "compound_type": "acid"},
        {"compound_name": "B", "s0_mg_per_mL": 10.0, "pka": 5.0, "compound_type": "acid"},
        {"compound_name": "C", "s0_mg_per_mL": 0.1, "pka": 5.0, "compound_type": "acid"},
    ]
    profiles = screen_solubility_profiles(compounds)
    s_jej_vals = [p.s_jejunum for p in profiles]
    assert s_jej_vals == sorted(s_jej_vals)


def test_screen_solubility_profiles_returns_list():
    compounds = [
        {"compound_name": "X", "s0_mg_per_mL": 1.0, "pka": 7.0, "compound_type": "neutral"},
    ]
    profiles = screen_solubility_profiles(compounds)
    assert isinstance(profiles, list)
    assert len(profiles) == 1
    assert isinstance(profiles[0], SolubilityPHProfile)
