"""
Tests for Phase 899 — Tear Fluid Drug Excretion
"""

import math

import pytest

from omega_pbpk.clinical.tear_fluid_excretion import (
    TearFluidExcretionResult,
    predict_tear_fluid_excretion,
    screen_tear_tdm,
)

# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    result = predict_tear_fluid_excretion("TestDrug", pka=7.4, logp=1.0)
    assert isinstance(result, TearFluidExcretionResult)


def test_frozen_dataclass():
    result = predict_tear_fluid_excretion("TestDrug", pka=7.4, logp=1.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_default_tear_volume():
    result = predict_tear_fluid_excretion("Drug", pka=7.0, logp=0.5)
    assert result.tear_volume_uL == 7.0


def test_drug_name_stored():
    result = predict_tear_fluid_excretion("Ibuprofen", pka=4.4, logp=3.97, ionization_type="acid")
    assert result.drug_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# Neutral drug
# ---------------------------------------------------------------------------


def test_neutral_tp_ratio_is_one():
    result = predict_tear_fluid_excretion("Neutral", pka=7.0, logp=0.0, ionization_type="neutral")
    assert math.isclose(result.tear_plasma_ratio, 1.0, rel_tol=1e-6)


def test_neutral_tear_conc_equals_free_plasma():
    # fu = max(0.01, min(1.0, 1/(1 + 10^(0.0 - 2.5)))) at logp=0
    result = predict_tear_fluid_excretion(
        "Neutral", pka=7.0, logp=0.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    fu = max(0.01, min(1.0, 1.0 / (1.0 + 10 ** (0.0 - 2.5))))
    expected = 1.0 * fu * 1000.0  # T/P=1
    assert math.isclose(result.predicted_tear_conc_ug_mL, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Basic T/P ratio: acid vs base
# ---------------------------------------------------------------------------


def test_acid_tp_ratio():
    # acid with pKa=4.4, tear_ph=7.4=plasma_ph → T/P = 1.0
    result = predict_tear_fluid_excretion(
        "AcidDrug", pka=4.4, logp=1.0, ionization_type="acid", tear_ph=7.4
    )
    assert math.isclose(result.tear_plasma_ratio, 1.0, rel_tol=1e-6)


def test_base_tp_ratio_higher_tear_ph():
    # base with pKa=9.0; tear_ph=7.8 > plasma_ph=7.4
    # T/P = (1+10^(9-7.8))/(1+10^(9-7.4)) — denominator > numerator → T/P < 1
    result = predict_tear_fluid_excretion(
        "BaseDrug", pka=9.0, logp=1.0, ionization_type="base", tear_ph=7.8
    )
    assert result.tear_plasma_ratio < 1.0


def test_base_tp_ratio_lower_tear_ph():
    # base with pKa=9.0; tear_ph=7.0 < plasma_ph=7.4
    # T/P = (1+10^(9-7.0))/(1+10^(9-7.4)) — numerator > denominator → T/P > 1
    result = predict_tear_fluid_excretion(
        "BaseDrug", pka=9.0, logp=1.0, ionization_type="base", tear_ph=7.0
    )
    assert result.tear_plasma_ratio > 1.0


# ---------------------------------------------------------------------------
# T/P clamping
# ---------------------------------------------------------------------------


def test_tp_ratio_clamped_min():
    # Extreme acid, high pKa → may clamp at 0.1
    result = predict_tear_fluid_excretion(
        "ExtremeDrug", pka=1.0, logp=0.0, ionization_type="base", tear_ph=5.0
    )
    assert result.tear_plasma_ratio >= 0.1


def test_tp_ratio_clamped_max():
    # Very strong base, high pKa with low tear_ph → clamp at 10.0
    result = predict_tear_fluid_excretion(
        "StrongBase", pka=12.0, logp=0.0, ionization_type="base", tear_ph=9.0
    )
    assert result.tear_plasma_ratio <= 10.0


# ---------------------------------------------------------------------------
# fu_plasma
# ---------------------------------------------------------------------------


def test_fu_plasma_explicit():
    result = predict_tear_fluid_excretion(
        "Drug", pka=7.0, logp=3.0, ionization_type="neutral", plasma_conc_mg_L=1.0, fu_plasma=0.5
    )
    # T/P = 1.0, predicted = 1.0 * 0.5 * 1000 = 500
    assert math.isclose(result.predicted_tear_conc_ug_mL, 500.0, rel_tol=1e-6)


def test_fu_plasma_estimated_from_logp():
    # logp=2.5 → fu = 1/(1 + 10^0) = 0.5
    result = predict_tear_fluid_excretion(
        "Drug", pka=7.0, logp=2.5, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    fu_expected = 1.0 / (1.0 + 10.0 ** (2.5 - 2.5))  # = 0.5
    expected_conc = 1.0 * fu_expected * 1000.0
    assert math.isclose(result.predicted_tear_conc_ug_mL, expected_conc, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Drug amount in tears
# ---------------------------------------------------------------------------


def test_drug_amount_in_tears_ng():
    result = predict_tear_fluid_excretion(
        "Drug", pka=7.0, logp=0.0, ionization_type="neutral", plasma_conc_mg_L=1.0, fu_plasma=1.0
    )
    # T/P=1, fu=1, conc = 1.0*1*1000 = 1000 µg/mL
    # amount = 1000 * 7.0 * 1000 = 7,000,000 ng
    expected = result.predicted_tear_conc_ug_mL * 7.0 * 1000.0
    assert math.isclose(result.drug_amount_in_tears_ng, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Contact lens accumulation
# ---------------------------------------------------------------------------


def test_contact_lens_logp_negative():
    result = predict_tear_fluid_excretion("Drug", pka=7.0, logp=-1.0)
    assert result.contact_lens_accumulation_factor == 1.0


def test_contact_lens_logp_zero():
    result = predict_tear_fluid_excretion("Drug", pka=7.0, logp=0.0)
    assert result.contact_lens_accumulation_factor == 1.0


def test_contact_lens_logp_positive():
    result = predict_tear_fluid_excretion("Drug", pka=7.0, logp=3.0)
    assert math.isclose(result.contact_lens_accumulation_factor, 6.0, rel_tol=1e-9)


def test_contact_lens_high_logp():
    result = predict_tear_fluid_excretion("Drug", pka=7.0, logp=5.0)
    assert math.isclose(result.contact_lens_accumulation_factor, 10.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Therapeutic ocular potential
# ---------------------------------------------------------------------------


def test_therapeutic_potential_true_when_conc_high():
    result = predict_tear_fluid_excretion(
        "Drug", pka=7.0, logp=0.0, plasma_conc_mg_L=1.0, fu_plasma=1.0
    )
    assert result.therapeutic_ocular_potential is True


def test_therapeutic_potential_false_when_conc_low():
    result = predict_tear_fluid_excretion(
        "Drug",
        pka=7.0,
        logp=5.0,
        plasma_conc_mg_L=0.0001,
        fu_plasma=0.01,
        ionization_type="neutral",
    )
    assert result.therapeutic_ocular_potential is False


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_therapeutic_monitoring():
    result = predict_tear_fluid_excretion(
        "Drug", pka=7.0, logp=0.0, plasma_conc_mg_L=1.0, fu_plasma=1.0
    )
    assert "ocular drug monitoring" in result.notes.lower()


def test_notes_tdm_feasible():
    # very low tear concentration, logp < 0 → no lens accumulation
    result = predict_tear_fluid_excretion(
        "Drug",
        pka=7.0,
        logp=-2.0,
        plasma_conc_mg_L=0.00001,
        fu_plasma=0.001,
        ionization_type="neutral",
    )
    assert (
        "TDM" in result.notes or "tdm" in result.notes.lower() or "systemic" in result.notes.lower()
    )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_pka_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_tear_fluid_excretion("Drug", pka=15.0, logp=1.0)


def test_invalid_plasma_conc_raises():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        predict_tear_fluid_excretion("Drug", pka=7.0, logp=1.0, plasma_conc_mg_L=0.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_tear_fluid_excretion("Drug", pka=7.0, logp=1.0, ionization_type="zwitterion")


# ---------------------------------------------------------------------------
# screen_tear_tdm
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    candidates = [
        {"name": "A", "pka": 7.0, "logp": 0.5},
        {"name": "B", "pka": 9.0, "logp": 1.0, "ionization_type": "base"},
    ]
    results = screen_tear_tdm(candidates)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_tp_ratio():
    candidates = [
        {"name": "A", "pka": 9.0, "logp": 1.0, "ionization_type": "base"},
        {"name": "B", "pka": 7.0, "logp": 0.5},
        {"name": "C", "pka": 4.0, "logp": 1.0, "ionization_type": "acid"},
    ]
    results = screen_tear_tdm(candidates)
    ratios = [r.tear_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_default_ionization_type():
    candidates = [{"name": "X", "pka": 7.0, "logp": 1.0}]
    results = screen_tear_tdm(candidates)
    assert results[0].ionization_type == "neutral"
