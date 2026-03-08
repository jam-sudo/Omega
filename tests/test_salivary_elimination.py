"""Tests for salivary drug elimination model — Phase 1100."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.salivary_elimination import (
    SalivaryEliminationResult,
    predict_salivary_concentration,
    simulate_salivary_profile,
)

# ── return type and fields ────────────────────────────────────────────────────


def test_returns_correct_type():
    result = predict_salivary_concentration("Codeine", 1.0, pka=8.2, ionization_type="base")
    assert isinstance(result, SalivaryEliminationResult)


def test_drug_name_preserved():
    result = predict_salivary_concentration("Morphine", 0.5, pka=8.0, ionization_type="base")
    assert result.drug_name == "Morphine"


def test_ionization_type_preserved():
    result = predict_salivary_concentration("X", 1.0, pka=5.0, ionization_type="acid")
    assert result.ionization_type == "acid"


# ── S/P ratio physics ─────────────────────────────────────────────────────────


def test_base_drug_sp_ratio_greater_than_fu():
    """Base drug: ion-trapping in saliva (lower pH) → S/P > fu_plasma."""
    fu = 0.5
    result = predict_salivary_concentration(
        "BaseDrug", 1.0, pka=9.0, ionization_type="base", fu_plasma=fu
    )
    assert result.sp_ratio > fu


def test_acid_drug_sp_ratio_less_than_fu():
    """Acid drug: ion-trapping in plasma (higher pH) → S/P < fu_plasma."""
    fu = 0.5
    result = predict_salivary_concentration(
        "AcidDrug", 1.0, pka=4.0, ionization_type="acid", fu_plasma=fu
    )
    assert result.sp_ratio < fu


def test_neutral_drug_sp_ratio_equals_fu():
    """Neutral drug: only unbound fraction → S/P = fu_plasma."""
    fu = 0.3
    result = predict_salivary_concentration(
        "NeutralDrug", 2.0, pka=10.0, ionization_type="neutral", fu_plasma=fu
    )
    assert abs(result.sp_ratio - fu) < 1e-9


def test_sp_ratio_positive():
    result = predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type="base")
    assert result.sp_ratio > 0.0


def test_c_saliva_equals_c_plasma_times_sp_ratio():
    c_plasma = 2.5
    result = predict_salivary_concentration(
        "X", c_plasma, pka=7.5, ionization_type="base", fu_plasma=0.4
    )
    assert abs(result.c_saliva_mg_L - c_plasma * result.sp_ratio) < 1e-9


def test_sp_ratio_measured_matches_theoretical():
    """sp_ratio_measured should equal sp_ratio when c_plasma > 0."""
    result = predict_salivary_concentration("X", 1.0, pka=8.0, ionization_type="base")
    assert abs(result.sp_ratio_measured - result.sp_ratio) < 1e-9


# ── monitoring_utility ────────────────────────────────────────────────────────


def test_monitoring_utility_valid():
    for ion_type in ["acid", "base", "neutral"]:
        result = predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type=ion_type)
        assert result.monitoring_utility in {"good", "moderate", "poor"}


def test_monitoring_utility_good_for_high_sp():
    """High S/P (>= 0.8) → good monitoring utility."""
    # Base drug with very high pKa and low pH saliva → very high S/P
    result = predict_salivary_concentration(
        "X", 1.0, pka=11.0, ionization_type="base", fu_plasma=0.9
    )
    assert result.monitoring_utility == "good"


def test_monitoring_utility_poor_for_low_sp():
    """Very low S/P → poor monitoring utility."""
    # Acid drug with low fu_plasma → very low S/P
    result = predict_salivary_concentration(
        "X", 1.0, pka=4.0, ionization_type="acid", fu_plasma=0.05
    )
    assert result.monitoring_utility == "poor"


# ── base drug ion-trapping comparison ────────────────────────────────────────


def test_base_drug_lower_saliva_ph_higher_sp():
    """For basic drug: lower pH_saliva → more ionization in saliva → higher S/P."""
    result_low_ph = predict_salivary_concentration(
        "X", 1.0, pka=8.0, ionization_type="base", fu_plasma=0.5, ph_saliva=6.0
    )
    result_high_ph = predict_salivary_concentration(
        "X", 1.0, pka=8.0, ionization_type="base", fu_plasma=0.5, ph_saliva=7.0
    )
    assert result_low_ph.sp_ratio > result_high_ph.sp_ratio


# ── simulate_salivary_profile ─────────────────────────────────────────────────


def test_simulate_returns_dict():
    result = simulate_salivary_profile("Codeine", 100.0, pka=8.2, ionization_type="base")
    assert isinstance(result, dict)


def test_simulate_dict_keys():
    result = simulate_salivary_profile("Codeine", 100.0, pka=8.2, ionization_type="base")
    assert set(result.keys()) == {"times_h", "c_plasma", "c_saliva", "sp_ratio"}


def test_simulate_list_lengths_match():
    result = simulate_salivary_profile(
        "X", 100.0, pka=7.0, ionization_type="base", t_end_h=6.0, dt_h=0.1
    )
    assert len(result["c_plasma"]) == len(result["c_saliva"])
    assert len(result["times_h"]) == len(result["c_plasma"])


def test_simulate_plasma_starts_at_zero():
    result = simulate_salivary_profile("X", 100.0, pka=7.0, ionization_type="base")
    assert result["c_plasma"][0] == 0.0


def test_simulate_saliva_starts_at_zero():
    result = simulate_salivary_profile("X", 100.0, pka=7.0, ionization_type="base")
    assert result["c_saliva"][0] == 0.0


def test_simulate_plasma_peaks_positive():
    result = simulate_salivary_profile("X", 100.0, pka=7.0, ionization_type="base")
    assert max(result["c_plasma"]) > 0.0


def test_simulate_sp_ratio_matches_static():
    fu = 0.5
    pka = 8.0
    result_sim = simulate_salivary_profile(
        "X", 100.0, pka=pka, ionization_type="base", fu_plasma=fu
    )
    result_static = predict_salivary_concentration(
        "X", 1.0, pka=pka, ionization_type="base", fu_plasma=fu
    )
    assert abs(result_sim["sp_ratio"] - result_static.sp_ratio) < 1e-9


def test_simulate_neutral_drug():
    fu = 0.4
    result = simulate_salivary_profile("X", 50.0, pka=7.0, ionization_type="neutral", fu_plasma=fu)
    assert abs(result["sp_ratio"] - fu) < 1e-9


# ── validation errors ─────────────────────────────────────────────────────────


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type="zwitterion")


def test_invalid_fu_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type="base", fu_plasma=0.0)


def test_invalid_fu_negative():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type="base", fu_plasma=-0.1)


def test_invalid_fu_greater_than_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type="base", fu_plasma=1.5)


def test_invalid_c_plasma_negative():
    with pytest.raises(ValueError, match="c_plasma_mg_L"):
        predict_salivary_concentration("X", -1.0, pka=7.0, ionization_type="base")


def test_invalid_ph_saliva_too_low():
    with pytest.raises(ValueError, match="ph_saliva"):
        predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type="base", ph_saliva=4.0)


def test_invalid_ph_saliva_too_high():
    with pytest.raises(ValueError, match="ph_saliva"):
        predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type="base", ph_saliva=9.0)


def test_invalid_ph_plasma_too_low():
    with pytest.raises(ValueError, match="ph_plasma"):
        predict_salivary_concentration("X", 1.0, pka=7.0, ionization_type="base", ph_plasma=6.5)


def test_simulate_invalid_ionization():
    with pytest.raises(ValueError, match="ionization_type"):
        simulate_salivary_profile("X", 100.0, pka=7.0, ionization_type="zwitterion")


def test_simulate_invalid_fu():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_salivary_profile("X", 100.0, pka=7.0, ionization_type="base", fu_plasma=0.0)
