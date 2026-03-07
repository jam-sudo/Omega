"""Tests for Phase 897 — Salivary Drug Excretion."""

import math
import pytest

from omega_pbpk.clinical.salivary_excretion import (
    SalivaryExcretionResult,
    predict_salivary_excretion,
    screen_salivary_tdm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLASMA_PH = 7.4


def expected_sp_ratio(pka, ionization_type, saliva_ph=6.8):
    if ionization_type == "base":
        return (1 + 10 ** (pka - saliva_ph)) / (1 + 10 ** (pka - PLASMA_PH))
    elif ionization_type == "acid":
        return (1 + 10 ** (saliva_ph - pka)) / (1 + 10 ** (PLASMA_PH - pka))
    return 1.0


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    r = predict_salivary_excretion("Drug", 8.0, 2.0, "base")
    assert isinstance(r, SalivaryExcretionResult)


# ---------------------------------------------------------------------------
# 2. Frozen / immutability
# ---------------------------------------------------------------------------


def test_frozen():
    r = predict_salivary_excretion("Drug", 8.0, 2.0, "base")
    with pytest.raises((AttributeError, TypeError)):
        r.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Neutral drug S/P = 1
# ---------------------------------------------------------------------------


def test_neutral_sp_ratio_one():
    r = predict_salivary_excretion("NeutralDrug", 7.0, 1.0, "neutral")
    assert r.saliva_plasma_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Basic drug with pKa > 7 accumulates in saliva (S/P > 1)
# ---------------------------------------------------------------------------


def test_basic_high_pka_accumulates():
    r = predict_salivary_excretion("BasicDrug", 9.5, 1.5, "base")
    assert r.saliva_plasma_ratio > 1.0


# ---------------------------------------------------------------------------
# 5. Acidic drug: saliva conc < plasma (S/P < 1)
# ---------------------------------------------------------------------------


def test_acidic_drug_lower_ratio():
    r = predict_salivary_excretion("AcidDrug", 4.5, 1.5, "acid")
    assert r.saliva_plasma_ratio < 1.0


# ---------------------------------------------------------------------------
# 6. S/P ratio matches Henderson-Hasselbalch formula (base)
# ---------------------------------------------------------------------------


def test_sp_ratio_matches_formula_base():
    pka = 8.5
    r = predict_salivary_excretion("Drug", pka, 1.0, "base")
    expected = max(0.01, min(20.0, expected_sp_ratio(pka, "base")))
    assert r.saliva_plasma_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. S/P ratio matches Henderson-Hasselbalch formula (acid)
# ---------------------------------------------------------------------------


def test_sp_ratio_matches_formula_acid():
    pka = 5.0
    r = predict_salivary_excretion("Drug", pka, 1.0, "acid")
    expected = max(0.01, min(20.0, expected_sp_ratio(pka, "acid")))
    assert r.saliva_plasma_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 8. Clamp at 20
# ---------------------------------------------------------------------------


def test_sp_ratio_clamped_upper():
    # pKa 14, base — numerator huge
    r = predict_salivary_excretion("ExtremBase", 14.0, 1.0, "base")
    assert r.saliva_plasma_ratio <= 20.0


# ---------------------------------------------------------------------------
# 9. Clamp at 0.01
# ---------------------------------------------------------------------------


def test_sp_ratio_clamped_lower():
    # pKa 0, acid — numerator ~1, denominator dominated by 10^(7.4-0)
    r = predict_salivary_excretion("ExtremAcid", 0.0, 1.0, "acid")
    assert r.saliva_plasma_ratio >= 0.01


# ---------------------------------------------------------------------------
# 10. Predicted saliva concentration calculation
# ---------------------------------------------------------------------------


def test_predicted_saliva_conc():
    pka, logp = 8.0, 2.5
    plasma_conc = 2.0
    fu = 0.3
    r = predict_salivary_excretion("Drug", pka, logp, "base", plasma_conc, fu_plasma=fu)
    expected_conc = r.saliva_plasma_ratio * plasma_conc * fu
    assert r.predicted_saliva_conc_mg_L == pytest.approx(expected_conc, rel=1e-6)


# ---------------------------------------------------------------------------
# 11. fu_plasma_estimate from logP when not provided
# ---------------------------------------------------------------------------


def test_fu_estimated_from_logp():
    logp = 1.0
    expected_fu = max(0.01, min(1.0, 1.0 / (1.0 + 10 ** (logp - 2.5))))
    r = predict_salivary_excretion("Drug", 7.0, logp, "neutral")
    assert r.fu_plasma_estimate == pytest.approx(expected_fu, rel=1e-6)


# ---------------------------------------------------------------------------
# 12. Provided fu_plasma overrides estimate
# ---------------------------------------------------------------------------


def test_provided_fu_plasma_used():
    r = predict_salivary_excretion("Drug", 7.0, 3.0, "neutral", fu_plasma=0.55)
    assert r.fu_plasma_estimate == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 13. TDM suitable when 0.5 <= S/P <= 2.0
# ---------------------------------------------------------------------------


def test_tdm_suitable_neutral():
    r = predict_salivary_excretion("Drug", 7.0, 1.0, "neutral")
    assert r.tdm_suitable is True


# ---------------------------------------------------------------------------
# 14. TDM not suitable for very acidic drug
# ---------------------------------------------------------------------------


def test_tdm_not_suitable_low_ratio():
    # Extreme acid → very low S/P
    r = predict_salivary_excretion("StrongAcid", 2.0, 1.0, "acid")
    assert r.tdm_suitable is False


# ---------------------------------------------------------------------------
# 15. Recommendation text — TDM suitable
# ---------------------------------------------------------------------------


def test_recommendation_tdm_suitable():
    r = predict_salivary_excretion("Drug", 7.0, 1.0, "neutral")
    assert "Saliva suitable for TDM" in r.saliva_sampling_recommendation


# ---------------------------------------------------------------------------
# 16. Recommendation text — high accumulation
# ---------------------------------------------------------------------------


def test_recommendation_high_accumulation():
    # strong base with high pKa => large S/P > 2
    r = predict_salivary_excretion("StrongBase", 10.0, 1.0, "base")
    assert "High salivary accumulation" in r.saliva_sampling_recommendation


# ---------------------------------------------------------------------------
# 17. Notes — basic drug pKa > 7
# ---------------------------------------------------------------------------


def test_notes_basic_high_pka():
    r = predict_salivary_excretion("Drug", 8.5, 1.0, "base")
    assert "ion trapping" in r.notes


# ---------------------------------------------------------------------------
# 18. Notes — acidic drug
# ---------------------------------------------------------------------------


def test_notes_acidic_drug():
    r = predict_salivary_excretion("Drug", 5.0, 1.0, "acid")
    assert "Acidic drug" in r.notes


# ---------------------------------------------------------------------------
# 19. Notes — neutral drug
# ---------------------------------------------------------------------------


def test_notes_neutral_drug():
    r = predict_salivary_excretion("Drug", 7.0, 1.0, "neutral")
    assert "Neutral drug" in r.notes


# ---------------------------------------------------------------------------
# 20. plasma_ph always 7.4
# ---------------------------------------------------------------------------


def test_plasma_ph_constant():
    r = predict_salivary_excretion("Drug", 7.0, 1.0, "neutral")
    assert r.plasma_ph == pytest.approx(7.4)


# ---------------------------------------------------------------------------
# 21. Validation errors
# ---------------------------------------------------------------------------


def test_invalid_pka_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_salivary_excretion("Drug", -1.0, 1.0, "base")


def test_invalid_pka_high_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_salivary_excretion("Drug", 15.0, 1.0, "base")


def test_invalid_plasma_conc_raises():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        predict_salivary_excretion("Drug", 7.0, 1.0, "base", plasma_conc_mg_L=0.0)


def test_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_salivary_excretion("Drug", 7.0, 1.0, "zwitterion")


# ---------------------------------------------------------------------------
# 22. screen_salivary_tdm — sorted descending by ratio
# ---------------------------------------------------------------------------


def test_screen_sorted_descending():
    candidates = [
        {"name": "Acid", "pka": 4.0, "logp": 1.0, "ionization_type": "acid"},
        {"name": "Base", "pka": 9.0, "logp": 1.0, "ionization_type": "base"},
        {"name": "Neutral", "pka": 7.0, "logp": 1.0, "ionization_type": "neutral"},
    ]
    results = screen_salivary_tdm(candidates)
    ratios = [r.saliva_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_default_ionization_neutral():
    candidates = [{"name": "Drug", "pka": 7.0, "logp": 1.0}]
    results = screen_salivary_tdm(candidates)
    assert results[0].ionization_type == "neutral"


def test_screen_returns_all():
    candidates = [
        {"name": f"D{i}", "pka": 7.0, "logp": float(i)}
        for i in range(5)
    ]
    results = screen_salivary_tdm(candidates)
    assert len(results) == 5
