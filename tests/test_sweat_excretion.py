"""Tests for Phase 898 — Sweat Drug Excretion."""

import math
import pytest

from omega_pbpk.clinical.sweat_excretion import (
    SweatExcretionResult,
    predict_sweat_excretion,
    screen_sweat_excretion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLASMA_PH = 7.4


def expected_ratio(pka, ionization_type, sweat_ph=5.5):
    if ionization_type == "base":
        return (1 + 10 ** (pka - sweat_ph)) / (1 + 10 ** (pka - PLASMA_PH))
    elif ionization_type == "acid":
        return (1 + 10 ** (sweat_ph - pka)) / (1 + 10 ** (PLASMA_PH - pka))
    return 1.0


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    r = predict_sweat_excretion("Drug", 8.0, 2.0, "base")
    assert isinstance(r, SweatExcretionResult)


# ---------------------------------------------------------------------------
# 2. Frozen / immutability
# ---------------------------------------------------------------------------


def test_frozen():
    r = predict_sweat_excretion("Drug", 8.0, 2.0, "base")
    with pytest.raises((AttributeError, TypeError)):
        r.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Neutral drug sweat/plasma = 1
# ---------------------------------------------------------------------------


def test_neutral_ratio_one():
    r = predict_sweat_excretion("NeutralDrug", 7.0, 1.0, "neutral")
    assert r.sweat_plasma_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Basic drug with high pKa accumulates in acidic sweat
# ---------------------------------------------------------------------------


def test_basic_high_pka_accumulates():
    r = predict_sweat_excretion("BasicDrug", 9.5, 1.0, "base")
    assert r.sweat_plasma_ratio > 1.0


# ---------------------------------------------------------------------------
# 5. Acidic drug: sweat ratio < 1
# ---------------------------------------------------------------------------


def test_acidic_lower_ratio():
    r = predict_sweat_excretion("AcidDrug", 4.5, 1.0, "acid")
    assert r.sweat_plasma_ratio < 1.0


# ---------------------------------------------------------------------------
# 6. Sweat/plasma ratio matches HH formula (base)
# ---------------------------------------------------------------------------


def test_ratio_matches_formula_base():
    pka = 8.0
    r = predict_sweat_excretion("Drug", pka, 1.0, "base")
    raw = expected_ratio(pka, "base")
    expected = max(0.01, min(100.0, raw))
    assert r.sweat_plasma_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. Sweat/plasma ratio matches HH formula (acid)
# ---------------------------------------------------------------------------


def test_ratio_matches_formula_acid():
    pka = 5.0
    r = predict_sweat_excretion("Drug", pka, 1.0, "acid")
    raw = expected_ratio(pka, "acid")
    expected = max(0.01, min(100.0, raw))
    assert r.sweat_plasma_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 8. Clamp at 100
# ---------------------------------------------------------------------------


def test_ratio_clamped_upper():
    r = predict_sweat_excretion("ExtremBase", 14.0, 1.0, "base", sweat_ph=4.0)
    assert r.sweat_plasma_ratio <= 100.0


# ---------------------------------------------------------------------------
# 9. Clamp at 0.01
# ---------------------------------------------------------------------------


def test_ratio_clamped_lower():
    r = predict_sweat_excretion("ExtremAcid", 0.0, 1.0, "acid")
    assert r.sweat_plasma_ratio >= 0.01


# ---------------------------------------------------------------------------
# 10. Predicted sweat concentration (µg/mL)
# ---------------------------------------------------------------------------


def test_predicted_sweat_conc():
    pka, logp = 8.0, 2.0
    plasma_conc = 1.0
    fu = 0.4
    r = predict_sweat_excretion("Drug", pka, logp, "base", plasma_conc, fu_plasma=fu)
    expected = r.sweat_plasma_ratio * plasma_conc * fu * 1000.0
    assert r.predicted_sweat_conc_ug_mL == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 11. Drug flux calculation
# ---------------------------------------------------------------------------


def test_drug_flux():
    r = predict_sweat_excretion("Drug", 8.0, 1.0, "base", sweat_rate_mL_h=1.0)
    assert r.drug_flux_ug_per_h == pytest.approx(
        r.predicted_sweat_conc_ug_mL * 1.0, rel=1e-6
    )


# ---------------------------------------------------------------------------
# 12. Higher logP → lower fu → lower predicted sweat conc
# ---------------------------------------------------------------------------


def test_logp_affects_sweat_conc():
    # logP=1 → higher fu than logP=4 → higher sweat concentration
    r_low = predict_sweat_excretion("Drug", 7.0, 1.0, "neutral")
    r_high = predict_sweat_excretion("Drug", 7.0, 4.0, "neutral")
    assert r_low.predicted_sweat_conc_ug_mL > r_high.predicted_sweat_conc_ug_mL


# ---------------------------------------------------------------------------
# 13. Providing fu_plasma raises concentration vs low fu estimate
# ---------------------------------------------------------------------------


def test_provided_fu_plasma_raises_conc():
    # logP=5 → estimated fu very low; providing fu=0.8 should give much higher conc
    r_estimated = predict_sweat_excretion("Drug", 7.0, 5.0, "neutral")
    r_provided = predict_sweat_excretion("Drug", 7.0, 5.0, "neutral", fu_plasma=0.8)
    assert r_provided.predicted_sweat_conc_ug_mL > r_estimated.predicted_sweat_conc_ug_mL


# ---------------------------------------------------------------------------
# 14. Detection window clamped to [0, 720]
# ---------------------------------------------------------------------------


def test_detection_window_clamped():
    r = predict_sweat_excretion("Drug", 14.0, 1.0, "base", plasma_conc_mg_L=100.0)
    assert 0.0 <= r.detection_window_h <= 720.0


# ---------------------------------------------------------------------------
# 15. Detection window formula
# ---------------------------------------------------------------------------


def test_detection_window_formula():
    plasma_conc = 2.0
    lod = 0.01
    r = predict_sweat_excretion(
        "Drug", 7.0, 1.0, "neutral", plasma_conc_mg_L=plasma_conc, lod_ug_mL=lod
    )
    raw = (plasma_conc * 1000.0 * r.sweat_plasma_ratio) / lod / 2.0
    expected = max(0.0, min(720.0, raw))
    assert r.detection_window_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 16. patch_test_suitable — True when flux >= 0.1
# ---------------------------------------------------------------------------


def test_patch_suitable_high_flux():
    # Basic drug, high pKa → large ratio → large flux
    r = predict_sweat_excretion(
        "HighFlux", 9.0, 1.0, "base", plasma_conc_mg_L=5.0, sweat_rate_mL_h=1.0
    )
    if r.drug_flux_ug_per_h >= 0.1:
        assert r.patch_test_suitable is True


# ---------------------------------------------------------------------------
# 17. patch_test_suitable — False when flux < 0.1
# ---------------------------------------------------------------------------


def test_patch_not_suitable_low_flux():
    # Very small plasma conc → tiny flux
    r = predict_sweat_excretion(
        "LowFlux",
        pka=7.0,
        logp=5.0,  # high logP → low fu
        ionization_type="neutral",
        plasma_conc_mg_L=0.0001,
        sweat_rate_mL_h=0.01,
    )
    assert r.patch_test_suitable is False


# ---------------------------------------------------------------------------
# 18. Notes — patch suitable
# ---------------------------------------------------------------------------


def test_notes_patch_suitable():
    r = predict_sweat_excretion(
        "Patch", 9.0, 1.0, "base", plasma_conc_mg_L=5.0, sweat_rate_mL_h=1.0
    )
    if r.patch_test_suitable:
        assert "Sweat patch suitable" in r.notes


# ---------------------------------------------------------------------------
# 19. Notes — high ratio not patch suitable
# ---------------------------------------------------------------------------


def test_notes_high_ratio_not_patch():
    # Create scenario: high ratio but low flux
    # ratio>5, patch_test_suitable=False
    r = predict_sweat_excretion(
        "HighRatio",
        pka=10.0,
        logp=4.5,  # → low fu → might make flux < 0.1
        ionization_type="base",
        plasma_conc_mg_L=0.0001,
        sweat_rate_mL_h=0.01,
    )
    if not r.patch_test_suitable and r.sweat_plasma_ratio > 5.0:
        assert "High sweat/plasma ratio" in r.notes


# ---------------------------------------------------------------------------
# 20. Validation errors
# ---------------------------------------------------------------------------


def test_invalid_pka_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_sweat_excretion("Drug", -0.5, 1.0, "base")


def test_invalid_pka_high_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_sweat_excretion("Drug", 14.1, 1.0, "base")


def test_invalid_plasma_conc_raises():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        predict_sweat_excretion("Drug", 7.0, 1.0, "base", plasma_conc_mg_L=-1.0)


def test_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_sweat_excretion("Drug", 7.0, 1.0, "amphoteric")


def test_invalid_sweat_rate_raises():
    with pytest.raises(ValueError, match="sweat_rate_mL_h"):
        predict_sweat_excretion("Drug", 7.0, 1.0, "base", sweat_rate_mL_h=0.0)


# ---------------------------------------------------------------------------
# 21. screen_sweat_excretion — sorted descending
# ---------------------------------------------------------------------------


def test_screen_sorted_descending():
    candidates = [
        {"name": "Acid", "pka": 3.0, "logp": 1.0, "ionization_type": "acid"},
        {"name": "Base", "pka": 9.0, "logp": 1.0, "ionization_type": "base"},
        {"name": "Neutral", "pka": 7.0, "logp": 1.0, "ionization_type": "neutral"},
    ]
    results = screen_sweat_excretion(candidates)
    ratios = [r.sweat_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


# ---------------------------------------------------------------------------
# 22. screen_sweat_excretion — default ionization neutral, returns all
# ---------------------------------------------------------------------------


def test_screen_default_ionization():
    candidates = [{"name": "Drug", "pka": 7.0, "logp": 1.0}]
    results = screen_sweat_excretion(candidates)
    assert results[0].ionization_type == "neutral"


def test_screen_returns_all():
    candidates = [
        {"name": f"D{i}", "pka": 7.0, "logp": float(i)}
        for i in range(6)
    ]
    results = screen_sweat_excretion(candidates)
    assert len(results) == 6
