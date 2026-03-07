"""Tests for DILI risk scoring (Phase 361)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.risk.dili_risk import (
    DILIRiskResult,
    assess_dili_risk,
    rank_dili_risk,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# A "clean" SMILES with no structural alerts: simple ethyl chain
CLEAN_SMILES = "CCC"
CLEAN_MW = 44.0
CLEAN_LOGP = 1.0  # low logP, low MW → no mito/BSEP risk

# Furan-containing SMILES
FURAN_SMILES = "c1ccoc1"  # furan itself
FURAN_MW = 68.0
FURAN_LOGP = 1.5

# Multiple alerts: furan + nitro
MULTI_SMILES = "c1ccoc1[n+](=o)[o-]"
MULTI_MW = 113.0
MULTI_LOGP = 0.5

# High logP + high MW compound (mitochondrial + BSEP risk)
HIGHMW_SMILES = "CCCCC"
HIGHMW_MW = 500.0
HIGHMW_LOGP = 4.0


def _clean_result(dose: float = 10.0) -> DILIRiskResult:
    return assess_dili_risk(
        compound_name="clean",
        smiles=CLEAN_SMILES,
        daily_dose_mg=dose,
        mw=CLEAN_MW,
        logP=CLEAN_LOGP,
    )


def _furan_result(dose: float = 10.0) -> DILIRiskResult:
    return assess_dili_risk(
        compound_name="furan_drug",
        smiles=FURAN_SMILES,
        daily_dose_mg=dose,
        mw=FURAN_MW,
        logP=FURAN_LOGP,
    )


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_returns_dili_risk_result():
    r = _clean_result()
    assert isinstance(r, DILIRiskResult)


def test_frozen_dataclass_immutable():
    r = _clean_result()
    with pytest.raises((AttributeError, TypeError)):
        r.dili_risk_score = 99.0  # type: ignore[misc]


def test_clean_compound_no_alerts():
    r = _clean_result()
    assert r.n_alerts == 0
    assert r.alerts_triggered == []


def test_clean_score_approximately_dose_risk():
    dose = 100.0
    r = assess_dili_risk("c", CLEAN_SMILES, dose, CLEAN_MW, CLEAN_LOGP)
    expected_dose_risk = math.log10(dose)
    # No alerts, no mito/BSEP → score ≈ dose_risk
    assert r.dili_risk_score == pytest.approx(expected_dose_risk, abs=0.01)


def test_dose_risk_score_value():
    dose = 50.0
    r = assess_dili_risk("c", CLEAN_SMILES, dose, CLEAN_MW, CLEAN_LOGP)
    assert r.dose_risk_score == pytest.approx(math.log10(dose), rel=1e-6)


# ---------------------------------------------------------------------------
# Furan alert
# ---------------------------------------------------------------------------


def test_furan_alert_detected():
    r = _furan_result()
    assert "furan_ring" in r.alerts_triggered


def test_furan_increases_score():
    clean = _clean_result(dose=10.0)
    furan = _furan_result(dose=10.0)
    assert furan.dili_risk_score > clean.dili_risk_score


def test_furan_score_increment_equals_weight():
    """furan weight=3.0; compare same dose, same mw/logP except SMILES."""
    # Use FURAN_LOGP=1.5, CLEAN_LOGP=1.0 — both below mito/BSEP thresholds
    clean = assess_dili_risk("c", CLEAN_SMILES, 10.0, FURAN_MW, FURAN_LOGP)
    furan = assess_dili_risk("f", FURAN_SMILES, 10.0, FURAN_MW, FURAN_LOGP)
    assert furan.dili_risk_score - clean.dili_risk_score == pytest.approx(3.0, abs=0.1)


def test_furan_reactive_metabolite_risk():
    r = _furan_result()
    assert r.reactive_metabolite_risk is True


# ---------------------------------------------------------------------------
# Multiple alerts
# ---------------------------------------------------------------------------


def test_multi_alert_compound():
    r = assess_dili_risk("multi", MULTI_SMILES, 10.0, MULTI_MW, MULTI_LOGP)
    # furan_ring (3.0) + nitro_group (2.0) = 5.0 at minimum
    assert r.n_alerts >= 2


def test_multi_alert_cumulative_score():
    single = _furan_result(dose=10.0)
    multi = assess_dili_risk("multi", MULTI_SMILES, 10.0, MULTI_MW, MULTI_LOGP)
    assert multi.dili_risk_score > single.dili_risk_score


def test_acyl_halide_alert():
    smiles = "CC(=O)Cl"  # acetyl chloride
    r = assess_dili_risk("acyl_cl", smiles, 10.0, 78.5, 0.5)
    assert "acyl_halide" in r.alerts_triggered


def test_aniline_alert():
    smiles = "Nc1ccccc1"
    r = assess_dili_risk("aniline", smiles, 10.0, 93.1, 1.0)
    assert "aniline" in r.alerts_triggered


# ---------------------------------------------------------------------------
# Dose effects
# ---------------------------------------------------------------------------


def test_higher_dose_higher_score():
    low = assess_dili_risk("x", CLEAN_SMILES, 10.0, CLEAN_MW, CLEAN_LOGP)
    high = assess_dili_risk("x", CLEAN_SMILES, 1000.0, CLEAN_MW, CLEAN_LOGP)
    assert high.dili_risk_score > low.dili_risk_score


def test_dose_proportional_to_log10():
    r1 = assess_dili_risk("x", CLEAN_SMILES, 10.0, CLEAN_MW, CLEAN_LOGP)
    r2 = assess_dili_risk("x", CLEAN_SMILES, 100.0, CLEAN_MW, CLEAN_LOGP)
    delta = r2.dili_risk_score - r1.dili_risk_score
    assert delta == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Category thresholds
# ---------------------------------------------------------------------------


def test_most_concern_category_when_score_ge_5():
    # furan (3.0) + nitro (2.0) + dose_risk(log10(100))=2 → 7 → most_concern
    r = assess_dili_risk("danger", MULTI_SMILES, 100.0, MULTI_MW, MULTI_LOGP)
    assert r.dili_category == "most_concern"


def test_no_concern_category_low_score():
    r = assess_dili_risk("safe", CLEAN_SMILES, 1.0, CLEAN_MW, CLEAN_LOGP)
    # score = log10(1.0)=0 → no_concern
    assert r.dili_category == "no_concern"


def test_less_concern_category():
    # Need score in [2, 5): furan(3.0) + log10(5)≈0.7 = 3.7 → less_concern
    r = assess_dili_risk("mild", FURAN_SMILES, 5.0, FURAN_MW, FURAN_LOGP)
    assert r.dili_category == "less_concern"


# ---------------------------------------------------------------------------
# Dose threshold & safe fraction
# ---------------------------------------------------------------------------


def test_safe_dose_fraction_above_threshold():
    # most_concern threshold=50mg; dose=500mg → fraction=10
    r = assess_dili_risk("danger", MULTI_SMILES, 500.0, MULTI_MW, MULTI_LOGP)
    assert r.safe_dose_fraction > 1.0


def test_safe_dose_fraction_below_threshold():
    # no_concern threshold=500mg; dose=10mg → fraction=0.02
    r = assess_dili_risk("safe", CLEAN_SMILES, 10.0, CLEAN_MW, CLEAN_LOGP)
    assert r.safe_dose_fraction < 1.0


def test_dose_threshold_most_concern():
    r = assess_dili_risk("danger", MULTI_SMILES, 100.0, MULTI_MW, MULTI_LOGP)
    if r.dili_category == "most_concern":
        assert r.dose_threshold_mg == 50.0


def test_dose_threshold_no_concern():
    r = assess_dili_risk("safe", CLEAN_SMILES, 1.0, CLEAN_MW, CLEAN_LOGP)
    assert r.dose_threshold_mg == 500.0


# ---------------------------------------------------------------------------
# Mitochondrial and BSEP risk
# ---------------------------------------------------------------------------


def test_mitochondrial_risk_high_logp_high_mw():
    r = assess_dili_risk("amphiphile", HIGHMW_SMILES, 50.0, HIGHMW_MW, HIGHMW_LOGP)
    assert r.mitochondrial_risk is True


def test_no_mitochondrial_risk_low_logp():
    r = assess_dili_risk("x", CLEAN_SMILES, 50.0, CLEAN_MW, CLEAN_LOGP)
    assert r.mitochondrial_risk is False


def test_bsep_risk_high_logp_high_mw():
    r = assess_dili_risk("bsep", HIGHMW_SMILES, 50.0, HIGHMW_MW, HIGHMW_LOGP)
    assert r.bile_acid_transport_risk is True


# ---------------------------------------------------------------------------
# Recommendation and notes
# ---------------------------------------------------------------------------


def test_recommendation_non_empty():
    r = _clean_result()
    assert len(r.recommendation) > 0


def test_recommendation_non_empty_high_risk():
    r = assess_dili_risk("danger", MULTI_SMILES, 500.0, MULTI_MW, MULTI_LOGP)
    assert len(r.recommendation) > 0


def test_notes_non_empty():
    r = _clean_result()
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------


def test_score_capped_at_10():
    # Pile on alerts with huge dose
    smiles = "c1ccoc1c1ccoc1[n+](=o)[o-]c(=o)cl"
    r = assess_dili_risk("pile", smiles, 10000.0, 800.0, 5.0)
    assert r.dili_risk_score <= 10.0


def test_score_non_negative():
    r = assess_dili_risk("safe", CLEAN_SMILES, 0.001, CLEAN_MW, CLEAN_LOGP)
    assert r.dili_risk_score >= 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        assess_dili_risk("x", CLEAN_SMILES, 0.0, CLEAN_MW, CLEAN_LOGP)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        assess_dili_risk("x", CLEAN_SMILES, -10.0, CLEAN_MW, CLEAN_LOGP)


def test_invalid_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        assess_dili_risk("x", CLEAN_SMILES, 10.0, 0.0, CLEAN_LOGP)


# ---------------------------------------------------------------------------
# rank_dili_risk
# ---------------------------------------------------------------------------


def test_rank_returns_sorted_descending():
    compounds = [
        {
            "name": "safe",
            "smiles": CLEAN_SMILES,
            "daily_dose_mg": 1.0,
            "mw": CLEAN_MW,
            "logP": CLEAN_LOGP,
        },
        {
            "name": "furan",
            "smiles": FURAN_SMILES,
            "daily_dose_mg": 10.0,
            "mw": FURAN_MW,
            "logP": FURAN_LOGP,
        },
        {
            "name": "danger",
            "smiles": MULTI_SMILES,
            "daily_dose_mg": 100.0,
            "mw": MULTI_MW,
            "logP": MULTI_LOGP,
        },
    ]
    ranked = rank_dili_risk(compounds)
    assert ranked[0].dili_risk_score >= ranked[1].dili_risk_score >= ranked[2].dili_risk_score


def test_rank_returns_list_of_results():
    compounds = [
        {
            "name": "a",
            "smiles": CLEAN_SMILES,
            "daily_dose_mg": 10.0,
            "mw": CLEAN_MW,
            "logP": CLEAN_LOGP,
        },
    ]
    result = rank_dili_risk(compounds)
    assert isinstance(result, list)
    assert all(isinstance(r, DILIRiskResult) for r in result)


def test_rank_length_matches_input():
    compounds = [
        {
            "name": f"cpd{i}",
            "smiles": CLEAN_SMILES,
            "daily_dose_mg": float(i * 10 + 1),
            "mw": CLEAN_MW,
            "logP": CLEAN_LOGP,
        }
        for i in range(5)
    ]
    ranked = rank_dili_risk(compounds)
    assert len(ranked) == 5


def test_rank_empty_list():
    assert rank_dili_risk([]) == []
