"""Tests for Phase 915 — Drug Cochlear Distribution."""

import pytest

from omega_pbpk.prediction.cochlear_distribution import (
    CochlearDistributionResult,
    predict_cochlear_distribution,
    screen_ototoxic_drugs,
)

# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw_Da must be > 0"):
        predict_cochlear_distribution("Drug", mw_Da=0.0)


def test_invalid_plasma_conc_raises():
    with pytest.raises(ValueError, match="plasma_conc_mg_L must be > 0"):
        predict_cochlear_distribution("Drug", plasma_conc_mg_L=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route must be"):
        predict_cochlear_distribution("Drug", route="inhaled")


# ---------------------------------------------------------------------------
# BLB permeability classification
# ---------------------------------------------------------------------------


def test_blb_high_permeability():
    # blb_score = logp - mw_Da/400; high if > 0.5
    # logp=3, mw=400 → score = 3-1 = 2.0 → high
    r = predict_cochlear_distribution("HighPermDrug", logp=3.0, mw_Da=400.0)
    assert r.blb_permeability_class == "high"


def test_blb_moderate_permeability():
    # logp=0, mw=200 → score = 0 - 0.5 = -0.5 → moderate
    r = predict_cochlear_distribution("ModPermDrug", logp=0.0, mw_Da=200.0)
    assert r.blb_permeability_class == "moderate"


def test_blb_low_permeability():
    # logp=-2, mw=800 → score = -2 - 2 = -4 → low
    r = predict_cochlear_distribution("LowPermDrug", logp=-2.0, mw_Da=800.0)
    assert r.blb_permeability_class == "low"


# ---------------------------------------------------------------------------
# Perilymph-plasma ratio
# ---------------------------------------------------------------------------


def test_high_blb_perilymph_ratio_systemic():
    r = predict_cochlear_distribution("HighPerm", logp=3.0, mw_Da=400.0, route="systemic")
    assert abs(r.perilymph_plasma_ratio - 0.3) < 1e-6


def test_moderate_blb_perilymph_ratio_systemic():
    r = predict_cochlear_distribution("ModPerm", logp=0.0, mw_Da=200.0, route="systemic")
    assert abs(r.perilymph_plasma_ratio - 0.05) < 1e-6


def test_low_blb_perilymph_ratio_systemic():
    r = predict_cochlear_distribution("LowPerm", logp=-2.0, mw_Da=800.0, route="systemic")
    assert abs(r.perilymph_plasma_ratio - 0.005) < 1e-6


def test_intratympanic_multiplies_ratio():
    r_sys = predict_cochlear_distribution("Drug", logp=0.0, mw_Da=200.0, route="systemic")
    r_it = predict_cochlear_distribution("Drug", logp=0.0, mw_Da=200.0, route="intratympanic")
    assert abs(r_it.perilymph_plasma_ratio - r_sys.perilymph_plasma_ratio * 10.0) < 1e-6


def test_met_channel_doubles_ratio():
    r_no = predict_cochlear_distribution("Drug", logp=0.0, mw_Da=200.0, has_met_channel_entry=False)
    r_yes = predict_cochlear_distribution("Drug", logp=0.0, mw_Da=200.0, has_met_channel_entry=True)
    assert abs(r_yes.perilymph_plasma_ratio - r_no.perilymph_plasma_ratio * 2.0) < 1e-6


def test_ratio_clamped_max():
    # logp=3, mw=100, intratympanic + MET: base=0.3*10*2=6 → clamp to 5.0
    r = predict_cochlear_distribution(
        "Drug", logp=3.0, mw_Da=100.0, route="intratympanic", has_met_channel_entry=True
    )
    assert r.perilymph_plasma_ratio == 5.0


# ---------------------------------------------------------------------------
# Predicted perilymph concentration
# ---------------------------------------------------------------------------


def test_predicted_perilymph_conc():
    # ratio=0.05, plasma=2.0 mg/L, fu=0.5 → 0.05*2.0*0.5*1000 = 50 ug/mL
    r = predict_cochlear_distribution(
        "Drug", logp=0.0, mw_Da=200.0, plasma_conc_mg_L=2.0, fu_plasma=0.5
    )
    assert abs(r.predicted_perilymph_conc_ug_mL - 50.0) < 1e-4


# ---------------------------------------------------------------------------
# Hair cell accumulation factor
# ---------------------------------------------------------------------------


def test_hair_cell_factor_met_channel():
    r = predict_cochlear_distribution("Aminoglycoside", has_met_channel_entry=True)
    assert r.hair_cell_accumulation_factor == 50.0


def test_hair_cell_factor_no_met_channel():
    # 1 + logp*2, logp=2 → 5.0
    r = predict_cochlear_distribution("Drug", logp=2.0, has_met_channel_entry=False)
    assert abs(r.hair_cell_accumulation_factor - 5.0) < 1e-6


def test_hair_cell_factor_clamped_low():
    # logp=-5 → 1 + (-5)*2 = -9 → clamp to 1.0
    r = predict_cochlear_distribution("Drug", logp=-5.0, has_met_channel_entry=False)
    assert r.hair_cell_accumulation_factor == 1.0


# ---------------------------------------------------------------------------
# Ototoxicity risk level
# ---------------------------------------------------------------------------


def test_ototoxicity_high_risk():
    # Use MET channel + high BLB to drive score high
    r = predict_cochlear_distribution(
        "Gentamicin", logp=3.0, mw_Da=400.0, route="intratympanic", has_met_channel_entry=True
    )
    assert r.ototoxicity_risk_level == "high"


def test_ototoxicity_low_risk():
    r = predict_cochlear_distribution("SafeDrug", logp=-2.0, mw_Da=800.0)
    assert r.ototoxicity_risk_level == "low"


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_high_risk():
    r = predict_cochlear_distribution(
        "ToxicDrug", logp=3.0, mw_Da=400.0, route="intratympanic", has_met_channel_entry=True
    )
    assert "audiometry" in r.notes.lower()


def test_notes_met_channel_moderate():
    # MET channel but not high risk → notes about MET
    # low BLB, systemic, MET → ratio = 0.005*2 = 0.01, score = 0.01*100 + 50*0.5 = 1+25 = 26 → low
    r = predict_cochlear_distribution(
        "Aminoglycoside", logp=-2.0, mw_Da=800.0, route="systemic", has_met_channel_entry=True
    )
    assert "MET channel" in r.notes


def test_notes_low_cochlear():
    r = predict_cochlear_distribution(
        "SafeDrug", logp=-2.0, mw_Da=800.0, has_met_channel_entry=False
    )
    assert "Low cochlear" in r.notes


# ---------------------------------------------------------------------------
# Return type and frozen
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    r = predict_cochlear_distribution("Drug")
    assert isinstance(r, CochlearDistributionResult)


def test_result_is_frozen():
    r = predict_cochlear_distribution("Drug")
    with pytest.raises((AttributeError, TypeError)):
        r.logp = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# screen_ototoxic_drugs
# ---------------------------------------------------------------------------


def test_screen_ototoxic_drugs_sorted():
    candidates = [
        {"name": "Safe", "logp": -2.0, "mw_Da": 800.0},
        {"name": "Toxic", "logp": 3.0, "mw_Da": 300.0, "has_met_channel_entry": True},
        {"name": "Medium", "logp": 1.0, "mw_Da": 400.0},
    ]
    results = screen_ototoxic_drugs(candidates)
    assert len(results) == 3
    scores = [r.ototoxicity_risk_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].drug_name == "Toxic"


def test_screen_empty_list():
    results = screen_ototoxic_drugs([])
    assert results == []
