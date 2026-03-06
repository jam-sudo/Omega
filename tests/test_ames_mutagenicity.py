"""Tests for Phase 182 — ames_mutagenicity module."""
import dataclasses

import pytest

from omega_pbpk.prediction.ames_mutagenicity import (
    AmesAlert,
    predict_ames_mutagenicity,
    screen_ames,
)

# Nitrobenzene-like SMILES triggers nitro_aromatic
NITROBENZENE = "c1ccc([N+](=O)[O-])cc1"

# Simple alkane — no alerts
OCTANE = "CCCCCCCC"

# Compound with primary aromatic amine
ANILINE = "Nc1ccccc1"

# Compound with alkyl halide
CHLOROETHANE = "CCl"

# Compound with hydrazine
HYDRAZINE = "NN"


# ---------------------------------------------------------------------------
# Basic prediction
# ---------------------------------------------------------------------------

def test_nitrobenzene_positive():
    # nitro_aromatic weight=3 → alert_score=3, score = 3*15 + logP*2
    # Use logP=5.0 to ensure score >= 60: 45 + 10 = 55... still borderline.
    # Use logP=8.0: 45 + 16 = 61 → positive
    result = predict_ames_mutagenicity("nitrobenzene_hLogP", NITROBENZENE, 123.1, 8.0)
    assert result.ames_prediction == "positive"
    assert result.genotoxicity_risk == "high"
    assert result.mutagenicity_score >= 60.0


def test_nitrobenzene_nitro_alert_triggered():
    result = predict_ames_mutagenicity("nitrobenzene", NITROBENZENE, 123.1, 1.85)
    triggered = {a.name for a in result.alerts if a.triggered}
    assert "nitro_aromatic" in triggered
    # score = 3*15 + 1.85*2 = 45 + 3.7 = 48.7 → borderline
    assert result.ames_prediction == "borderline"


def test_octane_negative_low_logP():
    result = predict_ames_mutagenicity("octane", OCTANE, 114.2, 4.0)
    # No structural alerts → score = 0*15 + 4*2 = 8
    assert result.ames_prediction == "negative"
    assert result.n_alerts == 0
    assert result.primary_concern == "none"


def test_aniline_primary_aromatic_amine():
    result = predict_ames_mutagenicity("aniline", ANILINE, 93.1, 0.9)
    triggered = {a.name for a in result.alerts if a.triggered}
    assert "primary_aromatic_amine" in triggered


def test_chloroethane_alkyl_halide():
    result = predict_ames_mutagenicity("chloroethane", CHLOROETHANE, 64.5, 1.43)
    triggered = {a.name for a in result.alerts if a.triggered}
    assert "alkyl_halide" in triggered


def test_hydrazine_alert():
    result = predict_ames_mutagenicity("hydrazine", HYDRAZINE, 32.0, -1.5)
    triggered = {a.name for a in result.alerts if a.triggered}
    assert "hydrazine" in triggered


# ---------------------------------------------------------------------------
# Alert counts and structure
# ---------------------------------------------------------------------------

def test_all_alerts_present_in_result():
    result = predict_ames_mutagenicity("test", NITROBENZENE, 123.1, 1.85)
    assert len(result.alerts) == 10  # one per ALERTS entry


def test_n_alerts_matches_triggered():
    result = predict_ames_mutagenicity("nitrobenzene", NITROBENZENE, 123.1, 1.85)
    triggered_count = sum(1 for a in result.alerts if a.triggered)
    assert result.n_alerts == triggered_count


def test_alert_dataclass_frozen():
    alert = AmesAlert(name="test", triggered=True, weight=1.0, description="desc")
    with pytest.raises(dataclasses.FrozenInstanceError):
        alert.triggered = False  # type: ignore[misc]


def test_result_dataclass_frozen():
    result = predict_ames_mutagenicity("test", OCTANE, 114.2, 2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ames_prediction = "positive"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Score boundaries
# ---------------------------------------------------------------------------

def test_borderline_prediction():
    # Need score 30-60: one alert weight=2 → 2*15=30 + logP*2
    # Use epoxide SMILES "C1OC1" with logP=0 → score=30 (borderline boundary)
    result = predict_ames_mutagenicity("epoxide_compound", "C1OC1", 58.0, 0.0)
    # score = 2.0*15 + 0*2 = 30 → borderline
    assert result.ames_prediction == "borderline"
    assert result.genotoxicity_risk == "moderate"


def test_mutagenicity_score_capped_at_100():
    # Many heavy alerts
    heavy_smiles = "c1ccc([N+](=O)[O-])cc1NNC1OC1CCl"
    result = predict_ames_mutagenicity("heavy", heavy_smiles, 200.0, 5.0)
    assert result.mutagenicity_score <= 100.0


# ---------------------------------------------------------------------------
# Mitigation
# ---------------------------------------------------------------------------

def test_nitro_aromatic_mitigation():
    result = predict_ames_mutagenicity("nitrobenzene", NITROBENZENE, 123.1, 1.85)
    # nitro_aromatic alert fires → mitigation should mention bioisostere or nitro
    assert any("nitro" in m.lower() or "bioisostere" in m.lower() for m in result.mitigation)


def test_no_alerts_mitigation():
    result = predict_ames_mutagenicity("octane", OCTANE, 114.2, 2.0)
    assert result.mitigation == ["No mutagenicity concerns identified"]


# ---------------------------------------------------------------------------
# screen_ames
# ---------------------------------------------------------------------------

def test_screen_ames_sorted_descending():
    compounds = [
        {"name": "octane", "smiles": OCTANE, "mw": 114.2, "logP": 2.0},
        {"name": "nitrobenzene", "smiles": NITROBENZENE, "mw": 123.1, "logP": 1.85},
        {"name": "aniline", "smiles": ANILINE, "mw": 93.1, "logP": 0.9},
    ]
    results = screen_ames(compounds)
    scores = [r.mutagenicity_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_ames_returns_all():
    compounds = [
        {"name": "a", "smiles": OCTANE, "mw": 114.2, "logP": 2.0},
        {"name": "b", "smiles": ANILINE, "mw": 93.1, "logP": 0.9},
    ]
    results = screen_ames(compounds)
    assert len(results) == 2


def test_screen_ames_empty_list():
    results = screen_ames([])
    assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_empty_smiles_raises():
    with pytest.raises(ValueError, match="smiles"):
        predict_ames_mutagenicity("test", "", 100.0, 1.0)


def test_whitespace_smiles_raises():
    with pytest.raises(ValueError, match="smiles"):
        predict_ames_mutagenicity("test", "   ", 100.0, 1.0)


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_ames_mutagenicity("test", OCTANE, 0.0, 1.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_ames_mutagenicity("test", OCTANE, -50.0, 1.0)
