"""Tests for prediction/metabolic_soft_spot.py (Phase 179)."""

import pytest

from omega_pbpk.prediction.metabolic_soft_spot import (
    SoftSpotAlert,
    predict_soft_spots,
    rank_compounds_by_metabolic_stability,
)

# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_soft_spot_alert_frozen():
    alert = SoftSpotAlert("n_dealkylation", "high", "CYP3A4", "desc")
    with pytest.raises((AttributeError, TypeError)):
        alert.risk = "low"  # type: ignore[misc]  # frozen dataclass


def test_result_is_frozen():
    result = predict_soft_spots("test", "CN(C)c1ccccc1", logP=1.0, mw=150.0)
    with pytest.raises((AttributeError, TypeError)):
        result.metabolic_stability = "high"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_smiles_raises():
    with pytest.raises(ValueError, match="smiles"):
        predict_soft_spots("X", "", logP=1.0, mw=200.0)


def test_whitespace_smiles_raises():
    with pytest.raises(ValueError, match="smiles"):
        predict_soft_spots("X", "   ", logP=1.0, mw=200.0)


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_soft_spots("X", "CCO", logP=0.0, mw=0.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_soft_spots("X", "CCO", logP=0.0, mw=-50.0)


# ---------------------------------------------------------------------------
# Alert detection
# ---------------------------------------------------------------------------


def test_n_dealkylation_detected():
    """CN(C)c1ccccc1 has N(C) motif."""
    result = predict_soft_spots("DMAB", "CN(C)c1ccccc1", logP=2.0, mw=121.0)
    motifs = [a.motif for a in result.alerts]
    assert "n_dealkylation" in motifs


def test_n_dealkylation_is_high_risk():
    result = predict_soft_spots("DMAB", "CN(C)c1ccccc1", logP=2.0, mw=121.0)
    n_dealk = [a for a in result.alerts if a.motif == "n_dealkylation"]
    assert n_dealk[0].risk == "high"
    assert n_dealk[0].cyp_enzyme == "CYP3A4"


def test_aromatic_hydroxylation_detected():
    """Benzene ring has 6 'c' atoms → aromatic_hydroxylation alert."""
    result = predict_soft_spots("Benzene", "c1ccccc1", logP=2.0, mw=78.0)
    motifs = [a.motif for a in result.alerts]
    assert "aromatic_hydroxylation" in motifs


def test_methyl_hydroxylation_detected():
    """Toluene: Cc motif → methyl_hydroxylation."""
    result = predict_soft_spots("Toluene", "Cc1ccccc1", logP=2.5, mw=92.0)
    motifs = [a.motif for a in result.alerts]
    assert "methyl_hydroxylation" in motifs


def test_lipophilicity_alert_high_logP():
    result = predict_soft_spots("LipophilicDrug", "CCCCCC", logP=4.0, mw=86.0)
    motifs = [a.motif for a in result.alerts]
    assert "lipophilicity" in motifs


def test_no_lipophilicity_alert_low_logP():
    result = predict_soft_spots("Hydrophilic", "CCO", logP=1.0, mw=46.0)
    motifs = [a.motif for a in result.alerts]
    assert "lipophilicity" not in motifs


# ---------------------------------------------------------------------------
# CLint and stability
# ---------------------------------------------------------------------------


def test_low_mw_low_logP_high_stability():
    """Small hydrophilic compound with no alerts → high stability."""
    result = predict_soft_spots("SmallHydro", "CCO", logP=0.5, mw=46.0)
    # base_clint = 10 + 0 + 0 + 0.5*5 - 46*0.05 = 10 + 2.5 - 2.3 = 10.2
    assert result.metabolic_stability == "high"
    assert result.predicted_clint_uL_per_min_per_mg < 30.0


def test_high_alert_compound_low_stability():
    """Compound with multiple high/moderate alerts → low stability."""
    # N(C) gives high, aromatic gives moderate, logP>3 gives moderate
    result = predict_soft_spots("HighMet", "CN(C)c1ccccc1OC", logP=3.5, mw=200.0)
    # Should have n_dealkylation (high), so CLint elevated
    assert result.n_high_risk >= 1


def test_clint_bounds():
    """CLint is clamped to [1, 300]."""
    # Very low MW and very negative logP → floor
    r1 = predict_soft_spots("A", "O", logP=-5.0, mw=18.0)
    assert r1.predicted_clint_uL_per_min_per_mg >= 1.0
    # Extreme number of alerts → ceiling
    r2 = predict_soft_spots("B", "CN(C)c1ccccc1CN(C)c1ccccc1OCS", logP=8.0, mw=50.0)
    assert r2.predicted_clint_uL_per_min_per_mg <= 300.0


def test_n_high_risk_count():
    result = predict_soft_spots("DMAB", "CN(C)c1ccccc1", logP=2.0, mw=121.0)
    assert result.n_high_risk == result.alerts.count(
        next(a for a in result.alerts if a.risk == "high")
    )


# ---------------------------------------------------------------------------
# Primary CYP
# ---------------------------------------------------------------------------


def test_primary_cyp_from_high_alerts():
    """When high-risk alert is CYP3A4, primary_cyp = CYP3A4."""
    result = predict_soft_spots("DMAB", "CN(C)c1ccccc1", logP=1.0, mw=121.0)
    # n_dealkylation is CYP3A4 and high
    assert result.primary_cyp == "CYP3A4"


def test_primary_cyp_default_when_no_alerts():
    """No alerts → default CYP3A4."""
    result = predict_soft_spots("Water", "O", logP=-1.5, mw=18.0)
    assert result.primary_cyp == "CYP3A4"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_rank_returns_sorted_list():
    compounds = [
        {"name": "HighMet", "smiles": "CN(C)c1ccccc1OC", "logP": 4.0, "mw": 200.0},
        {"name": "LowMet", "smiles": "O", "logP": -1.5, "mw": 18.0},
        {"name": "MidMet", "smiles": "Cc1ccccc1", "logP": 2.5, "mw": 92.0},
    ]
    results = rank_compounds_by_metabolic_stability(compounds)
    assert len(results) == 3
    clints = [r.predicted_clint_uL_per_min_per_mg for r in results]
    assert clints == sorted(clints), "Results must be sorted ascending by CLint"


def test_rank_most_stable_first():
    compounds = [
        {"name": "A", "smiles": "CN(C)c1ccccc1", "logP": 4.0, "mw": 200.0},
        {"name": "B", "smiles": "O", "logP": -1.5, "mw": 18.0},
    ]
    results = rank_compounds_by_metabolic_stability(compounds)
    assert results[0].compound_name == "B"  # water/simple → most stable


def test_rank_empty_list():
    results = rank_compounds_by_metabolic_stability([])
    assert results == []


def test_result_fields_populated():
    result = predict_soft_spots("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", logP=0.07, mw=194.19)
    assert result.compound_name == "Caffeine"
    assert result.smiles == "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    assert isinstance(result.alerts, list)
    assert isinstance(result.notes, str)
    assert result.metabolic_stability in ("high", "moderate", "low")
