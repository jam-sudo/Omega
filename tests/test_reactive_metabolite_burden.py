"""Tests for reactive_metabolite_burden module (Phase 279)."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.reactive_metabolite_burden import (
    MetaboliteBurdenResult,
    compute_metabolite_burden,
    screen_compounds,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BASE = dict(
    compound_name="TestDrug",
    daily_dose_mg=100.0,
    fm_cyp3a4=0.5,
    fm_cyp2d6=0.2,
    fm_cyp2c9=0.1,
    fm_cyp1a2=0.1,
    clint_L_per_h=1.0,
    mw=350.0,
    logP=2.0,
)


def _compute(**overrides) -> MetaboliteBurdenResult:
    kwargs = {**_BASE, **overrides}
    return compute_metabolite_burden(**kwargs)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_invalid_daily_dose_zero():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        _compute(daily_dose_mg=0.0)


def test_invalid_daily_dose_negative():
    with pytest.raises(ValueError, match="daily_dose_mg"):
        _compute(daily_dose_mg=-10.0)


def test_invalid_clint_zero():
    with pytest.raises(ValueError, match="clint"):
        _compute(clint_L_per_h=0.0)


def test_invalid_clint_negative():
    with pytest.raises(ValueError, match="clint"):
        _compute(clint_L_per_h=-1.0)


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        _compute(mw=0.0)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        _compute(mw=-100.0)


def test_invalid_fm_negative():
    with pytest.raises(ValueError, match="CYP3A4"):
        _compute(fm_cyp3a4=-0.1)


def test_invalid_fm_above_one():
    with pytest.raises(ValueError, match="CYP2D6"):
        _compute(fm_cyp2d6=1.1)


def test_invalid_fm_sum_exceeds_one():
    with pytest.raises(ValueError, match="Sum of fm"):
        _compute(fm_cyp3a4=0.6, fm_cyp2d6=0.5, fm_cyp2c9=0.0, fm_cyp1a2=0.0)


def test_fm_sum_exactly_one_valid():
    # Should not raise
    result = _compute(fm_cyp3a4=0.5, fm_cyp2d6=0.3, fm_cyp2c9=0.1, fm_cyp1a2=0.1)
    assert result.total_burden > 0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


def test_result_is_frozen():
    result = _compute()
    with pytest.raises(Exception):
        result.total_burden = 0.0  # type: ignore[misc]


def test_result_fields_populated():
    result = _compute()
    assert result.compound_name == "TestDrug"
    assert result.daily_dose_mg == 100.0
    assert result.fm_cyp3a4 == 0.5
    assert isinstance(result.pathway_burdens, dict)
    assert "CYP3A4" in result.pathway_burdens
    assert "CYP2D6" in result.pathway_burdens
    assert "CYP2C9" in result.pathway_burdens
    assert "CYP1A2" in result.pathway_burdens


# ---------------------------------------------------------------------------
# Burden computation
# ---------------------------------------------------------------------------


def test_total_burden_positive():
    result = _compute()
    assert result.total_burden > 0.0


def test_total_burden_is_sum_of_pathways():
    result = _compute()
    expected = sum(result.pathway_burdens.values())
    assert abs(result.total_burden - expected) < 1e-9


def test_logp_factor_above_2_increases_burden():
    low = _compute(logP=2.0)
    high = _compute(logP=5.0)
    assert high.total_burden > low.total_burden


def test_logp_factor_below_2_no_extra_burden():
    r1 = _compute(logP=0.0)
    r2 = _compute(logP=2.0)
    # Both should use logP_factor=1.0 (no extra for logP <= 2)
    assert abs(r1.total_burden - r2.total_burden) < 1e-9


def test_burden_scales_linearly_with_dose():
    r1 = _compute(daily_dose_mg=50.0)
    r2 = _compute(daily_dose_mg=100.0)
    assert abs(r2.total_burden / r1.total_burden - 2.0) < 1e-9


def test_burden_scales_linearly_with_clint():
    r1 = _compute(clint_L_per_h=0.5)
    r2 = _compute(clint_L_per_h=1.0)
    assert abs(r2.total_burden / r1.total_burden - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------


def test_risk_low():
    # tiny dose + tiny clint => low burden
    result = _compute(daily_dose_mg=1.0, clint_L_per_h=0.001, fm_cyp3a4=0.01,
                      fm_cyp2d6=0.0, fm_cyp2c9=0.0, fm_cyp1a2=0.0, logP=2.0)
    assert result.risk_category == "low"


def test_risk_high():
    # large dose + large clint => high burden
    result = _compute(daily_dose_mg=500.0, clint_L_per_h=5.0)
    assert result.risk_category == "high"


def test_risk_moderate():
    # Calibrate for moderate: total_burden in [1, 5)
    # fm_3a4=0.1, clint=0.5, dose=50 => 0.1*0.5*50*1 = 2.5
    result = _compute(
        daily_dose_mg=50.0,
        fm_cyp3a4=0.1,
        fm_cyp2d6=0.0,
        fm_cyp2c9=0.0,
        fm_cyp1a2=0.0,
        clint_L_per_h=0.5,
        logP=2.0,
    )
    assert result.risk_category == "moderate"


# ---------------------------------------------------------------------------
# Dominant pathway
# ---------------------------------------------------------------------------


def test_dominant_pathway_cyp3a4():
    result = _compute(fm_cyp3a4=0.8, fm_cyp2d6=0.1, fm_cyp2c9=0.05, fm_cyp1a2=0.05)
    assert result.dominant_pathway == "CYP3A4"


def test_dominant_pathway_cyp2d6():
    result = _compute(fm_cyp3a4=0.05, fm_cyp2d6=0.8, fm_cyp2c9=0.05, fm_cyp1a2=0.05)
    assert result.dominant_pathway == "CYP2D6"


def test_dominant_pathway_cyp1a2():
    result = _compute(fm_cyp3a4=0.05, fm_cyp2d6=0.05, fm_cyp2c9=0.05, fm_cyp1a2=0.8)
    assert result.dominant_pathway == "CYP1A2"


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


def test_screen_compounds_returns_sorted_desc():
    compounds = [
        {**_BASE, "compound_name": "Low", "daily_dose_mg": 10.0},
        {**_BASE, "compound_name": "High", "daily_dose_mg": 500.0},
        {**_BASE, "compound_name": "Mid", "daily_dose_mg": 100.0},
    ]
    results = screen_compounds(compounds)
    assert results[0].compound_name == "High"
    assert results[-1].compound_name == "Low"


def test_screen_compounds_all_results():
    compounds = [
        {**_BASE, "compound_name": "A"},
        {**_BASE, "compound_name": "B"},
    ]
    results = screen_compounds(compounds)
    assert len(results) == 2


def test_screen_compounds_empty():
    results = screen_compounds([])
    assert results == []
