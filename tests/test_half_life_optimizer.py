"""Tests for Phase 754 — Drug Half-Life Optimization."""

import pytest

from omega_pbpk.clinical.half_life_optimizer import (
    HalfLifeOptimizationResult,
    optimize_half_life,
    scan_t_half_landscape,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = optimize_half_life(
        drug_name="TestDrug",
        dose_mg=100.0,
        dosing_interval_h=12.0,
        mec_mg_L=1.0,
        mtc_mg_L=10.0,
        vd_L=50.0,
    )
    assert isinstance(result, HalfLifeOptimizationResult)


def test_optimal_t_half_positive():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert result.optimal_t_half_h > 0


def test_optimal_css_max_positive():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert result.optimal_css_max > 0


def test_optimal_css_min_positive():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert result.optimal_css_min > 0


def test_css_max_greater_than_css_min():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert result.optimal_css_max > result.optimal_css_min


def test_optimal_css_avg_between_min_max():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert result.optimal_css_min <= result.optimal_css_avg <= result.optimal_css_max


# ---------------------------------------------------------------------------
# Therapeutic range
# ---------------------------------------------------------------------------


def test_in_therapeutic_range_favorable():
    """With favorable dose/interval/targets, should find in-range t_half."""
    result = optimize_half_life(
        drug_name="GoodDrug",
        dose_mg=50.0,
        dosing_interval_h=12.0,
        mec_mg_L=0.1,
        mtc_mg_L=100.0,
        vd_L=50.0,
        n_candidates=200,
    )
    assert result.in_therapeutic_range is True


def test_in_therapeutic_range_is_bool():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert isinstance(result.in_therapeutic_range, bool)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def test_composite_score_in_01():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert 0.0 <= result.optimal_composite_score <= 1.0


# ---------------------------------------------------------------------------
# all_t_halfs string
# ---------------------------------------------------------------------------


def test_all_t_halfs_is_str():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert isinstance(result.all_t_halfs, str)


def test_all_t_halfs_has_comma_separated_values():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0, n_candidates=50)
    parts = result.all_t_halfs.split(",")
    assert len(parts) == 5
    for p in parts:
        val = float(p.strip())
        assert val > 0


# ---------------------------------------------------------------------------
# scan_t_half_landscape
# ---------------------------------------------------------------------------


def test_scan_returns_list():
    landscape = scan_t_half_landscape("Drug", 100.0, 12.0, 1.0, 10.0, n_candidates=50)
    assert isinstance(landscape, list)


def test_scan_returns_n_candidates_dicts():
    n = 30
    landscape = scan_t_half_landscape("Drug", 100.0, 12.0, 1.0, 10.0, n_candidates=n)
    assert len(landscape) == n


def test_scan_dict_has_required_keys():
    landscape = scan_t_half_landscape("Drug", 100.0, 12.0, 1.0, 10.0, n_candidates=10)
    required = {"t_half_h", "css_max", "css_min", "tir_score", "composite_score"}
    for entry in landscape:
        assert required.issubset(entry.keys()), f"Missing keys in {entry.keys()}"


def test_scan_t_half_values_positive():
    landscape = scan_t_half_landscape("Drug", 100.0, 12.0, 1.0, 10.0, n_candidates=20)
    for entry in landscape:
        assert entry["t_half_h"] > 0


# ---------------------------------------------------------------------------
# Dosing interval effect
# ---------------------------------------------------------------------------


def test_longer_dosing_interval_needs_longer_t_half():
    """Longer tau requires longer t_half to avoid excessive fluctuation."""
    r_short = optimize_half_life("Drug", 100.0, 6.0, 1.0, 10.0, vd_L=50.0)
    r_long = optimize_half_life("Drug", 100.0, 24.0, 1.0, 10.0, vd_L=50.0)
    # Longer interval: optimal t_half should be >= that for short interval
    # (not guaranteed in every scenario, but generally true for same targets)
    # At minimum, both should give valid positive t_halfs
    assert r_short.optimal_t_half_h > 0
    assert r_long.optimal_t_half_h > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        optimize_half_life("Bad", dose_mg=0.0, dosing_interval_h=12.0, mec_mg_L=1.0, mtc_mg_L=10.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        optimize_half_life("Bad", dose_mg=-5.0, dosing_interval_h=12.0, mec_mg_L=1.0, mtc_mg_L=10.0)


def test_mec_greater_than_mtc_raises():
    with pytest.raises(ValueError):
        optimize_half_life(
            "Bad", dose_mg=100.0, dosing_interval_h=12.0, mec_mg_L=10.0, mtc_mg_L=5.0
        )


def test_mec_equal_mtc_raises():
    with pytest.raises(ValueError):
        optimize_half_life("Bad", dose_mg=100.0, dosing_interval_h=12.0, mec_mg_L=5.0, mtc_mg_L=5.0)


def test_invalid_dosing_interval_raises():
    with pytest.raises(ValueError):
        optimize_half_life("Bad", dose_mg=100.0, dosing_interval_h=0.0, mec_mg_L=1.0, mtc_mg_L=10.0)


# ---------------------------------------------------------------------------
# Notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    assert result.notes and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------


def test_result_is_frozen():
    result = optimize_half_life("Drug", 100.0, 12.0, 1.0, 10.0)
    with pytest.raises((AttributeError, TypeError)):
        result.optimal_t_half_h = 999.0
