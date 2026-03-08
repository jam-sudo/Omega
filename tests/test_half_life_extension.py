"""Tests for half_life_extension.py (Phase 486)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.half_life_extension import (
    HalfLifeExtensionResult,
    compare_strategies,
    predict_half_life_extension,
)

_LN2 = 0.693147

_BASE_KWARGS = dict(
    drug_name="TestBiologic",
    strategy="PEGylation",
    parent_cl_l_per_h=2.0,
    parent_vd_l=10.0,
    parent_dose_mg=100.0,
)


def _run(**kwargs):
    kw = {**_BASE_KWARGS, **kwargs}
    return predict_half_life_extension(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = _run()
    assert isinstance(r, HalfLifeExtensionResult)


def test_result_is_frozen():
    r = _run()
    with pytest.raises((AttributeError, TypeError)):
        r.strategy = "albumin_fusion"


# ---------------------------------------------------------------------------
# Field correctness
# ---------------------------------------------------------------------------


def test_modified_cl_equals_parent_times_factor():
    r = _run(strategy="PEGylation", parent_cl_l_per_h=2.0)
    assert r.modified_cl_l_per_h == pytest.approx(2.0 * 0.2, rel=1e-9)


def test_modified_vd_equals_parent_times_factor():
    r = _run(strategy="PEGylation", parent_vd_l=10.0)
    assert r.modified_vd_l == pytest.approx(10.0 * 3.0, rel=1e-9)


def test_parent_t_half_formula():
    r = _run(parent_cl_l_per_h=2.0, parent_vd_l=10.0)
    expected = _LN2 * 10.0 / 2.0
    assert r.parent_t_half_h == pytest.approx(expected, rel=1e-6)


def test_modified_t_half_formula():
    r = _run(strategy="PEGylation", parent_cl_l_per_h=2.0, parent_vd_l=10.0)
    mod_cl = 2.0 * 0.2
    mod_vd = 10.0 * 3.0
    expected = _LN2 * mod_vd / mod_cl
    assert r.modified_t_half_h == pytest.approx(expected, rel=1e-6)


def test_t_half_extension_factor_exact():
    r = _run()
    expected = r.modified_t_half_h / r.parent_t_half_h
    assert r.t_half_extension_factor == pytest.approx(expected, rel=1e-9)


def test_target_dose_greater_than_parent_for_extended_t_half():
    r = _run()
    assert r.target_dose_mg > _BASE_KWARGS["parent_dose_mg"]


def test_target_dose_proportional_to_extension():
    parent_dose = 100.0
    r = _run(parent_dose_mg=parent_dose)
    expected_target = parent_dose * r.t_half_extension_factor
    assert r.target_dose_mg == pytest.approx(expected_target, rel=1e-9)


# ---------------------------------------------------------------------------
# FcRn_engineering highest t_half extension
# ---------------------------------------------------------------------------


def test_fcrn_engineering_highest_extension():
    # prodrug_depot: cl_factor=0.1, vd_factor=8.0 => extension=80x (highest)
    # FcRn_engineering: cl_factor=0.03, vd_factor=1.2 => extension=40x (second highest)
    # Verify FcRn_engineering beats PEGylation, albumin_fusion, nanoparticle
    strategies = ["PEGylation", "albumin_fusion", "FcRn_engineering", "nanoparticle"]
    results = {s: predict_half_life_extension("Drug", s, 2.0, 10.0, 100.0) for s in strategies}
    fcrn_factor = results["FcRn_engineering"].t_half_extension_factor
    for s, r in results.items():
        if s != "FcRn_engineering":
            assert fcrn_factor >= r.t_half_extension_factor


# ---------------------------------------------------------------------------
# Dosing frequency classification
# ---------------------------------------------------------------------------


def test_no_change_label():
    from omega_pbpk.clinical.half_life_extension import _classify_frequency

    assert _classify_frequency(1.5) == "no_change"


def test_2x_longer_label():
    from omega_pbpk.clinical.half_life_extension import _classify_frequency

    assert _classify_frequency(3.0) == "2x_longer"


def test_weekly_label():
    from omega_pbpk.clinical.half_life_extension import _classify_frequency

    assert _classify_frequency(5.0) == "weekly"


def test_biweekly_label():
    from omega_pbpk.clinical.half_life_extension import _classify_frequency

    assert _classify_frequency(10.0) == "biweekly"


def test_monthly_label():
    from omega_pbpk.clinical.half_life_extension import _classify_frequency

    assert _classify_frequency(20.0) == "monthly"


def test_quarterly_label():
    from omega_pbpk.clinical.half_life_extension import _classify_frequency

    assert _classify_frequency(50.0) == "quarterly"


def test_pegylation_frequency_label():
    # PEGylation: cl_factor=0.2, vd_factor=3.0 => extension = 3.0/0.2 = 15x => monthly (14-30x)
    r = _run(strategy="PEGylation")
    assert r.dosing_frequency_reduction == "monthly"


def test_fcrn_engineering_quarterly():
    # FcRn_engineering: extension = 1.2/0.03 = 40x => quarterly
    r = predict_half_life_extension("Drug", "FcRn_engineering", 2.0, 10.0, 100.0)
    assert r.dosing_frequency_reduction == "quarterly"


# ---------------------------------------------------------------------------
# compare_strategies
# ---------------------------------------------------------------------------


def test_compare_returns_five_items():
    results = compare_strategies("Drug", 2.0, 10.0, 100.0)
    assert len(results) == 5


def test_compare_sorted_by_modified_t_half_descending():
    results = compare_strategies("Drug", 2.0, 10.0, 100.0)
    t_halves = [r.modified_t_half_h for r in results]
    assert t_halves == sorted(t_halves, reverse=True)


def test_compare_all_strategies_represented():
    results = compare_strategies("Drug", 2.0, 10.0, 100.0)
    strats = {r.strategy for r in results}
    expected = {"PEGylation", "albumin_fusion", "FcRn_engineering", "nanoparticle", "prodrug_depot"}
    assert strats == expected


def test_compare_returns_half_life_extension_results():
    results = compare_strategies("Drug", 2.0, 10.0, 100.0)
    for r in results:
        assert isinstance(r, HalfLifeExtensionResult)


def test_compare_first_is_prodrug_depot():
    # prodrug_depot: cl_factor=0.1, vd_factor=8.0 => extension=80x (highest of all)
    results = compare_strategies("Drug", 2.0, 10.0, 100.0)
    assert results[0].strategy == "prodrug_depot"


# ---------------------------------------------------------------------------
# Stored fields
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    r = _run(drug_name="Adalimumab")
    assert r.drug_name == "Adalimumab"


def test_strategy_stored():
    r = _run(strategy="nanoparticle")
    assert r.strategy == "nanoparticle"


def test_notes_is_string():
    r = _run()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        _run(strategy="liposome")


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="parent_cl_l_per_h"):
        _run(parent_cl_l_per_h=0.0)


def test_negative_cl_raises():
    with pytest.raises(ValueError, match="parent_cl_l_per_h"):
        _run(parent_cl_l_per_h=-1.0)


def test_zero_vd_raises():
    with pytest.raises(ValueError, match="parent_vd_l"):
        _run(parent_vd_l=0.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="parent_dose_mg"):
        _run(parent_dose_mg=0.0)


def test_all_strategies_valid():
    for s in ("PEGylation", "albumin_fusion", "FcRn_engineering", "nanoparticle", "prodrug_depot"):
        r = predict_half_life_extension("Drug", s, 2.0, 10.0, 100.0)
        assert r.modified_t_half_h > 0.0
