"""Tests for Phase 674 — Dose Frequency Optimizer."""

import pytest

from omega_pbpk.clinical.dose_frequency_optimizer import (
    FrequencyOptimizationResult,
    optimize_dosing_frequency,
    quick_frequency_recommendation,
)


def _default_optimize(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        total_daily_dose_mg=200.0,
        t_half_h=8.0,
        mec_mg_L=0.5,
        mtc_mg_L=5.0,
        vd_L=50.0,
        f_oral=0.8,
        ka_per_h=1.0,
    )
    defaults.update(kwargs)
    return optimize_dosing_frequency(**defaults)


# ── Basic structure ──────────────────────────────────────────────────────────


def test_returns_list_of_results():
    results = _default_optimize()
    assert isinstance(results, list)
    assert all(isinstance(r, FrequencyOptimizationResult) for r in results)


def test_default_returns_four_results():
    results = _default_optimize()
    assert len(results) == 4


def test_exactly_one_recommended():
    results = _default_optimize()
    assert sum(1 for r in results if r.recommended) == 1


def test_all_composite_scores_in_range():
    results = _default_optimize()
    for r in results:
        assert 0.0 <= r.composite_score <= 100.0


def test_all_tir_in_range():
    results = _default_optimize()
    for r in results:
        assert 0.0 <= r.time_in_therapeutic_pct <= 100.0


def test_all_fluctuation_nonnegative():
    results = _default_optimize()
    for r in results:
        assert r.fluctuation_pct >= 0.0


def test_sorted_by_composite_score_descending():
    results = _default_optimize()
    scores = [r.composite_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bid_or_tid_recommended_for_t_half_8h():
    results = _default_optimize(t_half_h=8.0)
    rec = next(r for r in results if r.recommended)
    assert rec.frequency_per_day in {2, 3}


def test_qd_recommended_for_long_t_half():
    # Very long t_half → QD should dominate via accumulation and convenience
    results = _default_optimize(
        t_half_h=24.0,
        mec_mg_L=0.1,
        mtc_mg_L=10.0,
    )
    rec = next(r for r in results if r.recommended)
    assert rec.frequency_per_day == 1


def test_higher_frequency_lower_fluctuation():
    results = _default_optimize()
    # Sort by frequency
    freq_results = sorted(results, key=lambda r: r.frequency_per_day)
    # Fluctuation should generally decrease as frequency increases
    flucts = [r.fluctuation_pct for r in freq_results]
    # At least the highest frequency should have lower fluctuation than the lowest
    assert flucts[-1] < flucts[0]


def test_accumulation_ratio_gte_one():
    results = _default_optimize()
    for r in results:
        assert r.accumulation_ratio >= 1.0


def test_convenience_scores():
    results = _default_optimize()
    freq_map = {r.frequency_per_day: r.patient_convenience_score for r in results}
    assert freq_map[1] == pytest.approx(100.0)
    assert freq_map[2] == pytest.approx(75.0)
    assert freq_map[3] == pytest.approx(50.0)
    assert freq_map[4] == pytest.approx(25.0)


def test_css_max_gt_css_min():
    results = _default_optimize()
    for r in results:
        assert r.css_max_mg_L > r.css_min_mg_L


def test_notes_nonempty():
    results = _default_optimize()
    for r in results:
        assert len(r.notes) > 0


# ── quick_frequency_recommendation ──────────────────────────────────────────


def test_quick_returns_valid_recommendation():
    rec = quick_frequency_recommendation("Drug", t_half_h=10.0, mec_mg_L=0.5, mtc_mg_L=5.0)
    assert rec in {"QD", "BID", "TID", "QID"}


def test_quick_long_t_half_qd():
    rec = quick_frequency_recommendation("Drug", t_half_h=20.0, mec_mg_L=0.5, mtc_mg_L=5.0)
    assert rec == "QD"


def test_quick_short_t_half_qid():
    rec = quick_frequency_recommendation("Drug", t_half_h=2.0, mec_mg_L=0.5, mtc_mg_L=5.0)
    assert rec == "QID"


# ── Validation errors ────────────────────────────────────────────────────────


def test_validation_total_daily_dose_zero():
    with pytest.raises(ValueError):
        optimize_dosing_frequency("X", total_daily_dose_mg=0.0, t_half_h=8.0)


def test_validation_t_half_zero():
    with pytest.raises(ValueError):
        optimize_dosing_frequency("X", total_daily_dose_mg=100.0, t_half_h=0.0)


def test_validation_mec_gte_mtc():
    with pytest.raises(ValueError):
        optimize_dosing_frequency(
            "X",
            total_daily_dose_mg=100.0,
            t_half_h=8.0,
            mec_mg_L=5.0,
            mtc_mg_L=2.0,
        )
