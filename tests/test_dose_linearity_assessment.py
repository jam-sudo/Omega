"""Tests for Phase 694: dose linearity / dose proportionality assessment."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.dose_linearity_assessment import (
    DoseLinearityResult,
    assess_dose_proportionality,
    simulate_dose_proportionality_data,
)

# ---------------------------------------------------------------------------
# assess_dose_proportionality — return type and basic structure
# ---------------------------------------------------------------------------


def test_returns_dose_linearity_result():
    doses = [10.0, 30.0, 100.0]
    aucs = [10.0, 30.0, 100.0]  # perfect proportionality
    result = assess_dose_proportionality(doses, aucs)
    assert isinstance(result, DoseLinearityResult)


def test_metric_stored():
    doses = [10.0, 30.0, 100.0]
    aucs = [10.0, 30.0, 100.0]
    result = assess_dose_proportionality(doses, aucs, metric="Cmax")
    assert result.metric == "Cmax"


def test_n_doses_stored():
    doses = [10.0, 30.0, 100.0, 300.0]
    aucs = [10.0, 30.0, 100.0, 300.0]
    result = assess_dose_proportionality(doses, aucs)
    assert result.n_doses == 4


def test_dose_range_stored():
    doses = [10.0, 30.0, 100.0]
    aucs = [10.0, 30.0, 100.0]
    result = assess_dose_proportionality(doses, aucs)
    assert result.dose_range_min == pytest.approx(10.0)
    assert result.dose_range_max == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Power model correctness
# ---------------------------------------------------------------------------


def test_linear_pk_gives_beta_one():
    """Perfect proportionality: AUC = k * dose → beta = 1.0."""
    doses = [10.0, 30.0, 100.0, 300.0, 1000.0]
    aucs = [d * 2.5 for d in doses]  # exactly proportional
    result = assess_dose_proportionality(doses, aucs)
    assert result.power_law_beta == pytest.approx(1.0, abs=1e-6)


def test_linear_pk_class_dose_proportional():
    doses = [10.0, 30.0, 100.0, 300.0]
    aucs = [d * 3.0 for d in doses]
    result = assess_dose_proportionality(doses, aucs)
    assert result.proportionality_class == "dose_proportional"


def test_sub_proportional_classification():
    """AUC = a * dose^0.5 → beta = 0.5 → sub_proportional."""
    import math

    doses = [10.0, 30.0, 100.0, 300.0, 1000.0]
    aucs = [5.0 * math.sqrt(d) for d in doses]
    result = assess_dose_proportionality(doses, aucs)
    assert result.proportionality_class == "sub_proportional"
    assert result.power_law_beta < 0.8


def test_supra_proportional_classification():
    """AUC = a * dose^1.5 → beta = 1.5 → supra_proportional."""
    doses = [10.0, 30.0, 100.0, 300.0, 1000.0]
    aucs = [0.1 * (d**1.5) for d in doses]
    result = assess_dose_proportionality(doses, aucs)
    assert result.proportionality_class == "supra_proportional"
    assert result.power_law_beta > 1.25


def test_r_squared_in_range():
    doses = [10.0, 30.0, 100.0]
    aucs = [d * 2.0 for d in doses]
    result = assess_dose_proportionality(doses, aucs)
    assert 0.0 <= result.power_law_r_squared <= 1.0


def test_r_squared_perfect_fit():
    """Perfect power law fit should give R² = 1.0."""
    doses = [10.0, 30.0, 100.0, 300.0]
    aucs = [d * 2.5 for d in doses]
    result = assess_dose_proportionality(doses, aucs)
    assert result.power_law_r_squared == pytest.approx(1.0, abs=1e-8)


def test_notes_nonempty():
    doses = [10.0, 30.0, 100.0]
    aucs = [10.0, 30.0, 100.0]
    result = assess_dose_proportionality(doses, aucs)
    assert len(result.notes) > 0


def test_beta_interpretation_nonempty():
    doses = [10.0, 30.0, 100.0]
    aucs = [10.0, 30.0, 100.0]
    result = assess_dose_proportionality(doses, aucs)
    assert len(result.beta_interpretation) > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_fewer_than_3_raises():
    with pytest.raises(ValueError, match="3"):
        assess_dose_proportionality([10.0, 30.0], [10.0, 30.0])


def test_negative_auc_raises():
    with pytest.raises(ValueError):
        assess_dose_proportionality([10.0, 30.0, 100.0], [10.0, -5.0, 100.0])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        assess_dose_proportionality([10.0, 30.0, 100.0], [10.0, 30.0])


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        assess_dose_proportionality([0.0, 30.0, 100.0], [0.0, 30.0, 100.0])


# ---------------------------------------------------------------------------
# simulate_dose_proportionality_data
# ---------------------------------------------------------------------------


def test_simulate_returns_correct_length():
    doses = [10.0, 30.0, 100.0, 300.0]
    data = simulate_dose_proportionality_data(5.0, 50.0, 1.0, 0.8, doses)
    assert len(data) == 4


def test_simulate_keys_present():
    doses = [10.0, 100.0, 300.0]
    data = simulate_dose_proportionality_data(5.0, 50.0, 1.0, 0.8, doses)
    for item in data:
        assert "dose_mg" in item
        assert "auc" in item
        assert "cmax" in item


def test_simulate_dose_values_match():
    doses = [50.0, 150.0, 500.0]
    data = simulate_dose_proportionality_data(5.0, 50.0, 1.0, 0.8, doses)
    for i, item in enumerate(data):
        assert item["dose_mg"] == doses[i]


def test_simulate_auc_positive():
    doses = [10.0, 30.0, 100.0]
    data = simulate_dose_proportionality_data(5.0, 50.0, 1.0, 0.8, doses)
    for item in data:
        assert item["auc"] > 0.0


def test_simulate_cmax_positive():
    doses = [10.0, 30.0, 100.0]
    data = simulate_dose_proportionality_data(5.0, 50.0, 1.0, 0.8, doses)
    for item in data:
        assert item["cmax"] > 0.0


def test_simulate_linear_pk_proportional():
    """For linear 1-cpt model, AUC should be dose-proportional (beta ≈ 1)."""
    doses = [10.0, 30.0, 100.0, 300.0, 1000.0]
    data = simulate_dose_proportionality_data(
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_per_h=1.0,
        f_oral=0.8,
        doses_mg=doses,
        t_end_h=48.0,
        dt_h=0.05,
    )
    aucs = [d["auc"] for d in data]
    result = assess_dose_proportionality(doses, aucs)
    assert abs(result.power_law_beta - 1.0) < 0.2


def test_simulate_auc_increases_with_dose():
    doses = [10.0, 100.0, 1000.0]
    data = simulate_dose_proportionality_data(5.0, 50.0, 1.0, 0.8, doses)
    aucs = [d["auc"] for d in data]
    assert aucs[0] < aucs[1] < aucs[2]
