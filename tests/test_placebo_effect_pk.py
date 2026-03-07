"""Tests for Phase 811: placebo-controlled PK/PD model."""

import math

import pytest

from omega_pbpk.core.placebo_effect_pk import (
    PlaceboEffectResult,
    calculate_placebo_adjusted_effect,
)


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        drug_conc_mg_L=1.0,
    )
    defaults.update(kwargs)
    return calculate_placebo_adjusted_effect(**defaults)


# --- Return type and frozen ---


def test_returns_placebo_effect_result():
    r = make_result()
    assert isinstance(r, PlaceboEffectResult)


def test_result_is_frozen():
    r = make_result()
    with pytest.raises((AttributeError, TypeError)):
        r.drug_name = "other"  # type: ignore[misc]


def test_drug_name_preserved():
    r = make_result(drug_name="Aspirin")
    assert r.drug_name == "Aspirin"


# --- total_observed_effect in [0, 100] ---


def test_total_observed_in_range_normal():
    r = make_result(drug_conc_mg_L=2.0, placebo_effect_pct=20.0, nocebo_effect_pct=5.0)
    assert 0.0 <= r.total_observed_effect_pct <= 100.0


def test_total_observed_capped_at_100():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=100.0,
        emax_pct=100.0,
        ec50_mg_L=0.01,
        placebo_effect_pct=50.0,
        nocebo_effect_pct=0.0,
    )
    assert r.total_observed_effect_pct <= 100.0


def test_total_observed_not_negative():
    # Very large nocebo
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=0.0,
        placebo_effect_pct=5.0,
        nocebo_effect_pct=100.0,
    )
    assert r.total_observed_effect_pct >= 0.0


# --- Zero drug concentration → drug_effect = 0 ---


def test_zero_conc_drug_effect_zero():
    r = make_result(drug_conc_mg_L=0.0)
    assert r.drug_effect_pct == 0.0


def test_zero_conc_total_reflects_placebo_nocebo():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=0.0,
        placebo_effect_pct=25.0,
        nocebo_effect_pct=5.0,
    )
    assert abs(r.total_observed_effect_pct - 20.0) < 1e-9


# --- High emax → net_benefit > 0 ---


def test_high_emax_positive_net_benefit():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=5.0,
        emax_pct=80.0,
        ec50_mg_L=1.0,
        placebo_effect_pct=20.0,
    )
    assert r.net_benefit_pct > 0.0


def test_drug_effect_at_ec50_is_half_emax():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=1.0,
        emax_pct=80.0,
        ec50_mg_L=1.0,
        hill_coef=1.0,
        placebo_effect_pct=0.0,
        nocebo_effect_pct=0.0,
    )
    assert abs(r.drug_effect_pct - 40.0) < 1e-9


# --- NNT formula check ---


def test_nnt_formula_positive_net_benefit():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=2.0,
        emax_pct=80.0,
        ec50_mg_L=1.0,
        hill_coef=1.0,
        placebo_effect_pct=20.0,
        nocebo_effect_pct=0.0,
    )
    expected_drug_effect = 80.0 * (2.0**1.0) / (1.0**1.0 + 2.0**1.0)
    expected_net_benefit = expected_drug_effect - 20.0
    expected_nnt = 100.0 / expected_net_benefit
    assert abs(r.nnt - expected_nnt) < 1e-6


def test_nnt_infinite_when_no_net_benefit():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=0.0,
        placebo_effect_pct=30.0,
        nocebo_effect_pct=0.0,
    )
    assert math.isinf(r.nnt)


# --- therapeutic_significance logic ---


def test_therapeutic_significance_clinically_meaningful():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=100.0,
        emax_pct=80.0,
        ec50_mg_L=1.0,
        placebo_effect_pct=5.0,
        nocebo_effect_pct=0.0,
        therapeutic_threshold_pct=30.0,
    )
    assert r.therapeutic_significance == "clinically_meaningful"


def test_therapeutic_significance_marginal():
    # Use values that give net_effect > 0 but < threshold
    r2 = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=1.0,
        emax_pct=30.0,
        ec50_mg_L=1.0,
        hill_coef=1.0,
        placebo_effect_pct=10.0,
        nocebo_effect_pct=0.0,
        therapeutic_threshold_pct=30.0,
    )
    # drug_effect = 15%, net = 5%
    assert r2.therapeutic_significance == "marginal"


def test_therapeutic_significance_not_significant():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=0.0,
        placebo_effect_pct=30.0,
        nocebo_effect_pct=0.0,
        therapeutic_threshold_pct=30.0,
    )
    assert r.therapeutic_significance == "not_significant"


# --- Validation ---


def test_negative_conc_raises():
    with pytest.raises(ValueError):
        calculate_placebo_adjusted_effect("Drug", drug_conc_mg_L=-1.0)


def test_zero_emax_raises():
    with pytest.raises(ValueError):
        calculate_placebo_adjusted_effect("Drug", drug_conc_mg_L=1.0, emax_pct=0.0)


def test_negative_emax_raises():
    with pytest.raises(ValueError):
        calculate_placebo_adjusted_effect("Drug", drug_conc_mg_L=1.0, emax_pct=-10.0)


# --- placebo_responder_rate ---


def test_placebo_responder_rate():
    r = calculate_placebo_adjusted_effect(
        drug_name="Drug",
        drug_conc_mg_L=1.0,
        placebo_effect_pct=30.0,
    )
    assert abs(r.placebo_responder_rate - 0.30) < 1e-9


# --- notes is a string ---


def test_notes_is_string():
    r = make_result()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0
