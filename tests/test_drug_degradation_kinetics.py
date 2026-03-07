"""Tests for Phase 828 — Drug Degradation Kinetics."""

import math

import pytest

from omega_pbpk.prediction.drug_degradation_kinetics import (
    DegradationResult,
    predict_degradation,
    screen_storage_conditions,
)

# ---------------------------------------------------------------------------
# Default kwargs
# ---------------------------------------------------------------------------

DEFAULT_KW = dict(
    drug_name="TestDrug",
    logp=2.0,
    pka=7.0,
    mw_da=300.0,
    pathway="hydrolysis",
    storage_temp_celsius=25.0,
    ph=7.0,
    relative_humidity_pct=60.0,
)


def _run(**overrides):
    kw = dict(DEFAULT_KW)
    kw.update(overrides)
    return predict_degradation(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_degradation_result(self):
        result = _run()
        assert isinstance(result, DegradationResult)

    def test_drug_name_preserved(self):
        result = _run(drug_name="Aspirin")
        assert result.drug_name == "Aspirin"

    def test_pathway_preserved(self):
        result = _run(pathway="oxidation")
        assert result.degradation_pathway == "oxidation"

    def test_notes_nonempty(self):
        result = _run()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_activation_energy_fixed(self):
        result = _run()
        assert result.activation_energy_kJ_per_mol == 80.0


# ---------------------------------------------------------------------------
# Mathematical relationships
# ---------------------------------------------------------------------------


class TestMathRelationships:
    def test_t_half_equals_ln2_over_k(self):
        result = _run()
        expected = math.log(2.0) / result.rate_constant_per_h
        assert abs(result.t_half_h - expected) < 1e-9

    def test_t90_equals_ln_10_9_over_k(self):
        result = _run()
        expected = math.log(10.0 / 9.0) / result.rate_constant_per_h
        assert abs(result.t90_h - expected) < 1e-9

    def test_shelf_life_formula_consistency(self):
        result = _run()
        # shelf_life = ln(1/0.95) / k / (24*30) months
        expected = math.log(1.0 / 0.95) / result.rate_constant_per_h / (24.0 * 30.0)
        assert abs(result.shelf_life_months - expected) < 1e-9

    def test_concentration_at_1yr_formula(self):
        result = _run()
        expected = math.exp(-result.rate_constant_per_h * 8760.0)
        assert abs(result.concentration_at_1yr - expected) < 1e-9

    def test_concentration_at_1yr_in_0_to_1(self):
        result = _run()
        assert 0.0 < result.concentration_at_1yr <= 1.0

    def test_rate_constant_positive(self):
        result = _run()
        assert result.rate_constant_per_h > 0.0


# ---------------------------------------------------------------------------
# Temperature effect
# ---------------------------------------------------------------------------


class TestTemperatureEffect:
    def test_higher_temp_shorter_t_half(self):
        r_cool = _run(storage_temp_celsius=5.0)
        r_warm = _run(storage_temp_celsius=40.0)
        assert r_warm.t_half_h < r_cool.t_half_h

    def test_higher_temp_shorter_shelf_life(self):
        r_cool = _run(storage_temp_celsius=4.0)
        r_warm = _run(storage_temp_celsius=40.0)
        assert r_warm.shelf_life_months < r_cool.shelf_life_months

    def test_higher_temp_faster_rate(self):
        r_cool = _run(storage_temp_celsius=5.0)
        r_warm = _run(storage_temp_celsius=60.0)
        assert r_warm.rate_constant_per_h > r_cool.rate_constant_per_h


# ---------------------------------------------------------------------------
# ICH Q1A compliance
# ---------------------------------------------------------------------------


class TestIchQ1aCompliance:
    def test_ich_compliant_when_shelf_life_ge_24_months(self):
        # Low temp, low humidity, neutral pH should give long shelf life
        result = _run(storage_temp_celsius=5.0, ph=7.0, relative_humidity_pct=20.0)
        if result.shelf_life_months >= 24.0:
            assert result.ich_q1a_compliant is True

    def test_not_ich_compliant_when_shelf_life_lt_24_months(self):
        # High temp, harsh conditions → short shelf life
        result = _run(storage_temp_celsius=60.0, ph=2.0, relative_humidity_pct=90.0)
        if result.shelf_life_months < 24.0:
            assert result.ich_q1a_compliant is False

    def test_ich_compliance_flag_matches_shelf_life(self):
        for temp in [5.0, 25.0, 40.0]:
            result = _run(storage_temp_celsius=temp)
            expected = result.shelf_life_months >= 24.0
            assert result.ich_q1a_compliant == expected


# ---------------------------------------------------------------------------
# Risk level
# ---------------------------------------------------------------------------


class TestRiskLevel:
    def test_risk_level_valid_values(self):
        for pathway in ("hydrolysis", "oxidation", "photolysis", "combined"):
            result = _run(pathway=pathway)
            assert result.risk_level in ("low", "moderate", "high")

    def test_high_risk_when_shelf_life_lt_6_months(self):
        # Very high temperature → rapid degradation
        result = _run(storage_temp_celsius=70.0, ph=2.0, relative_humidity_pct=90.0)
        if result.shelf_life_months < 6.0:
            assert result.risk_level == "high"

    def test_low_risk_when_shelf_life_ge_24_months(self):
        result = _run(storage_temp_celsius=5.0, ph=7.0, relative_humidity_pct=20.0)
        if result.shelf_life_months >= 24.0:
            assert result.risk_level == "low"


# ---------------------------------------------------------------------------
# Pathway: combined gives higher k than individual
# ---------------------------------------------------------------------------


class TestCombinedPathway:
    def test_combined_rate_higher_than_hydrolysis(self):
        r_hyd = _run(pathway="hydrolysis")
        r_comb = _run(pathway="combined")
        assert r_comb.rate_constant_per_h > r_hyd.rate_constant_per_h

    def test_combined_rate_higher_than_oxidation(self):
        r_ox = _run(pathway="oxidation")
        r_comb = _run(pathway="combined")
        assert r_comb.rate_constant_per_h > r_ox.rate_constant_per_h

    def test_combined_rate_higher_than_photolysis(self):
        r_photo = _run(pathway="photolysis")
        r_comb = _run(pathway="combined")
        assert r_comb.rate_constant_per_h > r_photo.rate_constant_per_h

    def test_combined_shelf_life_shorter_than_individual(self):
        r_hyd = _run(pathway="hydrolysis")
        r_comb = _run(pathway="combined")
        assert r_comb.shelf_life_months < r_hyd.shelf_life_months


# ---------------------------------------------------------------------------
# Degradation products
# ---------------------------------------------------------------------------


class TestDegradationProducts:
    def test_hydrolysis_products(self):
        result = _run(pathway="hydrolysis")
        assert "carboxylic acid" in result.degradation_products

    def test_oxidation_products(self):
        result = _run(pathway="oxidation")
        assert "N-oxide" in result.degradation_products

    def test_photolysis_products(self):
        result = _run(pathway="photolysis")
        assert "radical" in result.degradation_products

    def test_combined_products_contains_all(self):
        result = _run(pathway="combined")
        assert "carboxylic acid" in result.degradation_products
        assert "N-oxide" in result.degradation_products
        assert "radical" in result.degradation_products


# ---------------------------------------------------------------------------
# screen_storage_conditions
# ---------------------------------------------------------------------------


class TestScreenStorageConditions:
    def test_returns_list(self):
        conditions = [
            {"temp_celsius": 5.0, "ph": 7.0, "rh_pct": 30.0},
            {"temp_celsius": 25.0, "ph": 7.0, "rh_pct": 60.0},
            {"temp_celsius": 40.0, "ph": 7.0, "rh_pct": 75.0},
        ]
        results = screen_storage_conditions("Drug", 2.0, 7.0, 300.0, "hydrolysis", conditions)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_descending_by_shelf_life(self):
        conditions = [
            {"temp_celsius": 40.0, "ph": 7.0, "rh_pct": 75.0},
            {"temp_celsius": 5.0, "ph": 7.0, "rh_pct": 20.0},
            {"temp_celsius": 25.0, "ph": 7.0, "rh_pct": 60.0},
        ]
        results = screen_storage_conditions("Drug", 2.0, 7.0, 300.0, "hydrolysis", conditions)
        shelf_lives = [r.shelf_life_months for r in results]
        assert shelf_lives == sorted(shelf_lives, reverse=True)

    def test_all_results_are_degradation_result(self):
        conditions = [
            {"temp_celsius": 5.0, "ph": 7.0, "rh_pct": 30.0},
            {"temp_celsius": 25.0, "ph": 7.0, "rh_pct": 60.0},
        ]
        results = screen_storage_conditions("Drug", 2.0, 7.0, 300.0, "oxidation", conditions)
        assert all(isinstance(r, DegradationResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rh_above_100_raises(self):
        with pytest.raises(ValueError):
            _run(relative_humidity_pct=101.0)

    def test_rh_negative_raises(self):
        with pytest.raises(ValueError):
            _run(relative_humidity_pct=-5.0)

    def test_temp_above_80_raises(self):
        with pytest.raises(ValueError):
            _run(storage_temp_celsius=85.0)

    def test_temp_below_minus_80_raises(self):
        with pytest.raises(ValueError):
            _run(storage_temp_celsius=-90.0)

    def test_invalid_pathway_raises(self):
        with pytest.raises(ValueError):
            _run(pathway="unknown")
