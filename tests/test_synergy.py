"""Tests for drug combination synergy analysis (Phase 66)."""
from __future__ import annotations

import math
import pytest

from omega_pbpk.clinical.synergy import (
    SynergyResult,
    dose_response,
    compute_bliss_independence,
    compute_loewe_ci,
    assess_combination_synergy,
    synergy_matrix,
)


class TestDoseResponse:
    def test_at_ic50_effect_is_half(self):
        assert abs(dose_response(10.0, ic50=10.0) - 0.5) < 1e-6

    def test_zero_dose_zero_effect(self):
        assert dose_response(0.0, ic50=10.0) == 0.0

    def test_high_dose_approaches_1(self):
        assert dose_response(1000.0, ic50=1.0) > 0.99

    def test_hill_n_steepens_curve(self):
        e_steep = dose_response(10.0, ic50=10.0, hill_n=2.0)
        e_shallow = dose_response(10.0, ic50=10.0, hill_n=0.5)
        # At IC50, both give 0.5; test above IC50 for steeper curve
        e_high_steep = dose_response(20.0, ic50=10.0, hill_n=2.0)
        e_high_shallow = dose_response(20.0, ic50=10.0, hill_n=0.5)
        assert e_high_steep > e_high_shallow


class TestBlissIndependence:
    def test_no_effect_drugs_give_zero(self):
        assert compute_bliss_independence(0.0, 0.0) == 0.0

    def test_bliss_formula(self):
        result = compute_bliss_independence(0.4, 0.3)
        expected = 0.4 + 0.3 - 0.4 * 0.3
        assert abs(result - expected) < 1e-9

    def test_bliss_bounded(self):
        result = compute_bliss_independence(0.9, 0.9)
        assert result <= 1.0


class TestLoeweCI:
    def test_returns_float(self):
        result = compute_loewe_ci(1.0, 1.0, ic50_A=1.0, ic50_B=1.0, effect=0.5)
        assert isinstance(result, float)

    def test_zero_effect_returns_nan(self):
        result = compute_loewe_ci(1.0, 1.0, ic50_A=1.0, ic50_B=1.0, effect=0.0)
        assert math.isnan(result)

    def test_one_effect_returns_nan(self):
        result = compute_loewe_ci(1.0, 1.0, ic50_A=1.0, ic50_B=1.0, effect=1.0)
        assert math.isnan(result)

    def test_positive_ci_for_valid_inputs(self):
        result = compute_loewe_ci(1.0, 1.0, ic50_A=2.0, ic50_B=2.0, effect=0.5)
        if not math.isnan(result):
            assert result > 0


class TestAssessSynergy:
    def test_returns_result_type(self):
        r = assess_combination_synergy("A", "B", dose_A=1.0, dose_B=1.0, ic50_A=2.0, ic50_B=2.0)
        assert isinstance(r, SynergyResult)

    def test_drug_names_stored(self):
        r = assess_combination_synergy("DrugX", "DrugY", 1.0, 1.0, 2.0, 2.0)
        assert r.drug_A == "DrugX"
        assert r.drug_B == "DrugY"

    def test_effect_A_alone_positive(self):
        r = assess_combination_synergy("A", "B", 1.0, 1.0, 2.0, 2.0)
        assert r.effect_A_alone > 0.0

    def test_bliss_delta_finite(self):
        r = assess_combination_synergy("A", "B", 1.0, 1.0, 2.0, 2.0)
        assert math.isfinite(r.bliss_delta)

    def test_observed_effect_used(self):
        r = assess_combination_synergy("A", "B", 1.0, 1.0, 2.0, 2.0, observed_effect=0.9)
        assert abs(r.combination_effect - 0.9) < 1e-9

    def test_bliss_expected_matches_formula(self):
        r = assess_combination_synergy("A", "B", 1.0, 1.0, 2.0, 2.0)
        expected = r.effect_A_alone + r.effect_B_alone - r.effect_A_alone * r.effect_B_alone
        assert abs(r.bliss_expected_effect - expected) < 1e-9

    def test_ci_class_is_string(self):
        r = assess_combination_synergy("A", "B", 1.0, 1.0, 2.0, 2.0)
        assert r.ci_classification in ("synergy", "additivity", "antagonism", "undefined")

    def test_bliss_class_is_string(self):
        r = assess_combination_synergy("A", "B", 1.0, 1.0, 2.0, 2.0)
        assert r.bliss_classification in ("synergy", "additivity", "antagonism")


class TestSynergyMatrix:
    def test_returns_2d_list(self):
        result = synergy_matrix("A", "B", [1.0, 2.0], [1.0, 2.0], ic50_A=5.0, ic50_B=5.0)
        assert len(result) == 2
        assert len(result[0]) == 2

    def test_all_results_are_synergy_result(self):
        result = synergy_matrix("A", "B", [1.0, 2.0], [1.0, 2.0], ic50_A=5.0, ic50_B=5.0)
        for row in result:
            for cell in row:
                assert isinstance(cell, SynergyResult)
