"""Tests for drug_synergy_matrix — Phase 499."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.drug_synergy_matrix import (
    SynergyMatrixResult,
    bliss_independence,
    bliss_synergy_score,
    build_synergy_matrix,
    classify_synergy,
    loewe_ci,
)

# ---------------------------------------------------------------------------
# bliss_independence
# ---------------------------------------------------------------------------


class TestBlissIndependence:
    def test_zero_effects(self):
        assert bliss_independence(0.0, 0.0) == pytest.approx(0.0)

    def test_unit_effects(self):
        # E_A + E_B - E_A*E_B = 1 + 1 - 1 = 1
        assert bliss_independence(1.0, 1.0) == pytest.approx(1.0)

    def test_formula(self):
        ea, eb = 0.4, 0.5
        expected = ea + eb - ea * eb
        assert bliss_independence(ea, eb) == pytest.approx(expected)

    def test_symmetry(self):
        assert bliss_independence(0.3, 0.7) == pytest.approx(bliss_independence(0.7, 0.3))

    def test_single_zero(self):
        assert bliss_independence(0.0, 0.5) == pytest.approx(0.5)
        assert bliss_independence(0.5, 0.0) == pytest.approx(0.5)

    def test_result_bounded_above_by_1(self):
        result = bliss_independence(0.9, 0.9)
        assert result <= 1.0

    def test_invalid_effect_a_negative(self):
        with pytest.raises(ValueError, match="effect_a"):
            bliss_independence(-0.1, 0.5)

    def test_invalid_effect_a_over_1(self):
        with pytest.raises(ValueError, match="effect_a"):
            bliss_independence(1.1, 0.5)

    def test_invalid_effect_b_negative(self):
        with pytest.raises(ValueError, match="effect_b"):
            bliss_independence(0.5, -0.1)

    def test_invalid_effect_b_over_1(self):
        with pytest.raises(ValueError, match="effect_b"):
            bliss_independence(0.5, 1.1)


# ---------------------------------------------------------------------------
# bliss_synergy_score
# ---------------------------------------------------------------------------


class TestBlissSynergyScore:
    def test_additive_gives_zero(self):
        ea, eb = 0.4, 0.5
        expected = bliss_independence(ea, eb)
        # When observed == expected, score == 0
        assert bliss_synergy_score(ea, eb, expected) == pytest.approx(0.0)

    def test_synergy_positive(self):
        ea, eb = 0.3, 0.3
        expected = bliss_independence(ea, eb)
        combo = min(expected + 0.1, 1.0)  # observed > expected
        assert bliss_synergy_score(ea, eb, combo) > 0.0

    def test_antagonism_negative(self):
        ea, eb = 0.6, 0.6
        expected = bliss_independence(ea, eb)
        combo = max(expected - 0.1, 0.0)
        assert bliss_synergy_score(ea, eb, combo) < 0.0

    def test_invalid_combo_over_1(self):
        with pytest.raises(ValueError, match="effect_combo"):
            bliss_synergy_score(0.5, 0.5, 1.2)

    def test_invalid_combo_negative(self):
        with pytest.raises(ValueError, match="effect_combo"):
            bliss_synergy_score(0.5, 0.5, -0.1)

    def test_invalid_effect_a(self):
        with pytest.raises(ValueError, match="effect_a"):
            bliss_synergy_score(1.5, 0.5, 0.8)

    def test_score_magnitude(self):
        # Large synergy
        score = bliss_synergy_score(0.5, 0.5, 0.95)
        # expected = 0.75, so score = 0.20
        assert score == pytest.approx(0.95 - 0.75, rel=1e-6)


# ---------------------------------------------------------------------------
# loewe_ci
# ---------------------------------------------------------------------------


class TestLoweCi:
    def test_identical_drug_additive(self):
        # When drug A == drug B (same EC50, equal dose split), CI should be 1.0
        ec50 = 10.0
        # At EC50, Hill effect = 0.5; each drug contributes half the isobologram dose
        # D_x that gives effect 0.5 with hill_n=1: D_x = EC50 * (0.5/0.5) = EC50
        # dose_a = dose_b = EC50/2 → CI = (EC50/2)/EC50 + (EC50/2)/EC50 = 1.0
        dose_a = ec50 / 2.0
        dose_b = ec50 / 2.0
        ci_val = loewe_ci(dose_a, dose_b, ec50, ec50, effect=0.5)
        assert ci_val == pytest.approx(1.0)

    def test_ci_less_than_1_synergy(self):
        # Lower combined doses achieving same effect → CI < 1
        # Use doses that together achieve more than additive
        ec50_a, ec50_b = 10.0, 10.0
        # If actual doses are less than isobologram predicts → CI < 1
        dose_a = 3.0
        dose_b = 3.0
        effect = 0.4
        # D_x = EC50 * (0.4/0.6) = 6.667; CI = 3/6.667 + 3/6.667 = 0.9
        ci_val = loewe_ci(dose_a, dose_b, ec50_a, ec50_b, effect=effect)
        assert ci_val < 1.0

    def test_ci_greater_than_1_antagonism(self):
        ec50_a, ec50_b = 10.0, 10.0
        dose_a = 8.0
        dose_b = 8.0
        effect = 0.4
        ci_val = loewe_ci(dose_a, dose_b, ec50_a, ec50_b, effect=effect)
        assert ci_val > 1.0

    def test_invalid_effect_boundary(self):
        with pytest.raises(ValueError, match="effect"):
            loewe_ci(1.0, 1.0, 10.0, 10.0, effect=0.0)
        with pytest.raises(ValueError, match="effect"):
            loewe_ci(1.0, 1.0, 10.0, 10.0, effect=1.0)

    def test_invalid_ec50(self):
        with pytest.raises(ValueError, match="ec50_a"):
            loewe_ci(1.0, 1.0, 0.0, 10.0, effect=0.5)
        with pytest.raises(ValueError, match="ec50_b"):
            loewe_ci(1.0, 1.0, 10.0, -1.0, effect=0.5)

    def test_hill_n_effect(self):
        # Different hill_n should yield different CI for asymmetric inputs
        # Use non-50% effect so the exponent 1/hill_n changes D_x differently
        ci_n1 = loewe_ci(5.0, 3.0, 10.0, 8.0, effect=0.3, hill_n=1.0)
        ci_n2 = loewe_ci(5.0, 3.0, 10.0, 8.0, effect=0.3, hill_n=2.0)
        assert abs(ci_n1 - ci_n2) > 1e-6


# ---------------------------------------------------------------------------
# classify_synergy
# ---------------------------------------------------------------------------


class TestClassifySynergy:
    def test_strong_synergy(self):
        assert classify_synergy(-0.3) == "strong_synergy"
        assert classify_synergy(-0.2) == "strong_synergy"

    def test_synergy(self):
        assert classify_synergy(-0.15) == "synergy"
        assert classify_synergy(-0.05) == "synergy"

    def test_additive(self):
        assert classify_synergy(0.0) == "additive"
        assert classify_synergy(0.04) == "additive"
        assert classify_synergy(-0.04) == "additive"

    def test_antagonism(self):
        assert classify_synergy(0.1) == "antagonism"
        assert classify_synergy(0.2) == "antagonism"

    def test_strong_antagonism(self):
        assert classify_synergy(0.25) == "strong_antagonism"
        assert classify_synergy(1.0) == "strong_antagonism"


# ---------------------------------------------------------------------------
# build_synergy_matrix
# ---------------------------------------------------------------------------


class TestBuildSynergyMatrix:
    def setup_method(self):
        self.doses_a = [1.0, 5.0, 10.0]
        self.doses_b = [2.0, 4.0, 8.0, 16.0]
        self.ec50_a = 10.0
        self.ec50_b = 8.0

    def test_output_type(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        assert isinstance(result, SynergyMatrixResult)

    def test_bliss_matrix_shape(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        assert len(result.bliss_matrix) == len(self.doses_a)
        for row in result.bliss_matrix:
            assert len(row) == len(self.doses_b)

    def test_loewe_ci_matrix_shape(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        assert len(result.loewe_ci_matrix) == len(self.doses_a)
        for row in result.loewe_ci_matrix:
            assert len(row) == len(self.doses_b)

    def test_doses_stored_correctly(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        assert result.doses_a == self.doses_a
        assert result.doses_b == self.doses_b

    def test_drug_names(self):
        result = build_synergy_matrix(
            self.doses_a,
            self.doses_b,
            self.ec50_a,
            self.ec50_b,
            drug_a="DrugX",
            drug_b="DrugY",
        )
        assert result.drug_a == "DrugX"
        assert result.drug_b == "DrugY"

    def test_mean_bliss_score_type(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        assert isinstance(result.mean_bliss_score, float)

    def test_synergy_classification_valid(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        valid = {"strong_synergy", "synergy", "additive", "antagonism", "strong_antagonism"}
        assert result.synergy_classification in valid

    def test_max_synergy_doses_within_grid(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        da, db = result.max_synergy_doses
        assert da in self.doses_a
        assert db in self.doses_b

    def test_loewe_ci_positive(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        for row in result.loewe_ci_matrix:
            for ci_val in row:
                assert ci_val > 0.0

    def test_empty_doses_a_raises(self):
        with pytest.raises(ValueError, match="doses_a"):
            build_synergy_matrix([], self.doses_b, self.ec50_a, self.ec50_b)

    def test_empty_doses_b_raises(self):
        with pytest.raises(ValueError, match="doses_b"):
            build_synergy_matrix(self.doses_a, [], self.ec50_a, self.ec50_b)

    def test_invalid_ec50_a(self):
        with pytest.raises(ValueError, match="ec50_a"):
            build_synergy_matrix(self.doses_a, self.doses_b, 0.0, self.ec50_b)

    def test_notes_present(self):
        result = build_synergy_matrix(self.doses_a, self.doses_b, self.ec50_a, self.ec50_b)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_single_dose_each(self):
        result = build_synergy_matrix([5.0], [5.0], 10.0, 10.0)
        assert len(result.bliss_matrix) == 1
        assert len(result.bliss_matrix[0]) == 1
        assert len(result.loewe_ci_matrix) == 1
        assert len(result.loewe_ci_matrix[0]) == 1
