"""Tests for Phase 616: Drug Plasma Stability Prediction.

Tests cover return type, frozen dataclass, t_half formula, stability classes,
ester/amide contributions, lactone effect, fraction remaining, screening sort,
and validation.
"""

import math

import pytest

from omega_pbpk.prediction.plasma_stability_p616 import (
    PlasmaStabilityResult,
    predict_plasma_stability,
    stability_screening,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stable_drug():
    """Drug with no reactive groups — should be stable."""
    return predict_plasma_stability(
        drug_name="StableDrug",
        mw_Da=300.0,
        logP=2.0,
        n_ester_groups=0,
        n_amide_groups=0,
        n_aromatic_rings=0,
    )


@pytest.fixture
def ester_drug():
    """Drug with 2 ester groups — should degrade faster."""
    return predict_plasma_stability(
        drug_name="EsterDrug",
        mw_Da=350.0,
        logP=2.0,
        n_ester_groups=2,
    )


@pytest.fixture
def lactone_drug():
    """Drug with lactone — should degrade faster than simple ester."""
    return predict_plasma_stability(
        drug_name="LactoneDrug",
        mw_Da=250.0,
        logP=1.5,
        lactone=True,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_plasma_stability_result(self, stable_drug):
        assert isinstance(stable_drug, PlasmaStabilityResult)

    def test_result_is_frozen_dataclass(self, stable_drug):
        with pytest.raises((AttributeError, TypeError)):
            stable_drug.drug_name = "Modified"

    def test_drug_name_preserved(self, stable_drug):
        assert stable_drug.drug_name == "StableDrug"

    def test_all_float_fields_are_float(self, stable_drug):
        assert isinstance(stable_drug.k_deg_per_h, float)
        assert isinstance(stable_drug.t_half_plasma_h, float)
        assert isinstance(stable_drug.fraction_remaining_1h, float)
        assert isinstance(stable_drug.fraction_remaining_2h, float)
        assert isinstance(stable_drug.fraction_remaining_4h, float)

    def test_bool_fields_are_bool(self, stable_drug):
        assert isinstance(stable_drug.ester_content, bool)
        assert isinstance(stable_drug.amide_content, bool)

    def test_string_fields_are_str(self, stable_drug):
        assert isinstance(stable_drug.smiles_features, str)
        assert isinstance(stable_drug.stability_class, str)
        assert isinstance(stable_drug.degradation_pathways, str)
        assert isinstance(stable_drug.adjustments, str)
        assert isinstance(stable_drug.notes, str)


# ---------------------------------------------------------------------------
# t_half formula tests
# ---------------------------------------------------------------------------
class TestTHalfFormula:
    def test_t_half_from_k_deg(self, stable_drug):
        expected_t_half = math.log(2.0) / stable_drug.k_deg_per_h
        assert stable_drug.t_half_plasma_h == pytest.approx(expected_t_half, rel=1e-6)

    def test_fraction_remaining_1h_formula(self, stable_drug):
        expected = math.exp(-stable_drug.k_deg_per_h * 1.0)
        assert stable_drug.fraction_remaining_1h == pytest.approx(expected, rel=1e-6)

    def test_fraction_remaining_2h_formula(self, stable_drug):
        expected = math.exp(-stable_drug.k_deg_per_h * 2.0)
        assert stable_drug.fraction_remaining_2h == pytest.approx(expected, rel=1e-6)

    def test_fraction_remaining_4h_formula(self, stable_drug):
        expected = math.exp(-stable_drug.k_deg_per_h * 4.0)
        assert stable_drug.fraction_remaining_4h == pytest.approx(expected, rel=1e-6)

    def test_fraction_remaining_ordering(self, ester_drug):
        # More time = less remaining
        assert ester_drug.fraction_remaining_1h > ester_drug.fraction_remaining_2h
        assert ester_drug.fraction_remaining_2h > ester_drug.fraction_remaining_4h

    def test_fraction_remaining_bounded(self, stable_drug):
        assert 0.0 < stable_drug.fraction_remaining_1h <= 1.0
        assert 0.0 < stable_drug.fraction_remaining_4h <= 1.0


# ---------------------------------------------------------------------------
# Stability class tests
# ---------------------------------------------------------------------------
class TestStabilityClass:
    def test_stable_drug_class(self, stable_drug):
        # k_base = 0.05, t_half = ln(2)/0.05 ≈ 13.86h > 6h
        assert stable_drug.stability_class == "stable"

    def test_unstable_drug_class(self):
        # Many esters → k_deg high → t_half < 0.5h
        r = predict_plasma_stability(drug_name="Unstable", mw_Da=300.0, logP=2.0, n_ester_groups=10)
        assert r.stability_class == "unstable"

    def test_labile_class(self):
        # 2 esters: k_deg = 0.05 + 2*0.3 = 0.65 → t_half = ln(2)/0.65 ≈ 1.07h (0.5-2h = labile)
        r = predict_plasma_stability(drug_name="Labile", mw_Da=300.0, logP=2.0, n_ester_groups=2)
        assert r.stability_class == "labile"

    def test_moderate_class(self):
        # 1 ester: k_deg = 0.05 + 0.3 = 0.35 → t_half = ln(2)/0.35 ≈ 1.98h → labile
        # Try 0.5 esters not possible; use lactone alone: k_deg = 0.05+0.5=0.55 → t_half ≈ 1.26h
        # For moderate (2-6h), need k between ln(2)/6 ≈ 0.116 and ln(2)/2 ≈ 0.347
        # Use amides only: 1 amide: k_deg = 0.05+0.02=0.07 → t_half ≈ 9.9h (stable)
        # Use 1 ester: k_deg=0.35, t_half≈1.98h → just below 2 = labile
        # For moderate, need 0.116 < k < 0.347. Use 2 aromatic rings with logP=4:
        # k_ox = 2*0.01*2 = 0.04; k_deg = 0.05+0.04=0.09 → stable
        # Use 1 aromatic with logP=10: k_ox=0.01*8=0.08; k_deg=0.05+0.08=0.13 → t_half=5.33h (moderate)
        r = predict_plasma_stability(
            drug_name="Moderate", mw_Da=400.0, logP=10.0, n_aromatic_rings=1
        )
        assert r.stability_class == "moderate"

    def test_valid_stability_classes(self):
        valid_classes = {"unstable", "labile", "moderate", "stable"}
        r = predict_plasma_stability("X", mw_Da=300.0, logP=1.0)
        assert r.stability_class in valid_classes


# ---------------------------------------------------------------------------
# Ester contribution tests
# ---------------------------------------------------------------------------
class TestEsterContribution:
    def test_ester_increases_k_deg(self, stable_drug, ester_drug):
        assert ester_drug.k_deg_per_h > stable_drug.k_deg_per_h

    def test_ester_content_true(self, ester_drug):
        assert ester_drug.ester_content is True

    def test_no_ester_content_false(self, stable_drug):
        assert stable_drug.ester_content is False

    def test_ester_in_degradation_pathways(self, ester_drug):
        assert "ester_hydrolysis" in ester_drug.degradation_pathways

    def test_each_ester_adds_03(self):
        r1 = predict_plasma_stability("D1", mw_Da=300.0, logP=2.0, n_ester_groups=1)
        r2 = predict_plasma_stability("D2", mw_Da=300.0, logP=2.0, n_ester_groups=2)
        # Difference should be ~0.3/h
        assert r2.k_deg_per_h - r1.k_deg_per_h == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Amide contribution tests
# ---------------------------------------------------------------------------
class TestAmideContribution:
    def test_amide_content_true(self):
        r = predict_plasma_stability("D", mw_Da=300.0, logP=2.0, n_amide_groups=2)
        assert r.amide_content is True

    def test_no_amide_content_false(self, stable_drug):
        assert stable_drug.amide_content is False

    def test_amide_in_degradation_pathways(self):
        r = predict_plasma_stability("D", mw_Da=300.0, logP=2.0, n_amide_groups=2)
        assert "amide_hydrolysis" in r.degradation_pathways

    def test_amide_less_effect_than_ester(self):
        r_ester = predict_plasma_stability("E", mw_Da=300.0, logP=2.0, n_ester_groups=1)
        r_amide = predict_plasma_stability("A", mw_Da=300.0, logP=2.0, n_amide_groups=1)
        # Ester should increase k_deg more than amide
        assert r_ester.k_deg_per_h > r_amide.k_deg_per_h


# ---------------------------------------------------------------------------
# Lactone tests
# ---------------------------------------------------------------------------
class TestLactone:
    def test_lactone_increases_k_deg(self, stable_drug, lactone_drug):
        assert lactone_drug.k_deg_per_h > stable_drug.k_deg_per_h

    def test_lactone_ester_content_true(self, lactone_drug):
        assert lactone_drug.ester_content is True

    def test_lactone_pathway_listed(self, lactone_drug):
        assert "ester_hydrolysis" in lactone_drug.degradation_pathways

    def test_lactone_adds_to_ester(self):
        r_ester = predict_plasma_stability("E", mw_Da=300.0, logP=2.0, n_ester_groups=1)
        r_lactone_ester = predict_plasma_stability(
            "LE", mw_Da=300.0, logP=2.0, n_ester_groups=1, lactone=True
        )
        # Lactone + ester should be worse than ester alone
        assert r_lactone_ester.k_deg_per_h > r_ester.k_deg_per_h


# ---------------------------------------------------------------------------
# Degradation pathway "none" test
# ---------------------------------------------------------------------------
class TestDegradationPathways:
    def test_no_reactive_groups_gives_none_pathway(self, stable_drug):
        assert stable_drug.degradation_pathways == "none"

    def test_multiple_pathways_comma_separated(self):
        r = predict_plasma_stability(
            "Multi", mw_Da=400.0, logP=5.0, n_ester_groups=1, n_aromatic_rings=2
        )
        assert "ester_hydrolysis" in r.degradation_pathways
        assert "oxidation" in r.degradation_pathways


# ---------------------------------------------------------------------------
# smiles_features tests
# ---------------------------------------------------------------------------
class TestSmilesFeatures:
    def test_smiles_features_format(self, stable_drug):
        assert "esters:0" in stable_drug.smiles_features
        assert "amides:0" in stable_drug.smiles_features
        assert "aromatics:0" in stable_drug.smiles_features

    def test_smiles_features_reflects_input(self):
        r = predict_plasma_stability(
            "X", mw_Da=300.0, logP=2.0, n_ester_groups=2, n_amide_groups=1, n_aromatic_rings=3
        )
        assert "esters:2" in r.smiles_features
        assert "amides:1" in r.smiles_features
        assert "aromatics:3" in r.smiles_features


# ---------------------------------------------------------------------------
# k_deg clamping tests
# ---------------------------------------------------------------------------
class TestKDegClamping:
    def test_k_deg_clamped_to_max_5(self):
        # 20 esters: k_raw = 0.05 + 20*0.3 = 6.05 > 5.0
        r = predict_plasma_stability("Heavy", mw_Da=600.0, logP=2.0, n_ester_groups=20)
        assert r.k_deg_per_h == pytest.approx(5.0, abs=1e-9)

    def test_k_deg_clamped_to_min_001(self):
        # With no reactive groups: k_deg = k_base = 0.05 > 0.01, so no clamping here
        # But if somehow k was negative (not possible by design), it would clamp
        r = predict_plasma_stability("X", mw_Da=300.0, logP=-5.0)
        assert r.k_deg_per_h >= 0.01


# ---------------------------------------------------------------------------
# Stability screening tests
# ---------------------------------------------------------------------------
class TestStabilityScreening:
    def test_screening_returns_list(self):
        compounds = [
            {"drug_name": "A", "mw_Da": 300.0, "logP": 2.0},
            {"drug_name": "B", "mw_Da": 350.0, "logP": 2.0, "n_ester_groups": 2},
        ]
        results = stability_screening(compounds)
        assert isinstance(results, list)

    def test_screening_sorted_descending_t_half(self):
        compounds = [
            {"drug_name": "Unstable", "mw_Da": 300.0, "logP": 2.0, "n_ester_groups": 5},
            {"drug_name": "Stable", "mw_Da": 300.0, "logP": 2.0},
            {"drug_name": "Medium", "mw_Da": 300.0, "logP": 2.0, "n_ester_groups": 1},
        ]
        results = stability_screening(compounds)
        t_halfs = [r.t_half_plasma_h for r in results]
        assert t_halfs == sorted(t_halfs, reverse=True)

    def test_screening_returns_correct_count(self):
        compounds = [{"drug_name": f"Drug{i}", "mw_Da": 300.0, "logP": float(i)} for i in range(5)]
        results = stability_screening(compounds)
        assert len(results) == 5

    def test_screening_all_results_are_correct_type(self):
        compounds = [
            {"drug_name": "A", "mw_Da": 300.0, "logP": 2.0},
            {"drug_name": "B", "mw_Da": 350.0, "logP": 1.0, "n_amide_groups": 1},
        ]
        results = stability_screening(compounds)
        for r in results:
            assert isinstance(r, PlasmaStabilityResult)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
class TestValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_plasma_stability("X", mw_Da=0.0, logP=2.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_plasma_stability("X", mw_Da=-100.0, logP=2.0)

    def test_negative_ester_raises(self):
        with pytest.raises(ValueError, match="n_ester"):
            predict_plasma_stability("X", mw_Da=300.0, logP=2.0, n_ester_groups=-1)

    def test_negative_amide_raises(self):
        with pytest.raises(ValueError, match="n_amide"):
            predict_plasma_stability("X", mw_Da=300.0, logP=2.0, n_amide_groups=-1)

    def test_negative_aromatic_raises(self):
        with pytest.raises(ValueError, match="n_aromatic"):
            predict_plasma_stability("X", mw_Da=300.0, logP=2.0, n_aromatic_rings=-1)

    def test_valid_pka_accepted(self):
        # pka is optional and should not raise
        r = predict_plasma_stability("X", mw_Da=300.0, logP=2.0, pka=7.4)
        assert isinstance(r, PlasmaStabilityResult)
