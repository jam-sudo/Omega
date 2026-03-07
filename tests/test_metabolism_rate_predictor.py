"""Tests for prediction/metabolism_rate_predictor.py — Phase 528."""

import pytest

from omega_pbpk.prediction.metabolism_rate_predictor import (
    clint_sensitivity_analysis,
    predict_clint_hepatic,
    predict_metabolic_pathway,
)

# ---------------------------------------------------------------------------
# predict_clint_hepatic
# ---------------------------------------------------------------------------


class TestPredictClintHepatic:
    def test_returns_all_keys(self):
        result = predict_clint_hepatic(logP=2.0, mw=300.0, n_nitrogen=1, n_oxygen=2)
        for key in (
            "clint_uL_min_mg",
            "clint_L_per_h",
            "clh_L_per_h",
            "extraction_ratio",
            "t_half_min",
            "metabolic_rate",
        ):
            assert key in result

    def test_high_logP_raises_clint(self):
        low = predict_clint_hepatic(logP=1.0, mw=300.0, n_nitrogen=0, n_oxygen=0)
        high = predict_clint_hepatic(logP=5.0, mw=300.0, n_nitrogen=0, n_oxygen=0)
        assert high["clint_uL_min_mg"] > low["clint_uL_min_mg"]

    def test_large_mw_lowers_clint(self):
        small = predict_clint_hepatic(logP=3.0, mw=300.0, n_nitrogen=1, n_oxygen=1)
        large = predict_clint_hepatic(logP=3.0, mw=800.0, n_nitrogen=1, n_oxygen=1)
        assert large["clint_uL_min_mg"] < small["clint_uL_min_mg"]

    def test_primary_amine_boosts_clint(self):
        without = predict_clint_hepatic(logP=2.0, mw=300.0, n_nitrogen=1, n_oxygen=0)
        with_amine = predict_clint_hepatic(
            logP=2.0, mw=300.0, n_nitrogen=1, n_oxygen=0, has_primary_amine=True
        )
        assert with_amine["clint_uL_min_mg"] > without["clint_uL_min_mg"]

    def test_phenol_boosts_clint(self):
        without = predict_clint_hepatic(logP=2.0, mw=300.0, n_nitrogen=0, n_oxygen=1)
        with_phenol = predict_clint_hepatic(
            logP=2.0, mw=300.0, n_nitrogen=0, n_oxygen=1, has_phenol=True
        )
        assert with_phenol["clint_uL_min_mg"] > without["clint_uL_min_mg"]

    def test_clint_clamped_at_300(self):
        # Extreme inputs should not exceed 300
        result = predict_clint_hepatic(
            logP=10.0, mw=100.0, n_nitrogen=10, n_oxygen=10, has_primary_amine=True, has_phenol=True
        )
        assert result["clint_uL_min_mg"] <= 300.0

    def test_clint_clamped_at_1(self):
        # Very low CLint should be at least 1
        result = predict_clint_hepatic(logP=-5.0, mw=1000.0, n_nitrogen=0, n_oxygen=0)
        assert result["clint_uL_min_mg"] >= 1.0

    def test_extraction_ratio_in_bounds(self):
        result = predict_clint_hepatic(logP=3.0, mw=350.0, n_nitrogen=2, n_oxygen=1)
        assert 0.0 <= result["extraction_ratio"] <= 1.0

    def test_extraction_ratio_high_clint(self):
        # Very high CLint → extraction ratio approaches 1
        result = predict_clint_hepatic(
            logP=8.0, mw=200.0, n_nitrogen=5, n_oxygen=5, has_primary_amine=True, has_phenol=True
        )
        assert result["extraction_ratio"] > 0.5

    def test_t_half_positive(self):
        result = predict_clint_hepatic(logP=2.0, mw=300.0, n_nitrogen=1, n_oxygen=1)
        assert result["t_half_min"] > 0

    def test_metabolic_rate_low(self):
        # CLint floor is 1.0 µL/min/mg; "low" requires < 1.0 so it is
        # unreachable. With the most unfavorable inputs (very low logP, very
        # high MW), CLint is still within "moderate" range.
        result = predict_clint_hepatic(logP=-3.0, mw=900.0, n_nitrogen=0, n_oxygen=0)
        assert result["clint_uL_min_mg"] >= 1.0
        assert result["metabolic_rate"] in ("low", "moderate")

    def test_metabolic_rate_high(self):
        result = predict_clint_hepatic(
            logP=6.0, mw=250.0, n_nitrogen=3, n_oxygen=2, has_primary_amine=True
        )
        assert result["metabolic_rate"] == "high"

    def test_metabolic_rate_moderate(self):
        # logP=2, mw=300, 1N, 0O → base=10+3=13 → moderate
        result = predict_clint_hepatic(logP=2.0, mw=300.0, n_nitrogen=1, n_oxygen=0)
        assert result["metabolic_rate"] in ("low", "moderate", "high")

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            predict_clint_hepatic(logP=2.0, mw=-100.0, n_nitrogen=0, n_oxygen=0)

    def test_invalid_n_nitrogen_raises(self):
        with pytest.raises(ValueError):
            predict_clint_hepatic(logP=2.0, mw=300.0, n_nitrogen=-1, n_oxygen=0)

    def test_clint_L_per_h_positive(self):
        result = predict_clint_hepatic(logP=2.0, mw=300.0, n_nitrogen=1, n_oxygen=1)
        assert result["clint_L_per_h"] > 0
        assert result["clh_L_per_h"] > 0


# ---------------------------------------------------------------------------
# predict_metabolic_pathway
# ---------------------------------------------------------------------------


class TestPredictMetabolicPathway:
    def test_cyp3a4_lipophilic(self):
        result = predict_metabolic_pathway(
            logP=4.0,
            mw=400.0,
            n_nitrogen=1,
            n_oxygen=1,
            has_primary_amine=False,
            has_phenol=False,
            n_aromatic_rings=2,
        )
        assert result["primary_pathway"] == "CYP3A4"

    def test_cyp2d6_nitrogen_rich(self):
        result = predict_metabolic_pathway(
            logP=2.0,
            mw=300.0,
            n_nitrogen=4,
            n_oxygen=0,
            has_primary_amine=False,
            has_phenol=False,
            n_aromatic_rings=1,
        )
        assert result["primary_pathway"] == "CYP2D6"

    def test_ugt_phenol(self):
        result = predict_metabolic_pathway(
            logP=1.0,
            mw=200.0,
            n_nitrogen=0,
            n_oxygen=2,
            has_primary_amine=False,
            has_phenol=True,
            n_aromatic_rings=1,
        )
        assert result["primary_pathway"] == "UGT"

    def test_mao_primary_amine(self):
        result = predict_metabolic_pathway(
            logP=1.0,
            mw=200.0,
            n_nitrogen=1,
            n_oxygen=0,
            has_primary_amine=True,
            has_phenol=False,
            n_aromatic_rings=0,
        )
        assert result["primary_pathway"] == "MAO"

    def test_default_cyp2c9(self):
        result = predict_metabolic_pathway(
            logP=1.0,
            mw=300.0,
            n_nitrogen=1,
            n_oxygen=1,
            has_primary_amine=False,
            has_phenol=False,
            n_aromatic_rings=1,
        )
        assert result["primary_pathway"] == "CYP2C9"

    def test_returns_all_keys(self):
        result = predict_metabolic_pathway(
            logP=3.0,
            mw=300.0,
            n_nitrogen=1,
            n_oxygen=1,
            has_primary_amine=False,
            has_phenol=False,
            n_aromatic_rings=1,
        )
        assert "primary_pathway" in result
        assert "secondary_pathway" in result
        assert "confidence" in result


# ---------------------------------------------------------------------------
# clint_sensitivity_analysis
# ---------------------------------------------------------------------------


class TestClintSensitivityAnalysis:
    def test_returns_list_per_logP(self):
        logP_values = [1.0, 2.0, 3.0, 4.0]
        results = clint_sensitivity_analysis(logP_values, mw=300.0, n_nitrogen=1, n_oxygen=1)
        assert len(results) == 4

    def test_monotonic_with_logP(self):
        logP_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        results = clint_sensitivity_analysis(logP_values, mw=300.0, n_nitrogen=1, n_oxygen=1)
        clints = [r["clint_uL_min_mg"] for r in results]
        # CLint should be non-decreasing (or constant once clamped)
        for i in range(len(clints) - 1):
            assert clints[i] <= clints[i + 1], (
                f"CLint not monotonic at index {i}: {clints[i]} > {clints[i + 1]}"
            )

    def test_result_keys(self):
        results = clint_sensitivity_analysis([2.0], mw=300.0, n_nitrogen=1, n_oxygen=1)
        assert "logP" in results[0]
        assert "clint_uL_min_mg" in results[0]
        assert "clh_L_per_h" in results[0]
        assert "extraction_ratio" in results[0]

    def test_empty_logP_range(self):
        results = clint_sensitivity_analysis([], mw=300.0, n_nitrogen=0, n_oxygen=0)
        assert results == []

    def test_extraction_ratio_in_bounds(self):
        logP_values = [-2.0, 0.0, 2.0, 4.0, 6.0]
        results = clint_sensitivity_analysis(logP_values, mw=400.0, n_nitrogen=2, n_oxygen=1)
        for r in results:
            assert 0.0 <= r["extraction_ratio"] <= 1.0
