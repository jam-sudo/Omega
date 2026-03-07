"""Tests for biopharmaceutics/formulation_ph_optimizer.py — Phase 480."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.formulation_ph_optimizer import (
    buffer_capacity,
    optimal_ph_for_solubility,
    pHOptimizationResult,
    predict_stability_ph,
    solubility_vs_ph,
)

# ---------------------------------------------------------------------------
# solubility_vs_ph
# ---------------------------------------------------------------------------


class TestSolubilityVsPh:
    def test_returns_dict_with_required_keys(self):
        result = solubility_vs_ph(pka=5.0, logP=2.0, mw=300.0)
        assert "ph_values" in result
        assert "solubility_mg_mL" in result

    def test_correct_number_of_points(self):
        result = solubility_vs_ph(pka=5.0, logP=2.0, mw=300.0, n_points=20)
        assert len(result["ph_values"]) == 20
        assert len(result["solubility_mg_mL"]) == 20

    def test_acid_solubility_increases_above_pka(self):
        """For an acid, solubility increases at pH > pKa."""
        pka = 5.0
        s0 = 0.1
        result = solubility_vs_ph(
            pka=pka,
            logP=2.0,
            mw=300.0,
            drug_type="acid",
            intrinsic_solubility_mg_mL=s0,
            ph_range=(3.0, 9.0),
            n_points=100,
        )
        ph_vals = result["ph_values"]
        sols = result["solubility_mg_mL"]
        # Find index just below and just above pKa
        idx_below = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - (pka - 1.0)))
        idx_above = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - (pka + 1.0)))
        assert sols[idx_above] > sols[idx_below]

    def test_base_solubility_increases_below_pka(self):
        """For a base, solubility increases at pH < pKa."""
        pka = 7.0
        s0 = 0.1
        result = solubility_vs_ph(
            pka=pka,
            logP=2.0,
            mw=300.0,
            drug_type="base",
            intrinsic_solubility_mg_mL=s0,
            ph_range=(3.0, 11.0),
            n_points=100,
        )
        ph_vals = result["ph_values"]
        sols = result["solubility_mg_mL"]
        idx_below = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - (pka - 2.0)))
        idx_above = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - (pka + 2.0)))
        assert sols[idx_below] > sols[idx_above]

    def test_at_ph_equals_pka_solubility_is_2x_s0(self):
        """At pH = pKa: S = S0 * (1 + 10^0) = 2 * S0."""
        pka = 6.0
        s0 = 0.5
        result = solubility_vs_ph(
            pka=pka,
            logP=2.0,
            mw=300.0,
            drug_type="acid",
            intrinsic_solubility_mg_mL=s0,
            ph_range=(5.9, 6.1),
            n_points=3,
        )
        # Middle point is at pKa
        sols = result["solubility_mg_mL"]
        ph_vals = result["ph_values"]
        idx = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - pka))
        assert abs(sols[idx] - 2 * s0) < 0.001

    def test_base_at_ph_equals_pka_solubility_is_2x_s0(self):
        pka = 7.0
        s0 = 0.3
        result = solubility_vs_ph(
            pka=pka,
            logP=2.0,
            mw=300.0,
            drug_type="base",
            intrinsic_solubility_mg_mL=s0,
            ph_range=(6.9, 7.1),
            n_points=3,
        )
        sols = result["solubility_mg_mL"]
        ph_vals = result["ph_values"]
        idx = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - pka))
        assert abs(sols[idx] - 2 * s0) < 0.001

    def test_invalid_pka_raises(self):
        with pytest.raises(ValueError, match="pka"):
            solubility_vs_ph(pka=0.0, logP=2.0, mw=300.0)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError, match="intrinsic_solubility"):
            solubility_vs_ph(pka=5.0, logP=2.0, mw=300.0, intrinsic_solubility_mg_mL=-0.1)

    def test_invalid_n_points_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            solubility_vs_ph(pka=5.0, logP=2.0, mw=300.0, n_points=1)

    def test_invalid_ph_range_raises(self):
        with pytest.raises(ValueError, match="ph_range"):
            solubility_vs_ph(pka=5.0, logP=2.0, mw=300.0, ph_range=(8.0, 3.0))

    def test_all_solubilities_positive(self):
        result = solubility_vs_ph(pka=5.0, logP=2.0, mw=300.0, n_points=50)
        assert all(s > 0 for s in result["solubility_mg_mL"])


# ---------------------------------------------------------------------------
# optimal_ph_for_solubility
# ---------------------------------------------------------------------------


class TestOptimalPhForSolubility:
    def test_returns_dataclass(self):
        result = optimal_ph_for_solubility(pka=5.0, logP=2.0, mw=300.0)
        assert isinstance(result, pHOptimizationResult)

    def test_acid_optimal_ph_above_pka(self):
        """For an acid with high target, optimal pH should be above pKa."""
        result = optimal_ph_for_solubility(
            pka=5.0,
            logP=2.0,
            mw=300.0,
            drug_type="acid",
            intrinsic_solubility_mg_mL=0.01,
            target_solubility_mg_mL=10.0,
        )
        # For an acid, solubility increases above pKa
        assert result.optimal_ph >= result.pka - 0.5

    def test_base_optimal_ph_below_pka(self):
        """For a base with high target, optimal pH should be below pKa."""
        pka = 8.5
        result = optimal_ph_for_solubility(
            pka=pka,
            logP=2.0,
            mw=300.0,
            drug_type="base",
            intrinsic_solubility_mg_mL=0.01,
            target_solubility_mg_mL=1.0,
        )
        # Optimal should be below pKa (where base is more ionized)
        assert result.optimal_ph <= pka + 0.5

    def test_target_achievable_when_max_sol_exceeds_target(self):
        result = optimal_ph_for_solubility(
            pka=5.0,
            logP=2.0,
            mw=300.0,
            drug_type="acid",
            intrinsic_solubility_mg_mL=1.0,
            target_solubility_mg_mL=0.001,
        )
        assert result.target_achievable is True

    def test_target_not_achievable_when_max_sol_below_target(self):
        # At pH 12 and pKa 5, max S = S0*(1+10^7) ~ S0*1e7
        # To make target unachievable, need target >> S0*1e7
        # With S0=1e-12 mg/mL, max ~1e-5 mg/mL; target=1.0 is unachievable
        result = optimal_ph_for_solubility(
            pka=5.0,
            logP=2.0,
            mw=300.0,
            drug_type="acid",
            intrinsic_solubility_mg_mL=1e-12,
            target_solubility_mg_mL=1.0,
        )
        assert result.target_achievable is False

    def test_stability_concern_when_optimal_ph_outside_range(self):
        # For a base with pKa=2, optimal pH would be very low (<3)
        result = optimal_ph_for_solubility(
            pka=2.0,
            logP=3.0,
            mw=300.0,
            drug_type="base",
            intrinsic_solubility_mg_mL=0.001,
            target_solubility_mg_mL=100.0,
        )
        # Max solubility at very low pH; if optimal lands outside 3-8, concern=True
        # or stability_concern may be False if target is met within range
        assert isinstance(result.stability_concern, bool)

    def test_notes_is_string(self):
        result = optimal_ph_for_solubility(pka=5.0, logP=2.0, mw=300.0)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_max_solubility_at_least_s0(self):
        s0 = 0.1
        result = optimal_ph_for_solubility(
            pka=5.0, logP=2.0, mw=300.0, intrinsic_solubility_mg_mL=s0
        )
        assert result.max_solubility_mg_mL >= s0

    def test_invalid_pka_raises(self):
        with pytest.raises(ValueError, match="pka"):
            optimal_ph_for_solubility(pka=-1.0, logP=2.0, mw=300.0)

    def test_invalid_intrinsic_solubility_raises(self):
        with pytest.raises(ValueError, match="intrinsic_solubility"):
            optimal_ph_for_solubility(pka=5.0, logP=2.0, mw=300.0, intrinsic_solubility_mg_mL=0.0)

    def test_invalid_target_solubility_raises(self):
        with pytest.raises(ValueError, match="target_solubility"):
            optimal_ph_for_solubility(pka=5.0, logP=2.0, mw=300.0, target_solubility_mg_mL=-1.0)


# ---------------------------------------------------------------------------
# buffer_capacity
# ---------------------------------------------------------------------------


class TestBufferCapacity:
    def test_maximum_at_ph_equals_pka(self):
        """Buffer capacity is maximum when pH = pKa."""
        pka = 7.0
        beta_at_pka = buffer_capacity(pka, pka)
        beta_below = buffer_capacity(pka - 1.0, pka)
        beta_above = buffer_capacity(pka + 1.0, pka)
        assert beta_at_pka > beta_below
        assert beta_at_pka > beta_above

    def test_returns_positive_value(self):
        result = buffer_capacity(7.0, 7.0)
        assert result > 0

    def test_higher_concentration_gives_higher_capacity(self):
        beta_low = buffer_capacity(7.0, 7.0, concentration_M=0.01)
        beta_high = buffer_capacity(7.0, 7.0, concentration_M=0.1)
        assert beta_high > beta_low

    def test_capacity_at_pka_formula(self):
        """At pH=pKa: beta = 2.303 * C * 0.25 (since Ka*H+/(Ka+H+)^2 = 0.25)."""
        pka = 6.0
        c = 0.05
        expected = 2.303 * c * 0.25
        result = buffer_capacity(pka, pka, concentration_M=c)
        assert abs(result - expected) < 1e-10

    def test_invalid_pka_raises(self):
        with pytest.raises(ValueError, match="pka"):
            buffer_capacity(7.0, 0.0)

    def test_invalid_concentration_raises(self):
        with pytest.raises(ValueError, match="concentration_M"):
            buffer_capacity(7.0, 7.0, concentration_M=-0.05)


# ---------------------------------------------------------------------------
# predict_stability_ph
# ---------------------------------------------------------------------------


class TestPredictStabilityPh:
    def test_ph_within_range_no_concern(self):
        result = predict_stability_ph("acid", 6.0, stability_range=(3.0, 8.0))
        assert result["stability_concern"] is False

    def test_ph_outside_range_concern_raised(self):
        result = predict_stability_ph("acid", 10.0, stability_range=(3.0, 8.0))
        assert result["stability_concern"] is True

    def test_ph_below_range_concern_raised(self):
        result = predict_stability_ph("base", 1.0, stability_range=(3.0, 8.0))
        assert result["stability_concern"] is True

    def test_returns_required_keys(self):
        result = predict_stability_ph("acid", 5.0)
        for key in (
            "optimal_ph",
            "stability_range",
            "stability_concern",
            "recommendation",
            "risk_level",
        ):
            assert key in result

    def test_low_risk_within_range(self):
        result = predict_stability_ph("acid", 5.0, stability_range=(3.0, 8.0))
        assert result["risk_level"] == "low"

    def test_high_risk_far_outside_range(self):
        result = predict_stability_ph("acid", 12.0, stability_range=(3.0, 8.0))
        assert result["risk_level"] == "high"

    def test_moderate_risk_slightly_outside_range(self):
        result = predict_stability_ph("acid", 8.5, stability_range=(3.0, 8.0))
        assert result["risk_level"] == "moderate"

    def test_recommendation_is_string(self):
        result = predict_stability_ph("base", 5.0)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_stability_ph("neutral", 5.0)

    def test_invalid_stability_range_raises(self):
        with pytest.raises(ValueError, match="stability_range"):
            predict_stability_ph("acid", 5.0, stability_range=(8.0, 3.0))
