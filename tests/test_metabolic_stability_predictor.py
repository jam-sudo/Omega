"""Tests for Phase 798 — metabolic_stability_predictor.py"""

import pytest

from omega_pbpk.prediction.metabolic_stability_predictor import (
    MetabolicStabilityResult,
    predict_metabolic_stability,
    screen_metabolic_stability,
)

# ---------------------------------------------------------------------------
# Return type and basic field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_type(self):
        result = predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0)
        assert isinstance(result, MetabolicStabilityResult)

    def test_result_is_frozen(self):
        result = predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0)
        with pytest.raises((AttributeError, TypeError)):
            result.compound_name = "Other"

    def test_t_half_positive(self):
        result = predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0)
        assert result.predicted_t_half_mic_min > 0.0

    def test_clint_positive(self):
        result = predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0)
        assert result.predicted_clint_mic_uL_per_min_per_mg > 0.0

    def test_human_cl_positive(self):
        result = predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0)
        assert result.scalable_to_human_cl_L_per_h > 0.0


# ---------------------------------------------------------------------------
# CYP site effects
# ---------------------------------------------------------------------------


class TestCypSiteEffects:
    def test_more_cyp3a4_sites_shorter_thalf(self):
        r0 = predict_metabolic_stability("D0", logp=2.0, mw_da=300.0, n_cyp3a4_sites=0)
        r2 = predict_metabolic_stability("D2", logp=2.0, mw_da=300.0, n_cyp3a4_sites=2)
        assert r2.predicted_t_half_mic_min < r0.predicted_t_half_mic_min

    def test_more_cyp2d6_sites_shorter_thalf(self):
        r0 = predict_metabolic_stability("D0", logp=2.0, mw_da=300.0, n_cyp2d6_sites=0)
        r2 = predict_metabolic_stability("D2", logp=2.0, mw_da=300.0, n_cyp2d6_sites=2)
        assert r2.predicted_t_half_mic_min < r0.predicted_t_half_mic_min

    def test_more_cyp2c9_sites_shorter_thalf(self):
        r0 = predict_metabolic_stability("D0", logp=2.0, mw_da=300.0, n_cyp2c9_sites=0)
        r2 = predict_metabolic_stability("D2", logp=2.0, mw_da=300.0, n_cyp2c9_sites=2)
        assert r2.predicted_t_half_mic_min < r0.predicted_t_half_mic_min

    def test_cyp_sites_stored(self):
        result = predict_metabolic_stability(
            "Drug", logp=2.0, mw_da=300.0, n_cyp3a4_sites=2, n_cyp2d6_sites=1, n_cyp2c9_sites=1
        )
        assert result.n_cyp3a4_sites == 2
        assert result.n_cyp2d6_sites == 1
        assert result.n_cyp2c9_sites == 1


# ---------------------------------------------------------------------------
# Structural modifier tests
# ---------------------------------------------------------------------------


class TestStructuralModifiers:
    def test_amide_more_stable_than_ester(self):
        """Amide (resistance) should give longer t_half than ester (labile)."""
        r_ester = predict_metabolic_stability("Ester", logp=2.0, mw_da=300.0, has_ester=True)
        r_amide = predict_metabolic_stability("Amide", logp=2.0, mw_da=300.0, has_amide=True)
        assert r_amide.predicted_t_half_mic_min > r_ester.predicted_t_half_mic_min

    def test_michael_acceptor_faster_metabolism(self):
        r_no = predict_metabolic_stability("NoMA", logp=2.0, mw_da=300.0)
        r_ma = predict_metabolic_stability("MA", logp=2.0, mw_da=300.0, has_michael_acceptor=True)
        assert r_ma.predicted_t_half_mic_min < r_no.predicted_t_half_mic_min


# ---------------------------------------------------------------------------
# Stability class tests
# ---------------------------------------------------------------------------


class TestStabilityClass:
    def test_stability_class_valid_values(self):
        result = predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0)
        assert result.metabolic_stability_class in ("high", "moderate", "low")

    def test_high_stability_long_thalf(self):
        """Low logP, high MW → high stability (t½ > 60 min)."""
        result = predict_metabolic_stability("Stable", logp=-1.0, mw_da=600.0)
        assert result.metabolic_stability_class == "high"
        assert result.predicted_t_half_mic_min > 60.0

    def test_low_stability_short_thalf(self):
        """High logP, low MW, multiple CYP sites → low stability (t½ < 30 min)."""
        result = predict_metabolic_stability(
            "Labile", logp=5.0, mw_da=200.0, n_cyp3a4_sites=3, n_cyp2d6_sites=2
        )
        assert result.metabolic_stability_class == "low"
        assert result.predicted_t_half_mic_min < 30.0


# ---------------------------------------------------------------------------
# Primary CYP determination
# ---------------------------------------------------------------------------


class TestPrimaryCyp:
    def test_cyp3a4_dominant(self):
        result = predict_metabolic_stability("D3A4", logp=2.0, mw_da=300.0, n_cyp3a4_sites=3)
        assert result.primary_cyp == "CYP3A4"

    def test_no_cyp_sites_non_cyp(self):
        result = predict_metabolic_stability("NonCYP", logp=2.0, mw_da=300.0)
        assert result.primary_cyp == "Non-CYP"


# ---------------------------------------------------------------------------
# Screening tests
# ---------------------------------------------------------------------------


class TestScreening:
    def test_screen_returns_list(self):
        compounds = [
            {"compound_name": "A", "logp": 1.0, "mw_da": 300.0},
            {"compound_name": "B", "logp": 3.0, "mw_da": 400.0},
        ]
        results = screen_metabolic_stability(compounds)
        assert isinstance(results, list)

    def test_screen_sorted_by_thalf_descending(self):
        """Most stable compound should be first."""
        compounds = [
            {"compound_name": "Labile", "logp": 5.0, "mw_da": 200.0, "n_cyp3a4_sites": 3},
            {"compound_name": "Stable", "logp": -1.0, "mw_da": 600.0},
            {"compound_name": "Moderate", "logp": 2.0, "mw_da": 350.0},
        ]
        results = screen_metabolic_stability(compounds)
        t_halves = [r.predicted_t_half_mic_min for r in results]
        assert t_halves == sorted(t_halves, reverse=True)

    def test_screen_correct_count(self):
        compounds = [{"compound_name": f"C{i}", "logp": float(i), "mw_da": 300.0} for i in range(5)]
        results = screen_metabolic_stability(compounds)
        assert len(results) == 5

    def test_screen_all_result_types(self):
        compounds = [
            {"compound_name": "A", "logp": 1.0, "mw_da": 300.0},
            {"compound_name": "B", "logp": 3.0, "mw_da": 400.0},
        ]
        results = screen_metabolic_stability(compounds)
        for r in results:
            assert isinstance(r, MetabolicStabilityResult)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError):
            predict_metabolic_stability("Drug", logp=2.0, mw_da=0.0)

    def test_invalid_mw_negative(self):
        with pytest.raises(ValueError):
            predict_metabolic_stability("Drug", logp=2.0, mw_da=-100.0)

    def test_invalid_fup_zero(self):
        with pytest.raises(ValueError):
            predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0, fup=0.0)

    def test_invalid_fup_greater_than_one(self):
        with pytest.raises(ValueError):
            predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0, fup=1.5)

    def test_valid_fup_one(self):
        # fup=1.0 should be valid (fully unbound)
        result = predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0, fup=1.0)
        assert result.predicted_t_half_mic_min > 0.0


# ---------------------------------------------------------------------------
# n_metabolic_alerts test
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_alerts_nonnegative(self):
        result = predict_metabolic_stability("Drug", logp=2.0, mw_da=300.0)
        assert result.n_metabolic_alerts >= 0

    def test_more_alerts_with_risk_factors(self):
        r_clean = predict_metabolic_stability("Clean", logp=2.0, mw_da=300.0)
        r_risky = predict_metabolic_stability(
            "Risky",
            logp=5.5,
            mw_da=300.0,
            has_ester=True,
            has_michael_acceptor=True,
            n_aromatic_rings=4,
        )
        assert r_risky.n_metabolic_alerts > r_clean.n_metabolic_alerts
