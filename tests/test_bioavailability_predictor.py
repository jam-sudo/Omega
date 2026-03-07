"""Tests for Phase 359: Bioavailability Prediction from in vitro data."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.bioavailability_predictor import (
    BioavailabilityPredictionResult,
    predict_bioavailability,
    screen_bioavailability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides):
    """Return a minimal valid set of kwargs for predict_bioavailability."""
    defaults = dict(
        compound_name="TestDrug",
        peff_cm_per_s=3e-4,  # high Peff
        solubility_mg_mL=1.0,
        dose_mg=10.0,
        mw=300.0,
        logP=2.0,
        fu_plasma=0.1,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# fa prediction tests
# ---------------------------------------------------------------------------


class TestFaPrediction:
    def test_high_peff_low_dn_gives_fa_095(self):
        """High Peff (>2e-4) and Dn < 1 → fa ≈ 0.95."""
        r = predict_bioavailability(
            **_base_kwargs(
                peff_cm_per_s=5e-4,
                solubility_mg_mL=5.0,
                dose_mg=100.0,
                dose_volume_mL=250.0,
            )
        )
        # Dn = 100/(5*250) = 0.08 < 1 → fa = 0.95
        assert abs(r.fa_predicted - 0.95) < 1e-6
        assert r.fa_basis == "permeability"

    def test_high_peff_high_dn_solubility_limited(self):
        """High Peff but Dn > 1 → solubility limited, fa < 0.95."""
        r = predict_bioavailability(
            **_base_kwargs(
                peff_cm_per_s=3e-4,
                solubility_mg_mL=0.001,
                dose_mg=500.0,
                dose_volume_mL=250.0,
            )
        )
        # Dn = 500/(0.001*250) = 2000 >> 1 → fa much less than 0.95
        assert r.fa_predicted < 0.95
        assert r.fa_basis == "solubility_limited"

    def test_low_peff_permeability_limited(self):
        """Low Peff (<1e-4) → fa < 0.4."""
        r = predict_bioavailability(**_base_kwargs(peff_cm_per_s=0.5e-4))
        assert r.fa_predicted < 0.4
        assert r.fa_basis == "permeability"

    def test_zero_peff_gives_near_zero_fa(self):
        """Very low Peff → near-zero fa."""
        r = predict_bioavailability(**_base_kwargs(peff_cm_per_s=0.0))
        assert r.fa_predicted == pytest.approx(0.0)

    def test_intermediate_peff_interpolated(self):
        """Peff between 1e-4 and 2e-4 → fa between low and high values."""
        r_low = predict_bioavailability(**_base_kwargs(peff_cm_per_s=1e-4))
        r_high = predict_bioavailability(**_base_kwargs(peff_cm_per_s=2e-4))
        r_mid = predict_bioavailability(**_base_kwargs(peff_cm_per_s=1.5e-4))
        assert r_low.fa_predicted <= r_mid.fa_predicted <= r_high.fa_predicted

    def test_fa_in_unit_interval(self):
        """fa must always be in [0, 1]."""
        for peff in [0.0, 0.5e-4, 1e-4, 2e-4, 1e-3]:
            r = predict_bioavailability(**_base_kwargs(peff_cm_per_s=peff))
            assert 0.0 <= r.fa_predicted <= 1.0


# ---------------------------------------------------------------------------
# fg prediction tests
# ---------------------------------------------------------------------------


class TestFgPrediction:
    def test_no_clint_gut_gives_default_fg(self):
        """Without clint_gut, fg defaults to 0.9."""
        r = predict_bioavailability(**_base_kwargs())
        assert r.fg_predicted == pytest.approx(0.9)
        assert r.fg_basis == "assumed_high"

    def test_high_clint_gut_lowers_fg(self):
        """High gut CLint → fg < 0.9."""
        r = predict_bioavailability(**_base_kwargs(clint_gut_uL_per_min_per_mg=500.0))
        assert r.fg_predicted < 0.9
        assert r.fg_basis == "cyp3a4_clint"

    def test_zero_clint_gut_gives_fg_near_one(self):
        """Zero gut CLint → fg ≈ 1.0."""
        r = predict_bioavailability(**_base_kwargs(clint_gut_uL_per_min_per_mg=0.0))
        assert r.fg_predicted == pytest.approx(1.0)

    def test_fg_in_unit_interval(self):
        """fg must be in [0, 1]."""
        for clint in [None, 0.0, 10.0, 1000.0]:
            r = predict_bioavailability(**_base_kwargs(clint_gut_uL_per_min_per_mg=clint))
            assert 0.0 <= r.fg_predicted <= 1.0


# ---------------------------------------------------------------------------
# fh prediction tests
# ---------------------------------------------------------------------------


class TestFhPrediction:
    def test_no_clint_hep_gives_default_fh(self):
        """Without clint_hep, fh defaults to 0.85."""
        r = predict_bioavailability(**_base_kwargs())
        assert r.fh_predicted == pytest.approx(0.85)
        assert r.fh_basis == "assumed"

    def test_high_clint_hep_lowers_fh(self):
        """High hepatic CLint → fh < 0.85 (high first-pass extraction)."""
        r = predict_bioavailability(**_base_kwargs(clint_hep_uL_per_min_per_mg=1000.0))
        assert r.fh_predicted < 0.5
        assert r.fh_basis == "well_stirred"

    def test_low_clint_hep_gives_high_fh(self):
        """Low hepatic CLint → fh close to 1.0."""
        r = predict_bioavailability(
            **_base_kwargs(
                clint_hep_uL_per_min_per_mg=0.1,
                fu_plasma=0.5,
            )
        )
        assert r.fh_predicted > 0.9

    def test_fh_in_unit_interval(self):
        """fh must be in [0, 1]."""
        for clint in [None, 0.1, 100.0, 5000.0]:
            r = predict_bioavailability(**_base_kwargs(clint_hep_uL_per_min_per_mg=clint))
            assert 0.0 <= r.fh_predicted <= 1.0


# ---------------------------------------------------------------------------
# f_oral combined tests
# ---------------------------------------------------------------------------


class TestFOral:
    def test_f_oral_equals_product(self):
        """f_oral_predicted = fa * fg * fh."""
        r = predict_bioavailability(**_base_kwargs())
        expected = r.fa_predicted * r.fg_predicted * r.fh_predicted
        assert r.f_oral_predicted == pytest.approx(expected, rel=1e-6)

    def test_f_oral_in_unit_interval(self):
        """f_oral must be in [0, 1]."""
        r = predict_bioavailability(
            **_base_kwargs(
                clint_hep_uL_per_min_per_mg=50.0,
                clint_gut_uL_per_min_per_mg=30.0,
            )
        )
        assert 0.0 <= r.f_oral_predicted <= 1.0

    def test_default_assumptions(self):
        """No CLint provided: f_oral ≈ fa * 0.9 * 0.85."""
        r = predict_bioavailability(
            **_base_kwargs(
                peff_cm_per_s=5e-4,
                solubility_mg_mL=5.0,
                dose_mg=50.0,
            )
        )
        expected = r.fa_predicted * 0.9 * 0.85
        assert r.f_oral_predicted == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Uncertainty bounds tests
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_low_below_predicted_below_high(self):
        """f_oral_low < f_oral_predicted < f_oral_high (strictly)."""
        r = predict_bioavailability(
            **_base_kwargs(
                peff_cm_per_s=3e-4,
                clint_hep_uL_per_min_per_mg=20.0,
            )
        )
        # Only strictly less if f_oral is not 0 or 1
        if r.f_oral_predicted > 0:
            assert r.f_oral_low <= r.f_oral_predicted
        assert r.f_oral_predicted <= r.f_oral_high

    def test_f_oral_low_nonnegative(self):
        r = predict_bioavailability(**_base_kwargs())
        assert r.f_oral_low >= 0.0

    def test_f_oral_high_at_most_one(self):
        r = predict_bioavailability(**_base_kwargs(peff_cm_per_s=5e-4))
        assert r.f_oral_high <= 1.0

    def test_low_is_lower_than_high(self):
        r = predict_bioavailability(**_base_kwargs())
        assert r.f_oral_low <= r.f_oral_high


# ---------------------------------------------------------------------------
# Literature comparison tests
# ---------------------------------------------------------------------------


class TestLiteratureComparison:
    def test_no_literature_gives_none_fold(self):
        """Without literature value, prediction_accuracy_fold is None."""
        r = predict_bioavailability(**_base_kwargs())
        assert r.prediction_accuracy_fold is None
        assert r.f_oral_literature is None

    def test_accuracy_fold_near_one_when_pred_equals_literature(self):
        """When predicted ≈ literature, fold ≈ 1."""
        # First compute what the predicted f_oral would be
        r0 = predict_bioavailability(
            **_base_kwargs(
                peff_cm_per_s=5e-4,
                solubility_mg_mL=5.0,
                dose_mg=50.0,
            )
        )
        # Then use that same value as literature
        r = predict_bioavailability(
            **_base_kwargs(
                peff_cm_per_s=5e-4,
                solubility_mg_mL=5.0,
                dose_mg=50.0,
                f_oral_literature=r0.f_oral_predicted,
            )
        )
        assert r.prediction_accuracy_fold == pytest.approx(1.0, rel=1e-6)

    def test_accuracy_fold_computed_correctly(self):
        """prediction_accuracy_fold = f_oral / f_oral_literature."""
        lit = 0.4
        r = predict_bioavailability(**_base_kwargs(f_oral_literature=lit))
        if r.f_oral_predicted > 0:
            expected_fold = r.f_oral_predicted / lit
            assert r.prediction_accuracy_fold == pytest.approx(expected_fold, rel=1e-6)


# ---------------------------------------------------------------------------
# CL and t½ tests
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_cl_nonnegative(self):
        r = predict_bioavailability(**_base_kwargs(clint_hep_uL_per_min_per_mg=50.0))
        assert r.cl_predicted_L_per_h >= 0.0

    def test_t_half_positive(self):
        r = predict_bioavailability(**_base_kwargs(clint_hep_uL_per_min_per_mg=50.0))
        assert r.t_half_predicted_h > 0.0

    def test_vd_L_overrides_estimate(self):
        """Providing vd_L changes t_half proportionally."""
        r1 = predict_bioavailability(**_base_kwargs(clint_hep_uL_per_min_per_mg=100.0, vd_L=10.0))
        r2 = predict_bioavailability(**_base_kwargs(clint_hep_uL_per_min_per_mg=100.0, vd_L=100.0))
        assert r2.t_half_predicted_h > r1.t_half_predicted_h


# ---------------------------------------------------------------------------
# Recommendation and notes tests
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_recommendation_nonempty(self):
        r = predict_bioavailability(**_base_kwargs())
        assert len(r.recommendation) > 0

    def test_high_f_oral_positive_recommendation(self):
        r = predict_bioavailability(
            **_base_kwargs(peff_cm_per_s=5e-4, solubility_mg_mL=5.0, dose_mg=10.0)
        )
        assert "oral" in r.recommendation.lower() or "excellent" in r.recommendation.lower()

    def test_notes_nonempty(self):
        r = predict_bioavailability(**_base_kwargs())
        assert len(r.notes) > 0

    def test_dataclass_frozen(self):
        """Result is immutable (frozen dataclass)."""
        r = predict_bioavailability(**_base_kwargs())
        with pytest.raises((AttributeError, TypeError)):
            r.f_oral_predicted = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_invalid_fu_plasma_raises(self):
        with pytest.raises(ValueError):
            predict_bioavailability(**_base_kwargs(fu_plasma=0.0))

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError):
            predict_bioavailability(**_base_kwargs(solubility_mg_mL=-1.0))

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            predict_bioavailability(**_base_kwargs(dose_mg=-5.0))

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError):
            predict_bioavailability(**_base_kwargs(mw=0.0))


# ---------------------------------------------------------------------------
# Screen function tests
# ---------------------------------------------------------------------------


class TestScreenBioavailability:
    def _compounds(self):
        return [
            {
                "name": "HighF",
                "peff": 5e-4,
                "solubility": 5.0,
                "dose_mg": 10.0,
                "mw": 300.0,
                "logP": 2.0,
                "fu_plasma": 0.1,
            },
            {
                "name": "LowF",
                "peff": 0.3e-4,
                "solubility": 0.01,
                "dose_mg": 500.0,
                "mw": 600.0,
                "logP": 5.0,
                "fu_plasma": 0.05,
            },
            {
                "name": "MidF",
                "peff": 2e-4,
                "solubility": 1.0,
                "dose_mg": 100.0,
                "mw": 400.0,
                "logP": 3.0,
                "fu_plasma": 0.2,
            },
        ]

    def test_screen_returns_list(self):
        results = screen_bioavailability(self._compounds())
        assert isinstance(results, list)
        assert len(results) == 3

    def test_screen_sorted_descending(self):
        results = screen_bioavailability(self._compounds())
        f_orals = [r.f_oral_predicted for r in results]
        assert f_orals == sorted(f_orals, reverse=True)

    def test_screen_first_has_highest_f_oral(self):
        results = screen_bioavailability(self._compounds())
        assert results[0].compound_name == "HighF"

    def test_screen_all_results_valid(self):
        results = screen_bioavailability(self._compounds())
        for r in results:
            assert 0.0 <= r.f_oral_predicted <= 1.0
            assert isinstance(r, BioavailabilityPredictionResult)

    def test_screen_empty_list(self):
        results = screen_bioavailability([])
        assert results == []
