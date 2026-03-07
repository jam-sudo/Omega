"""Tests for pediatric dose optimization module (Phase 312)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pediatric_dose_optimization import (
    PediatricDoseOptResult,
    PediatricRegimenResult,
    _maturation_factor,
    _round_to_practical,
    mg_per_kg_dosing,
    optimize_pediatric_regimen,
    target_auc_dose,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestMaturationFactor:
    def test_adult_returns_one(self):
        assert _maturation_factor(24.0) == pytest.approx(1.0)

    def test_neonate_is_low(self):
        assert _maturation_factor(0.0) < 0.1

    def test_monotonically_increasing(self):
        ages = [0.5, 1, 3, 6, 12, 18, 24]
        values = [_maturation_factor(a) for a in ages]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_three_month_half_mature(self):
        """At TM50 = 3.8 months, maturation ~ 0.5."""
        mat = _maturation_factor(3.8)
        assert 0.4 < mat < 0.6


class TestRoundToPractical:
    def test_small_dose_rounds_to_5(self):
        assert _round_to_practical(3.0) == 5.0

    def test_exact_match(self):
        assert _round_to_practical(100.0) == 100.0

    def test_midpoint_rounds(self):
        result = _round_to_practical(60.0)
        assert result in [50.0, 75.0]

    def test_zero_returns_first_step(self):
        assert _round_to_practical(0.0) == 5.0


# ---------------------------------------------------------------------------
# target_auc_dose
# ---------------------------------------------------------------------------


class TestTargetAucDose:
    def _base_params(self, weight=15.0, age=24.0):
        return dict(
            drug_name="Amoxicillin",
            target_auc_mg_h_per_L=50.0,
            cl_adult_L_per_h=5.0,
            weight_kg=weight,
            age_months=age,
        )

    def test_returns_result(self):
        result = target_auc_dose(**self._base_params())
        assert isinstance(result, PediatricDoseOptResult)

    def test_pediatric_cl_less_than_adult(self):
        """Child CL (15 kg) < adult CL (70 kg)."""
        result = target_auc_dose(**self._base_params(weight=15.0))
        assert result.cl_pediatric < 5.0

    def test_dose_per_kg_positive(self):
        result = target_auc_dose(**self._base_params())
        assert result.dose_per_kg > 0.0

    def test_practical_dose_in_steps(self):
        """Practical dose must be one of the defined steps."""
        from omega_pbpk.clinical.pediatric_dose_optimization import _PRACTICAL_DOSE_STEPS

        result = target_auc_dose(**self._base_params())
        assert result.practical_dose_mg in _PRACTICAL_DOSE_STEPS

    def test_neonate_lower_cl(self):
        """Neonate (1 month) has lower CL than toddler (12 months)."""
        neonate = target_auc_dose(**self._base_params(weight=4.0, age=1.0))
        toddler = target_auc_dose(**self._base_params(weight=10.0, age=12.0))
        # CL per kg should be lower for neonate due to maturation
        neonate_cl_per_kg = neonate.cl_pediatric / neonate.weight_kg
        toddler_cl_per_kg = toddler.cl_pediatric / toddler.weight_kg
        assert neonate_cl_per_kg < toddler_cl_per_kg

    def test_invalid_weight(self):
        params = self._base_params()
        params["weight_kg"] = 0.0
        with pytest.raises(ValueError, match="weight_kg must be > 0"):
            target_auc_dose(**params)

    def test_invalid_age(self):
        params = self._base_params()
        params["age_months"] = -1.0
        with pytest.raises(ValueError, match="age_months must be >= 0"):
            target_auc_dose(**params)

    def test_invalid_target_auc(self):
        params = self._base_params()
        params["target_auc_mg_h_per_L"] = 0.0
        with pytest.raises(ValueError, match="target_auc_mg_h_per_L must be > 0"):
            target_auc_dose(**params)

    def test_invalid_cl(self):
        params = self._base_params()
        params["cl_adult_L_per_h"] = -1.0
        with pytest.raises(ValueError, match="cl_adult_L_per_h must be > 0"):
            target_auc_dose(**params)

    def test_result_frozen(self):
        result = target_auc_dose(**self._base_params())
        with pytest.raises(Exception):
            result.recommended_dose_mg = 999.0  # type: ignore[misc]

    def test_notes_nonempty(self):
        result = target_auc_dose(**self._base_params())
        assert len(result.notes) > 0

    def test_age_zero_months(self):
        """Newborn (0 months) should work and give very low CL."""
        result = target_auc_dose(**self._base_params(weight=3.5, age=0.0))
        assert result.cl_pediatric > 0.0
        assert "Neonate" in result.notes


# ---------------------------------------------------------------------------
# mg_per_kg_dosing
# ---------------------------------------------------------------------------


class TestMgPerKgDosing:
    def test_basic_dose(self):
        result = mg_per_kg_dosing("Amoxicillin", 10.0, weight_kg=20.0, age_months=24.0)
        assert isinstance(result, PediatricDoseOptResult)
        assert result.recommended_dose_mg == pytest.approx(200.0)

    def test_max_dose_cap(self):
        result = mg_per_kg_dosing(
            "Amoxicillin", 10.0, weight_kg=80.0, age_months=192.0, max_dose_mg=500.0
        )
        assert result.recommended_dose_mg == pytest.approx(500.0)
        assert "capped" in result.notes

    def test_no_cap_when_below_max(self):
        result = mg_per_kg_dosing(
            "Amoxicillin", 5.0, weight_kg=10.0, age_months=24.0, max_dose_mg=500.0
        )
        assert result.recommended_dose_mg == pytest.approx(50.0)
        assert "capped" not in result.notes

    def test_dose_per_kg_correct(self):
        result = mg_per_kg_dosing("Drug", 10.0, weight_kg=20.0, age_months=24.0)
        assert result.dose_per_kg == pytest.approx(10.0)

    def test_invalid_weight(self):
        with pytest.raises(ValueError, match="weight_kg must be > 0"):
            mg_per_kg_dosing("Drug", 10.0, weight_kg=0.0, age_months=24.0)

    def test_invalid_age(self):
        with pytest.raises(ValueError, match="age_months must be >= 0"):
            mg_per_kg_dosing("Drug", 10.0, weight_kg=20.0, age_months=-1.0)

    def test_invalid_ref_dose(self):
        with pytest.raises(ValueError, match="reference_dose_mg_per_kg must be > 0"):
            mg_per_kg_dosing("Drug", 0.0, weight_kg=20.0, age_months=24.0)

    def test_result_frozen(self):
        result = mg_per_kg_dosing("Drug", 10.0, weight_kg=20.0, age_months=24.0)
        with pytest.raises(Exception):
            result.recommended_dose_mg = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# optimize_pediatric_regimen
# ---------------------------------------------------------------------------


class TestOptimizePediatricRegimen:
    def _base_params(self):
        return dict(
            drug_name="Vancomycin",
            target_trough_mg_L=10.0,
            target_peak_mg_L=40.0,
            cl_adult_L_per_h=5.0,
            vd_adult_L=50.0,
            weight_kg=20.0,
            age_months=48.0,
            route="iv",
            interval_h=12.0,
        )

    def test_returns_result(self):
        result = optimize_pediatric_regimen(**self._base_params())
        assert isinstance(result, PediatricRegimenResult)

    def test_interval_is_valid(self):
        result = optimize_pediatric_regimen(**self._base_params())
        assert result.interval_h in [8.0, 12.0, 24.0]

    def test_recommended_dose_positive(self):
        result = optimize_pediatric_regimen(**self._base_params())
        assert result.recommended_dose_mg > 0.0

    def test_predicted_trough_positive(self):
        result = optimize_pediatric_regimen(**self._base_params())
        assert result.predicted_trough > 0.0

    def test_predicted_peak_positive(self):
        result = optimize_pediatric_regimen(**self._base_params())
        assert result.predicted_peak > 0.0

    def test_oral_route_works(self):
        params = self._base_params()
        params["route"] = "oral"
        result = optimize_pediatric_regimen(**params)
        assert result.predicted_trough >= 0.0

    def test_invalid_weight(self):
        params = self._base_params()
        params["weight_kg"] = 0.0
        with pytest.raises(ValueError, match="weight_kg must be > 0"):
            optimize_pediatric_regimen(**params)

    def test_invalid_age(self):
        params = self._base_params()
        params["age_months"] = -1.0
        with pytest.raises(ValueError, match="age_months must be >= 0"):
            optimize_pediatric_regimen(**params)

    def test_invalid_target_trough(self):
        params = self._base_params()
        params["target_trough_mg_L"] = 0.0
        with pytest.raises(ValueError, match="target_trough_mg_L must be > 0"):
            optimize_pediatric_regimen(**params)

    def test_invalid_target_peak(self):
        params = self._base_params()
        params["target_peak_mg_L"] = -1.0
        with pytest.raises(ValueError, match="target_peak_mg_L must be > 0"):
            optimize_pediatric_regimen(**params)

    def test_invalid_cl(self):
        params = self._base_params()
        params["cl_adult_L_per_h"] = 0.0
        with pytest.raises(ValueError, match="cl_adult_L_per_h must be > 0"):
            optimize_pediatric_regimen(**params)

    def test_invalid_vd(self):
        params = self._base_params()
        params["vd_adult_L"] = -1.0
        with pytest.raises(ValueError, match="vd_adult_L must be > 0"):
            optimize_pediatric_regimen(**params)

    def test_invalid_route(self):
        params = self._base_params()
        params["route"] = "subcutaneous"
        with pytest.raises(ValueError, match="route must be 'oral' or 'iv'"):
            optimize_pediatric_regimen(**params)

    def test_result_frozen(self):
        result = optimize_pediatric_regimen(**self._base_params())
        with pytest.raises(Exception):
            result.recommended_dose_mg = 999.0  # type: ignore[misc]

    def test_notes_nonempty(self):
        result = optimize_pediatric_regimen(**self._base_params())
        assert len(result.notes) > 0

    def test_neonate_works(self):
        params = self._base_params()
        params["weight_kg"] = 3.5
        params["age_months"] = 0.5
        result = optimize_pediatric_regimen(**params)
        assert result.predicted_trough >= 0.0
