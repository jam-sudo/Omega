"""Tests for pediatric biopharmaceutics module (Phase 295)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.pediatric_biopharmaceutics import (
    PediatricAbsorptionResult,
    PediatricGIParams,
    compare_age_groups,
    pediatric_gastric_ph,
    pediatric_gi_parameters,
    simulate_pediatric_absorption,
)

# ---------------------------------------------------------------------------
# pediatric_gastric_ph
# ---------------------------------------------------------------------------


class TestPediatricGastricPH:
    def test_neonate_ph_near_65(self):
        # At birth (0 months) pH ≈ 6.5
        ph = pediatric_gastric_ph(0.0)
        assert abs(ph - 6.5) < 0.01

    def test_ph_decreases_with_age(self):
        ph_neonate = pediatric_gastric_ph(0.0)
        ph_infant = pediatric_gastric_ph(3.0)
        ph_adult = pediatric_gastric_ph(24.0)
        assert ph_neonate > ph_infant > ph_adult

    def test_adult_ph_approaches_2(self):
        # At 24+ months, pH should be close to adult (~2)
        ph = pediatric_gastric_ph(36.0)
        assert ph < 2.5

    def test_clamp_lower_bound(self):
        # Should never go below 1.5
        ph = pediatric_gastric_ph(1000.0)
        assert ph >= 1.5

    def test_clamp_upper_bound(self):
        # Should never exceed 7.0
        ph = pediatric_gastric_ph(0.0)
        assert ph <= 7.0

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="age_months must be >= 0"):
            pediatric_gastric_ph(-1.0)

    def test_1_5_month_ph_between_neonate_and_adult(self):
        ph = pediatric_gastric_ph(1.5)
        assert 1.5 < ph < 6.5

    def test_3_month_low_ph(self):
        # By 3 months, pH should be much lower (acid production matures)
        ph = pediatric_gastric_ph(3.0)
        assert ph < 4.0


# ---------------------------------------------------------------------------
# pediatric_gi_parameters
# ---------------------------------------------------------------------------


class TestPediatricGIParameters:
    def _get(self, age_months=0.5, weight_kg=3.5):
        return pediatric_gi_parameters(age_months, weight_kg)

    def test_returns_dataclass(self):
        assert isinstance(self._get(), PediatricGIParams)

    def test_neonate_slow_emptying(self):
        # Neonates have slower gastric emptying
        neonate = pediatric_gi_parameters(0.5, 3.5)
        adult = pediatric_gi_parameters(216.0, 70.0)
        assert neonate.gastric_emptying_h > adult.gastric_emptying_h

    def test_gastric_emptying_clamp_at_12_months(self):
        # At >=12 months, gastric emptying time plateaus
        params_12 = pediatric_gi_parameters(12.0, 10.0)
        params_24 = pediatric_gi_parameters(24.0, 12.0)
        # Both should have emptying_h = 0.5 (min value)
        assert abs(params_12.gastric_emptying_h - params_24.gastric_emptying_h) < 0.01

    def test_surface_fraction_neonate_small(self):
        neonate = pediatric_gi_parameters(0.5, 3.5)
        assert neonate.surface_fraction < 0.3

    def test_surface_fraction_adult_near_1(self):
        adult = pediatric_gi_parameters(216.0, 70.0)
        assert abs(adult.surface_fraction - 1.0) < 0.01

    def test_bile_acid_factor_neonate_low(self):
        neonate = pediatric_gi_parameters(0.5, 3.5)
        assert neonate.bile_acid_factor < 0.4

    def test_bile_acid_factor_adult_high(self):
        adult = pediatric_gi_parameters(216.0, 70.0)
        assert adult.bile_acid_factor >= 0.99

    def test_transit_time_neonate_longer(self):
        neonate = pediatric_gi_parameters(0.5, 3.5)
        adult = pediatric_gi_parameters(216.0, 70.0)
        assert neonate.transit_time_h > adult.transit_time_h

    def test_negative_age_raises(self):
        with pytest.raises(ValueError):
            pediatric_gi_parameters(-1.0, 3.5)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            pediatric_gi_parameters(1.0, 0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            pediatric_gi_parameters(1.0, -5.0)

    def test_frozen_dataclass(self):
        params = self._get()
        with pytest.raises(Exception):
            params.gastric_ph = 3.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# simulate_pediatric_absorption
# ---------------------------------------------------------------------------


class TestSimulatePediatricAbsorption:
    def _run(self, **kw):
        defaults = dict(
            drug_name="TestDrug",
            dose_mg_per_kg=10.0,
            weight_kg=7.0,
            age_months=6.0,
            logP=2.0,
            pka=7.4,
            solubility_mg_mL=1.0,
            cl_adult_L_per_h_per_kg=0.3,
            vd_adult_L_per_kg=1.0,
        )
        defaults.update(kw)
        return simulate_pediatric_absorption(**defaults)

    def test_returns_result_type(self):
        assert isinstance(self._run(), PediatricAbsorptionResult)

    def test_result_is_frozen(self):
        r = self._run()
        with pytest.raises(Exception):
            r.cmax = 999.0  # type: ignore[misc]

    def test_times_start_at_zero(self):
        r = self._run()
        assert r.times_h[0] == 0.0

    def test_cmax_positive(self):
        r = self._run()
        assert r.cmax > 0.0

    def test_auc_positive(self):
        r = self._run()
        assert r.auc > 0.0

    def test_dose_mg_equals_dose_per_kg_times_weight(self):
        r = self._run(dose_mg_per_kg=10.0, weight_kg=7.0)
        assert abs(r.dose_mg - 70.0) < 1e-4

    def test_ka_effective_positive(self):
        r = self._run()
        assert r.ka_effective > 0.0

    def test_neonate_lower_ka_than_adult(self):
        neonate = self._run(age_months=0.5, weight_kg=3.5)
        adult = self._run(age_months=216.0, weight_kg=70.0)
        assert neonate.ka_effective < adult.ka_effective

    def test_concentration_profile_length(self):
        r = self._run(t_end_h=24.0, dt_h=0.1)
        # Should have n_steps + 1 points
        assert len(r.times_h) == len(r.c_plasma)
        assert len(r.times_h) > 1

    def test_concentration_nonnegative(self):
        r = self._run()
        assert all(c >= 0.0 for c in r.c_plasma)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            self._run(weight_kg=0.0)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            self._run(dose_mg_per_kg=-1.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            self._run(cl_adult_L_per_h_per_kg=0.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            self._run(vd_adult_L_per_kg=-0.5)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError):
            self._run(solubility_mg_mL=0.0)

    def test_notes_contains_gastric_ph(self):
        r = self._run()
        assert "Gastric pH" in r.notes

    def test_higher_weight_higher_dose(self):
        r_light = self._run(weight_kg=7.0, dose_mg_per_kg=10.0)
        r_heavy = self._run(weight_kg=70.0, dose_mg_per_kg=10.0)
        assert r_heavy.dose_mg > r_light.dose_mg


# ---------------------------------------------------------------------------
# compare_age_groups
# ---------------------------------------------------------------------------


class TestCompareAgeGroups:
    def _run(self, **kw):
        defaults = dict(
            drug_name="Amoxicillin",
            dose_mg_per_kg=10.0,
            logP=0.87,
            pka=3.5,
            solubility_mg_mL=4.0,
            cl_adult_per_kg=0.25,
            vd_adult_per_kg=0.3,
        )
        defaults.update(kw)
        return compare_age_groups(**defaults)

    def test_returns_dict(self):
        assert isinstance(self._run(), dict)

    def test_has_all_age_groups(self):
        result = self._run()
        expected = {"neonate", "infant", "toddler", "child", "adult"}
        assert set(result.keys()) == expected

    def test_all_values_are_results(self):
        result = self._run()
        for v in result.values():
            assert isinstance(v, PediatricAbsorptionResult)

    def test_adult_higher_ka_than_neonate(self):
        result = self._run()
        assert result["adult"].ka_effective > result["neonate"].ka_effective

    def test_dose_mg_increases_with_weight(self):
        result = self._run(dose_mg_per_kg=10.0)
        # adult (70 kg) > child (25 kg) > toddler (12 kg) > infant (7 kg) > neonate (3.5 kg)
        assert result["adult"].dose_mg > result["neonate"].dose_mg

    def test_all_auc_positive(self):
        result = self._run()
        for label, r in result.items():
            assert r.auc > 0.0, f"{label} has non-positive AUC"
