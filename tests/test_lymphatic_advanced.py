"""Tests for advanced lymphatic absorption model with TRL association (Phase 286)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.lymphatic_advanced import (
    LymphaticAdvancedResult,
    compare_fat_meals,
    simulate_lymphatic_advanced_pk,
    trl_association_fraction,
)

# ---------------------------------------------------------------------------
# trl_association_fraction — unit tests
# ---------------------------------------------------------------------------


class TestTrlAssociationFraction:
    """Tests for trl_association_fraction() function."""

    def test_returns_float(self):
        result = trl_association_fraction(logP=5.0, fat_content_g=20.0)
        assert isinstance(result, float)

    def test_range_zero_to_point95(self):
        """Result must be in [0, 0.95]."""
        for logP in [-2.0, 0.0, 3.0, 5.0, 7.0, 10.0]:
            for fat in [0.0, 10.0, 50.0, 100.0]:
                result = trl_association_fraction(logP, fat)
                assert 0.0 <= result <= 0.95

    def test_fasted_zero_fat_gives_zero(self):
        """No fat in meal → no chylomicron formation → no TRL association."""
        result = trl_association_fraction(logP=7.0, fat_content_g=0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_high_logp_high_fat_gives_high_fraction(self):
        """Highly lipophilic drug + high fat → high TRL association."""
        result = trl_association_fraction(logP=8.0, fat_content_g=50.0)
        assert result > 0.3

    def test_low_logp_gives_low_fraction(self):
        """Hydrophilic drug (logP < 2) → negligible TRL association."""
        result = trl_association_fraction(logP=1.0, fat_content_g=50.0)
        assert result < 0.05

    def test_increases_with_fat_content(self):
        """Higher fat content should increase TRL fraction."""
        r_low = trl_association_fraction(logP=6.0, fat_content_g=5.0)
        r_high = trl_association_fraction(logP=6.0, fat_content_g=45.0)
        assert r_high > r_low

    def test_increases_with_logp(self):
        """Higher logP should increase TRL fraction."""
        r_low = trl_association_fraction(logP=3.0, fat_content_g=30.0)
        r_high = trl_association_fraction(logP=7.0, fat_content_g=30.0)
        assert r_high > r_low

    def test_negative_fat_raises(self):
        with pytest.raises(ValueError, match="fat_content_g"):
            trl_association_fraction(logP=5.0, fat_content_g=-1.0)

    def test_saturates_at_high_fat(self):
        """Fat factor saturates at 50 g → no further increase beyond."""
        r_50 = trl_association_fraction(logP=7.0, fat_content_g=50.0)
        r_100 = trl_association_fraction(logP=7.0, fat_content_g=100.0)
        assert r_50 == pytest.approx(r_100, rel=1e-6)


# ---------------------------------------------------------------------------
# simulate_lymphatic_advanced_pk — unit tests
# ---------------------------------------------------------------------------


class TestSimulateLymphaticAdvancedPk:
    """Tests for simulate_lymphatic_advanced_pk() function."""

    def _default(self, **kwargs) -> LymphaticAdvancedResult:
        defaults = dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            logP=5.0,
            cl_L_per_h=10.0,
            vd_L=100.0,
            fat_content_g=20.0,
            ka_portal_per_h=1.5,
            ka_lymph_per_h=0.15,
            trl_clearance_per_h=0.05,
            t_end_h=24.0,
            dt_h=0.1,
        )
        defaults.update(kwargs)
        return simulate_lymphatic_advanced_pk(**defaults)

    def test_returns_lymphatic_advanced_result(self):
        result = self._default()
        assert isinstance(result, LymphaticAdvancedResult)

    def test_result_is_frozen(self):
        result = self._default()
        with pytest.raises((AttributeError, TypeError)):
            result.cmax = 0.0  # type: ignore[misc]

    def test_times_start_at_zero(self):
        result = self._default()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = self._default(t_end_h=12.0)
        assert result.times_h[-1] == pytest.approx(12.0)

    def test_initial_plasma_conc_zero(self):
        result = self._default()
        assert result.c_plasma[0] == pytest.approx(0.0)

    def test_cmax_positive(self):
        result = self._default()
        assert result.cmax > 0.0

    def test_auc_positive(self):
        result = self._default()
        assert result.auc > 0.0

    def test_profile_length_matches_times(self):
        result = self._default()
        assert len(result.c_plasma) == len(result.times_h)

    def test_drug_name_stored(self):
        result = self._default(drug_name="MyDrug")
        assert result.drug_name == "MyDrug"

    def test_dose_stored(self):
        result = self._default(dose_mg=200.0)
        assert result.dose_mg == pytest.approx(200.0)

    def test_logp_stored(self):
        result = self._default(logP=4.5)
        assert result.logP == pytest.approx(4.5)

    def test_fat_content_stored(self):
        result = self._default(fat_content_g=40.0)
        assert result.fat_content_g == pytest.approx(40.0)

    def test_lymph_fraction_range(self):
        """Lymph fraction must be in [0, 0.8]."""
        result = self._default()
        assert 0.0 <= result.lymph_fraction <= 0.8

    def test_trl_fraction_range(self):
        """TRL fraction must be in [0, 0.95]."""
        result = self._default()
        assert 0.0 <= result.trl_fraction <= 0.95

    def test_high_fat_increases_trl(self):
        """Higher fat content should increase TRL fraction."""
        r_fasted = self._default(fat_content_g=0.0)
        r_fed = self._default(fat_content_g=40.0)
        assert r_fed.trl_fraction >= r_fasted.trl_fraction

    def test_hydrophilic_drug_low_lymph(self):
        """Low logP drug → low lymphatic absorption."""
        result = self._default(logP=0.0)
        assert result.lymph_fraction < 0.1

    def test_lipophilic_drug_high_lymph(self):
        """High logP → high lymphatic fraction."""
        result = self._default(logP=8.0)
        assert result.lymph_fraction > 0.3

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            self._default(dose_mg=0.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            self._default(cl_L_per_h=-1.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            self._default(vd_L=0.0)

    def test_invalid_fat_content_raises(self):
        with pytest.raises(ValueError, match="fat_content_g"):
            self._default(fat_content_g=-5.0)

    def test_invalid_ka_portal_raises(self):
        with pytest.raises(ValueError, match="ka_portal"):
            self._default(ka_portal_per_h=0.0)

    def test_invalid_ka_lymph_raises(self):
        with pytest.raises(ValueError, match="ka_lymph"):
            self._default(ka_lymph_per_h=-0.1)

    def test_invalid_trl_clearance_raises(self):
        with pytest.raises(ValueError, match="trl_clearance"):
            self._default(trl_clearance_per_h=0.0)

    def test_notes_is_string(self):
        result = self._default()
        assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# compare_fat_meals — tests
# ---------------------------------------------------------------------------


class TestCompareFatMeals:
    """Tests for compare_fat_meals() function."""

    def test_returns_dict_with_three_keys(self):
        result = compare_fat_meals(
            drug_name="Drug",
            dose_mg=100.0,
            logP=6.0,
            cl_L_per_h=10.0,
            vd_L=100.0,
        )
        assert set(result.keys()) == {"fasted", "low_fat", "high_fat"}

    def test_each_value_is_lymphatic_advanced_result(self):
        result = compare_fat_meals(
            drug_name="Drug",
            dose_mg=100.0,
            logP=6.0,
            cl_L_per_h=10.0,
            vd_L=100.0,
        )
        for key, val in result.items():
            assert isinstance(val, LymphaticAdvancedResult), f"{key} wrong type"

    def test_fat_content_values_correct(self):
        result = compare_fat_meals("Drug", 100.0, 6.0, 10.0, 100.0)
        assert result["fasted"].fat_content_g == pytest.approx(0.0)
        assert result["low_fat"].fat_content_g == pytest.approx(10.0)
        assert result["high_fat"].fat_content_g == pytest.approx(40.0)

    def test_trl_ordering_for_lipophilic_drug(self):
        """For a lipophilic drug, TRL fraction should increase with fat content."""
        result = compare_fat_meals("LipoDrug", 100.0, 7.0, 10.0, 100.0)
        assert result["fasted"].trl_fraction == pytest.approx(0.0, abs=1e-10)
        assert result["high_fat"].trl_fraction >= result["low_fat"].trl_fraction
