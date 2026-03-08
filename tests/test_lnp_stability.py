"""Tests for Phase 534 — Drug Lipid Nanoparticle Stability."""

import math

import pytest

from omega_pbpk.prediction.lnp_stability import (
    LNPStabilityResult,
    compare_storage_temperatures,
    predict_lnp_stability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(**kwargs):
    defaults = dict(
        formulation_name="LNP-A",
        drug_name="mRNA",
        pka_lipid=6.5,
        initial_diameter_nm=100.0,
        initial_pdi=0.1,
        temperature_celsius=4.0,
        ph_storage=7.4,
        t_max_days=90.0,
    )
    defaults.update(kwargs)
    return predict_lnp_stability(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = make_result()
        assert isinstance(r, LNPStabilityResult)

    def test_not_frozen(self):
        r = make_result()
        r.notes = "updated"  # should not raise
        assert r.notes == "updated"


# ---------------------------------------------------------------------------
# Fields presence
# ---------------------------------------------------------------------------


class TestFields:
    def test_formulation_name(self):
        r = make_result(formulation_name="TestLNP")
        assert r.formulation_name == "TestLNP"

    def test_drug_name(self):
        r = make_result(drug_name="siRNA")
        assert r.drug_name == "siRNA"

    def test_pka_stored(self):
        r = make_result(pka_lipid=6.2)
        assert r.pka_lipid == pytest.approx(6.2)

    def test_temperature_stored(self):
        r = make_result(temperature_celsius=25.0)
        assert r.temperature_celsius == pytest.approx(25.0)

    def test_ph_stored(self):
        r = make_result(ph_storage=7.4)
        assert r.ph_storage == pytest.approx(7.4)

    def test_times_days_is_list(self):
        r = make_result()
        assert isinstance(r.times_days, list)

    def test_fraction_retained_is_list(self):
        r = make_result()
        assert isinstance(r.fraction_retained, list)

    def test_pdi_values_is_list(self):
        r = make_result()
        assert isinstance(r.pdi_values, list)

    def test_diameter_nm_is_list(self):
        r = make_result()
        assert isinstance(r.diameter_nm, list)

    def test_notes_is_str(self):
        r = make_result()
        assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# fraction_retained[0] = 1.0
# ---------------------------------------------------------------------------


class TestFractionRetainedStart:
    def test_first_value_is_one(self):
        r = make_result()
        assert r.fraction_retained[0] == pytest.approx(1.0)

    def test_first_time_is_zero(self):
        r = make_result()
        assert r.times_days[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fraction_retained is decreasing
# ---------------------------------------------------------------------------


class TestFractionRetainedDecreasing:
    def test_monotone_decreasing(self):
        r = make_result()
        for i in range(1, len(r.fraction_retained)):
            assert r.fraction_retained[i] <= r.fraction_retained[i - 1]


# ---------------------------------------------------------------------------
# Higher temperature → higher k_leak → lower t50
# ---------------------------------------------------------------------------


class TestTemperatureEffect:
    def test_higher_temp_higher_kleak(self):
        r_cold = make_result(temperature_celsius=4.0)
        r_warm = make_result(temperature_celsius=25.0)
        assert r_warm.k_leak_per_h > r_cold.k_leak_per_h

    def test_higher_temp_lower_t50(self):
        r_cold = make_result(temperature_celsius=4.0)
        r_warm = make_result(temperature_celsius=25.0)
        assert r_warm.t50_days < r_cold.t50_days


# ---------------------------------------------------------------------------
# t50 = ln(2) / (k_leak * 24)
# ---------------------------------------------------------------------------


class TestT50Formula:
    def test_t50_matches_formula(self):
        r = make_result(temperature_celsius=4.0, ph_storage=7.4, pka_lipid=6.5)
        expected_t50 = math.log(2.0) / (r.k_leak_per_h * 24.0)
        assert r.t50_days == pytest.approx(expected_t50, rel=1e-6)

    def test_t50_at_reference_temp(self):
        """At 4°C: T_factor=1, pH 7.4 > pKa 6.5 → ph_factor=2."""
        r = make_result(temperature_celsius=4.0, ph_storage=7.4, pka_lipid=6.5)
        k_expected = 0.01 * 1.0 * 2.0  # k_base * T_factor=1 * pH_factor=2
        t50_expected = math.log(2.0) / (k_expected * 24.0)
        assert r.t50_days == pytest.approx(t50_expected, rel=1e-6)


# ---------------------------------------------------------------------------
# shelf_life_90pct_days
# ---------------------------------------------------------------------------


class TestShelfLife:
    def test_shelf_life_minus_one_if_retained(self):
        """Very low k_leak → always > 90% retention in 90 days."""
        # pH below pKa → ph_factor=1; very cold temp (-80°C) → tiny T_factor
        # At -80°C: T_factor=2^(-8.4)≈0.00296, k_leak≈0.0000296/h
        # fraction at day 90 ≈ 0.938 > 90% → shelf_life = -1
        r = make_result(temperature_celsius=-80.0, ph_storage=4.0, pka_lipid=6.5)
        assert r.shelf_life_90pct_days == pytest.approx(-1.0)

    def test_shelf_life_positive_for_high_temp(self):
        """High temperature causes rapid leakage → shelf_life < 90."""
        r = make_result(temperature_celsius=50.0, ph_storage=7.4, pka_lipid=6.5)
        # Should drop below 90% well before 90 days
        assert r.shelf_life_90pct_days > 0


# ---------------------------------------------------------------------------
# stability_class boundaries
# ---------------------------------------------------------------------------


class TestStabilityClass:
    def test_stable_class(self):
        """Very low temperature → t50 > 30d → stable class."""
        # At -80°C, T_factor≈0.0000296, t50 >> 30 → stable
        r = make_result(temperature_celsius=-80.0, ph_storage=4.0, pka_lipid=6.5)
        assert r.stability_class == "stable"

    def test_stability_class_is_string(self):
        r = make_result()
        assert r.stability_class in ("stable", "moderate", "unstable")

    def test_high_kleak_gives_unstable(self):
        """Very high temperature and high pH → unstable."""
        r = make_result(temperature_celsius=60.0, ph_storage=7.4, pka_lipid=5.0)
        assert r.stability_class == "unstable"

    def test_low_kleak_gives_stable(self):
        """Very low temperature → stable."""
        r = make_result(temperature_celsius=-80.0, ph_storage=4.0, pka_lipid=6.5)
        # T_factor = 2^((-80-4)/10) = 2^(-8.4) ≈ tiny → k_leak tiny → t50 >> 30
        assert r.stability_class == "stable"

    def test_moderate_class(self):
        """Moderate conditions give moderate class: t50 between 10 and 30 days."""
        # At -20°C, pH below pKa: T_factor≈0.189, k_leak≈0.00189/h, t50≈15.2d → moderate
        r_mod = make_result(temperature_celsius=-20.0, ph_storage=4.0, pka_lipid=6.5)
        assert r_mod.stability_class == "moderate"


# ---------------------------------------------------------------------------
# aggregation_risk
# ---------------------------------------------------------------------------


class TestAggregationRisk:
    def test_high_pdi_gives_aggregation(self):
        """initial_pdi=0.24 → PDI at day 90 = 0.24 + 0.09 = 0.33 > 0.25."""
        r = make_result(initial_pdi=0.24)
        assert r.aggregation_risk is True

    def test_low_pdi_no_aggregation(self):
        """initial_pdi=0.1 → PDI at day 90 = 0.1 + 0.09 = 0.19 < 0.25."""
        r = make_result(initial_pdi=0.1)
        assert r.aggregation_risk is False

    def test_borderline_pdi(self):
        """initial_pdi=0.16 → PDI at 90 days = 0.16 + 0.09 = 0.25, not > 0.25."""
        r = make_result(initial_pdi=0.16)
        assert r.aggregation_risk is False

    def test_just_above_borderline(self):
        """initial_pdi=0.161 → PDI at 90 days > 0.25."""
        r = make_result(initial_pdi=0.161)
        assert r.aggregation_risk is True


# ---------------------------------------------------------------------------
# storage_recommendation
# ---------------------------------------------------------------------------


class TestStorageRecommendation:
    def test_frozen(self):
        r = make_result(temperature_celsius=-20.0)
        assert r.storage_recommendation == "frozen"

    def test_refrigerated(self):
        r = make_result(temperature_celsius=4.0)
        assert r.storage_recommendation == "refrigerated"

    def test_room_temp(self):
        r = make_result(temperature_celsius=25.0)
        assert r.storage_recommendation == "room_temp"

    def test_boundary_2_celsius(self):
        r = make_result(temperature_celsius=2.0)
        assert r.storage_recommendation == "frozen"

    def test_boundary_8_celsius(self):
        r = make_result(temperature_celsius=8.0)
        assert r.storage_recommendation == "refrigerated"


# ---------------------------------------------------------------------------
# compare_storage_temperatures
# ---------------------------------------------------------------------------


class TestCompareStorageTemperatures:
    def test_returns_list(self):
        results = compare_storage_temperatures("LNP-A", "mRNA", 6.5, [4.0, 25.0, -20.0])
        assert isinstance(results, list)

    def test_length(self):
        results = compare_storage_temperatures("LNP-A", "mRNA", 6.5, [4.0, 25.0, -20.0])
        assert len(results) == 3

    def test_sorted_by_t50_descending(self):
        results = compare_storage_temperatures("LNP-A", "mRNA", 6.5, [4.0, 25.0, -20.0])
        t50s = [r.t50_days for r in results]
        assert t50s == sorted(t50s, reverse=True)

    def test_coldest_first(self):
        """Coldest temperature has longest t50 → appears first."""
        results = compare_storage_temperatures("LNP-A", "mRNA", 6.5, [4.0, 25.0, -20.0])
        assert results[0].temperature_celsius == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_pka_zero_raises(self):
        with pytest.raises(ValueError, match="pka_lipid"):
            make_result(pka_lipid=0.0)

    def test_pka_negative_raises(self):
        with pytest.raises(ValueError, match="pka_lipid"):
            make_result(pka_lipid=-1.0)

    def test_diameter_zero_raises(self):
        with pytest.raises(ValueError, match="initial_diameter_nm"):
            make_result(initial_diameter_nm=0.0)

    def test_diameter_negative_raises(self):
        with pytest.raises(ValueError, match="initial_diameter_nm"):
            make_result(initial_diameter_nm=-10.0)

    def test_pdi_zero_raises(self):
        with pytest.raises(ValueError, match="initial_pdi"):
            make_result(initial_pdi=0.0)

    def test_pdi_one_raises(self):
        with pytest.raises(ValueError, match="initial_pdi"):
            make_result(initial_pdi=1.0)

    def test_pdi_negative_raises(self):
        with pytest.raises(ValueError, match="initial_pdi"):
            make_result(initial_pdi=-0.1)
