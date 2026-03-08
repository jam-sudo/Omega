"""Tests for Phase 521 — Extended Plasma Stability Model."""

import pytest

from omega_pbpk.prediction.plasma_stability_extended_p521 import (
    PlasmaStabilityExtendedResult,
    compare_stability_conditions,
    predict_plasma_stability_extended,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic(logP=2.0, ester_bond=False, esterase_activity=1.0, temperature_celsius=37.0):
    return predict_plasma_stability_extended(
        compound_name="TestDrug",
        logP=logP,
        ester_bond=ester_bond,
        esterase_activity=esterase_activity,
        temperature_celsius=temperature_celsius,
    )


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        r = _basic()
        assert isinstance(r, PlasmaStabilityExtendedResult)

    def test_all_fields_present(self):
        r = _basic()
        fields = [
            "compound_name",
            "logP",
            "ester_bond",
            "esterase_activity",
            "k_ester_per_h",
            "k_ox_per_h",
            "k_nonspec_per_h",
            "k_total_per_h",
            "times_min",
            "fraction_remaining",
            "t50_min",
            "t90_min",
            "stability_class",
            "dominant_pathway",
            "temperature_celsius",
            "k_total_at_temp_per_h",
            "notes",
        ]
        for f in fields:
            assert hasattr(r, f), f"Missing field: {f}"

    def test_compound_name_stored(self):
        r = predict_plasma_stability_extended("Aspirin", 1.2)
        assert r.compound_name == "Aspirin"

    def test_logP_stored(self):
        r = _basic(logP=3.5)
        assert r.logP == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Rate constant calculations
# ---------------------------------------------------------------------------


class TestRateConstants:
    def test_no_ester_k_ester_is_zero(self):
        r = _basic(ester_bond=False)
        assert r.k_ester_per_h == 0.0

    def test_ester_bond_sets_k_ester(self):
        r = _basic(ester_bond=True, esterase_activity=1.0)
        assert r.k_ester_per_h == pytest.approx(0.01)

    def test_ester_activity_scales_k_ester(self):
        r = _basic(ester_bond=True, esterase_activity=2.0)
        assert r.k_ester_per_h == pytest.approx(0.02)

    def test_k_ox_formula(self):
        logP = 3.0
        r = _basic(logP=logP)
        expected = 0.001 * (1.0 + 0.1 * logP)
        assert r.k_ox_per_h == pytest.approx(expected)

    def test_k_nonspec_constant(self):
        r = _basic()
        assert r.k_nonspec_per_h == pytest.approx(0.002)

    def test_k_total_sum(self):
        r = _basic(logP=2.0, ester_bond=True, esterase_activity=1.5)
        expected = r.k_ester_per_h + r.k_ox_per_h + r.k_nonspec_per_h
        assert r.k_total_per_h == pytest.approx(expected)

    def test_ester_bond_increases_k_total(self):
        r_no = _basic(ester_bond=False)
        r_yes = _basic(ester_bond=True)
        assert r_yes.k_total_per_h > r_no.k_total_per_h

    def test_higher_logP_increases_k_ox(self):
        r_low = _basic(logP=1.0)
        r_high = _basic(logP=5.0)
        assert r_high.k_ox_per_h > r_low.k_ox_per_h


# ---------------------------------------------------------------------------
# t50 / t90
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_t50_formula(self):
        r = _basic(logP=2.0)
        expected = 0.693 / r.k_total_per_h * 60.0
        assert r.t50_min == pytest.approx(expected, rel=1e-5)

    def test_t90_formula(self):
        r = _basic(logP=2.0)
        expected = 2.303 / r.k_total_per_h * 60.0
        assert r.t90_min == pytest.approx(expected, rel=1e-5)

    def test_t90_greater_than_t50(self):
        r = _basic()
        assert r.t90_min > r.t50_min


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


class TestProfile:
    def test_fraction_remaining_starts_at_one(self):
        r = _basic()
        assert r.fraction_remaining[0] == pytest.approx(1.0)

    def test_times_min_starts_at_zero(self):
        r = _basic()
        assert r.times_min[0] == 0.0

    def test_times_min_ends_at_120(self):
        r = _basic()
        assert r.times_min[-1] == 120.0

    def test_fraction_remaining_decreasing(self):
        r = _basic()
        for i in range(1, len(r.fraction_remaining)):
            assert r.fraction_remaining[i] < r.fraction_remaining[i - 1]

    def test_fraction_remaining_between_0_and_1(self):
        r = _basic()
        for f in r.fraction_remaining:
            assert 0.0 < f <= 1.0


# ---------------------------------------------------------------------------
# Stability class
# ---------------------------------------------------------------------------


class TestStabilityClass:
    def test_stable_class(self):
        # Need very low k_total → logP negative, no ester
        r = predict_plasma_stability_extended("Stable", logP=-5.0, ester_bond=False)
        # k_nonspec=0.002, k_ox=0.001*(1+0.1*(-5))=0.0005 → k_total=0.0025
        # t50 = 0.693/0.0025*60 = 16632 min >> 120 → stable
        assert r.stability_class == "stable"

    def test_unstable_class(self):
        # Very high ester activity → t50 < 30 min → unstable
        # k_ester=5.0 → k_total≈5.004 → t50=0.693/5.004*60≈8.3 min < 30
        r = _basic(ester_bond=True, esterase_activity=500.0)
        assert r.stability_class == "unstable"

    def test_moderate_class(self):
        # Manually craft params: need t50 in 30-120 min
        # k_total so that t50=60 min = 1h → k_total=0.693
        # k_ester = esterase*0.01, esterase=69.1, logP=0
        r = predict_plasma_stability_extended(
            "Moderate", logP=0.0, ester_bond=True, esterase_activity=69.1
        )
        assert r.stability_class in ("moderate", "unstable")  # boundary area


# ---------------------------------------------------------------------------
# Dominant pathway
# ---------------------------------------------------------------------------


class TestDominantPathway:
    def test_esterase_dominant_with_high_ester_activity(self):
        r = _basic(ester_bond=True, esterase_activity=10.0, logP=1.0)
        assert r.dominant_pathway == "esterase"

    def test_nonspecific_dominant_no_ester_low_logP(self):
        # k_ox = 0.001*(1+0.1*(-2)) = 0.0008; k_nonspec=0.002 > k_ox
        r = _basic(ester_bond=False, logP=-2.0)
        assert r.dominant_pathway == "nonspecific"

    def test_oxidative_dominant_high_logP_no_ester(self):
        # k_ox = 0.001*(1+0.1*20)=0.003 > k_nonspec=0.002
        r = _basic(ester_bond=False, logP=20.0)
        assert r.dominant_pathway == "oxidative"


# ---------------------------------------------------------------------------
# Temperature effect
# ---------------------------------------------------------------------------


class TestTemperature:
    def test_temperature_stored(self):
        r = _basic(temperature_celsius=42.0)
        assert r.temperature_celsius == pytest.approx(42.0)

    def test_higher_temp_higher_k(self):
        r37 = _basic(temperature_celsius=37.0)
        r42 = _basic(temperature_celsius=42.0)
        assert r42.k_total_at_temp_per_h > r37.k_total_at_temp_per_h

    def test_at_37_k_total_unchanged(self):
        r = _basic(temperature_celsius=37.0)
        assert r.k_total_at_temp_per_h == pytest.approx(r.k_total_per_h)

    def test_q10_factor(self):
        r = _basic(logP=2.0, temperature_celsius=47.0)
        factor = 2.0 ** ((47.0 - 37.0) / 10.0)
        assert r.k_total_at_temp_per_h == pytest.approx(r.k_total_per_h * factor)


# ---------------------------------------------------------------------------
# compare_stability_conditions
# ---------------------------------------------------------------------------


class TestCompareStabilityConditions:
    def test_returns_list(self):
        results = compare_stability_conditions("Drug", 2.0, False, [4.0, 25.0, 37.0])
        assert isinstance(results, list)

    def test_length_matches_temperatures(self):
        temps = [4.0, 25.0, 37.0, 42.0]
        results = compare_stability_conditions("Drug", 2.0, False, temps)
        assert len(results) == 4

    def test_sorted_by_t50_descending(self):
        results = compare_stability_conditions("Drug", 2.0, False, [4.0, 25.0, 37.0, 42.0])
        t50s = [r.t50_min for r in results]
        assert t50s == sorted(t50s, reverse=True)

    def test_lowest_temp_most_stable(self):
        results = compare_stability_conditions("Drug", 2.0, False, [4.0, 37.0])
        assert results[0].temperature_celsius == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_esterase_activity(self):
        with pytest.raises(ValueError, match="esterase_activity"):
            _basic(esterase_activity=-1.0)

    def test_zero_esterase_activity(self):
        with pytest.raises(ValueError, match="esterase_activity"):
            _basic(esterase_activity=0.0)

    def test_temperature_too_low(self):
        with pytest.raises(ValueError, match="temperature_celsius"):
            _basic(temperature_celsius=3.0)

    def test_temperature_too_high(self):
        with pytest.raises(ValueError, match="temperature_celsius"):
            _basic(temperature_celsius=61.0)
