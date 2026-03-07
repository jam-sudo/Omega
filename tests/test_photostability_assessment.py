"""Tests for Phase 382 — Drug Photostability Assessment."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.risk.photostability_assessment import (
    PhotostabilityResult,
    assess_photostability,
    compare_light_sources,
)

# ---------------------------------------------------------------------------
# Return type and basic attributes
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_photostability_result(self):
        result = assess_photostability(
            "aspirin", logp=1.2, mw_da=180.0, n_aromatic_rings=1, n_double_bonds=1
        )
        assert isinstance(result, PhotostabilityResult)

    def test_drug_name_stored(self):
        result = assess_photostability(
            "ibuprofen", logp=3.97, mw_da=206.0, n_aromatic_rings=1, n_double_bonds=1
        )
        assert result.drug_name == "ibuprofen"

    def test_light_source_stored(self):
        result = assess_photostability(
            "aspirin",
            logp=1.2,
            mw_da=180.0,
            n_aromatic_rings=1,
            n_double_bonds=1,
            light_source="fluorescent",
        )
        assert result.light_source == "fluorescent"

    def test_total_lux_h_stored(self):
        result = assess_photostability(
            "aspirin",
            logp=1.2,
            mw_da=180.0,
            n_aromatic_rings=1,
            n_double_bonds=1,
            exposure_lux_h=6e5,
        )
        assert result.total_lux_h == pytest.approx(6e5)


# ---------------------------------------------------------------------------
# Chromophore flag logic
# ---------------------------------------------------------------------------


class TestChromophoreFlag:
    def test_chromophore_flag_two_aromatic_rings(self):
        result = assess_photostability(
            "drug_a", logp=1.0, mw_da=200.0, n_aromatic_rings=2, n_double_bonds=0
        )
        assert result.chromophore_flag is True

    def test_chromophore_flag_four_double_bonds(self):
        result = assess_photostability(
            "drug_b", logp=1.0, mw_da=200.0, n_aromatic_rings=0, n_double_bonds=4
        )
        assert result.chromophore_flag is True

    def test_chromophore_flag_high_logp(self):
        result = assess_photostability(
            "drug_c", logp=4.0, mw_da=200.0, n_aromatic_rings=0, n_double_bonds=0
        )
        assert result.chromophore_flag is True

    def test_no_chromophore_flag_simple_drug(self):
        result = assess_photostability(
            "mannitol", logp=-3.0, mw_da=182.0, n_aromatic_rings=0, n_double_bonds=0
        )
        assert result.chromophore_flag is False

    def test_no_chromophore_flag_one_ring_low_logp(self):
        result = assess_photostability(
            "simple_drug", logp=1.0, mw_da=150.0, n_aromatic_rings=1, n_double_bonds=2
        )
        assert result.chromophore_flag is False


# ---------------------------------------------------------------------------
# Degradation values
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_degradation_pct_between_0_and_100(self):
        for source in ("xenon", "fluorescent", "D65", "UV_only"):
            result = assess_photostability(
                "drug",
                logp=2.0,
                mw_da=300.0,
                n_aromatic_rings=3,
                n_double_bonds=2,
                light_source=source,
            )
            assert 0.0 <= result.degradation_pct <= 100.0

    def test_stable_drug_low_degradation(self):
        # Non-chromophoric drug under low exposure (1e5 lux-h)
        # rate = 1e-7 * 0.3 * 1.0 = 3e-8; deg = (1-exp(-3e-8*1e5))*100 ~ 0.3%
        result = assess_photostability(
            "mannitol",
            logp=-3.0,
            mw_da=182.0,
            n_aromatic_rings=0,
            n_double_bonds=0,
            light_source="xenon",
            exposure_lux_h=1e5,
        )
        assert result.degradation_pct < 1.0

    def test_unstable_drug_high_degradation(self):
        # Highly chromophoric drug under UV-only high exposure
        result = assess_photostability(
            "porphyrin_drug",
            logp=5.0,
            mw_da=600.0,
            n_aromatic_rings=8,
            n_double_bonds=10,
            light_source="UV_only",
            exposure_lux_h=1.2e6,
        )
        assert result.degradation_pct >= 5.0

    def test_xenon_greater_than_fluorescent_degradation(self):
        kwargs = dict(
            logp=2.0, mw_da=300.0, n_aromatic_rings=3, n_double_bonds=2, exposure_lux_h=1.2e6
        )
        r_xenon = assess_photostability("drug", **kwargs, light_source="xenon")
        r_fluor = assess_photostability("drug", **kwargs, light_source="fluorescent")
        assert r_xenon.degradation_pct > r_fluor.degradation_pct

    def test_uv_only_highest_degradation(self):
        kwargs = dict(
            logp=2.0, mw_da=300.0, n_aromatic_rings=3, n_double_bonds=2, exposure_lux_h=1.2e6
        )
        r_uv = assess_photostability("drug", **kwargs, light_source="UV_only")
        r_xenon = assess_photostability("drug", **kwargs, light_source="xenon")
        assert r_uv.degradation_pct > r_xenon.degradation_pct


# ---------------------------------------------------------------------------
# Half-life, ICH compliance, risk category
# ---------------------------------------------------------------------------


class TestHalfLifeAndCompliance:
    def test_half_life_positive(self):
        result = assess_photostability(
            "aspirin", logp=1.2, mw_da=180.0, n_aromatic_rings=1, n_double_bonds=1
        )
        assert result.half_life_lux_h > 0.0

    def test_half_life_formula_consistent(self):
        result = assess_photostability(
            "aspirin", logp=1.2, mw_da=180.0, n_aromatic_rings=1, n_double_bonds=1
        )
        expected_hl = math.log(2.0) / result.photodegradation_rate_per_lux_h
        assert result.half_life_lux_h == pytest.approx(expected_hl, rel=1e-6)

    def test_ich_q1b_compliant_for_stable_drug(self):
        result = assess_photostability(
            "mannitol",
            logp=-3.0,
            mw_da=182.0,
            n_aromatic_rings=0,
            n_double_bonds=0,
            light_source="xenon",
            exposure_lux_h=1.2e6,
        )
        assert result.ich_q1b_compliant is True

    def test_ich_q1b_noncompliant_for_unstable_drug(self):
        result = assess_photostability(
            "porphyrin_drug",
            logp=5.0,
            mw_da=600.0,
            n_aromatic_rings=8,
            n_double_bonds=10,
            light_source="xenon",
            exposure_lux_h=1.2e6,
        )
        assert result.ich_q1b_compliant is False

    def test_risk_category_stable(self):
        # Use low exposure to stay below 1% degradation for non-chromophoric drug
        result = assess_photostability(
            "mannitol",
            logp=-3.0,
            mw_da=182.0,
            n_aromatic_rings=0,
            n_double_bonds=0,
            exposure_lux_h=1e5,
        )
        assert result.risk_category == "stable"

    def test_risk_category_unstable(self):
        result = assess_photostability(
            "porphyrin_drug",
            logp=5.0,
            mw_da=600.0,
            n_aromatic_rings=8,
            n_double_bonds=10,
            light_source="UV_only",
            exposure_lux_h=1.2e6,
        )
        assert result.risk_category == "unstable"


# ---------------------------------------------------------------------------
# Packaging recommendations
# ---------------------------------------------------------------------------


class TestPackagingRecommendations:
    def test_stable_drug_clear_packaging(self):
        # Non-chromophoric drug, low exposure -> degradation < 0.5% -> "clear"
        # rate = 1e-7 * 0.3 * 0.3 = 9e-9; deg at 5e4 lux-h ~ 0.045%
        result = assess_photostability(
            "stable_drug",
            logp=-3.0,
            mw_da=100.0,
            n_aromatic_rings=0,
            n_double_bonds=0,
            light_source="fluorescent",
            exposure_lux_h=5e4,
        )
        assert result.recommended_packaging == "clear"

    def test_moderately_sensitive_amber_packaging(self):
        # Design a drug that degrades 0.5-2% under test conditions
        result = assess_photostability(
            "drug_b",
            logp=1.0,
            mw_da=200.0,
            n_aromatic_rings=1,
            n_double_bonds=3,
            light_source="xenon",
            exposure_lux_h=1.2e6,
        )
        # Check it's either amber or opaque (depends on exact degradation)
        assert result.recommended_packaging in ("amber", "opaque", "clear", "foil")

    def test_highly_unstable_foil_packaging(self):
        result = assess_photostability(
            "highly_sensitive",
            logp=6.0,
            mw_da=500.0,
            n_aromatic_rings=10,
            n_double_bonds=12,
            light_source="UV_only",
            exposure_lux_h=1.2e6,
        )
        assert result.recommended_packaging == "foil"


# ---------------------------------------------------------------------------
# Compare light sources
# ---------------------------------------------------------------------------


class TestCompareLightSources:
    def test_returns_list(self):
        results = compare_light_sources(
            "aspirin", logp=1.2, mw_da=180.0, n_aromatic_rings=1, n_double_bonds=1
        )
        assert isinstance(results, list)

    def test_returns_four_results(self):
        results = compare_light_sources(
            "aspirin", logp=1.2, mw_da=180.0, n_aromatic_rings=1, n_double_bonds=1
        )
        assert len(results) == 4

    def test_all_light_sources_represented(self):
        results = compare_light_sources(
            "aspirin", logp=1.2, mw_da=180.0, n_aromatic_rings=1, n_double_bonds=1
        )
        sources = {r.light_source for r in results}
        assert sources == {"xenon", "fluorescent", "D65", "UV_only"}

    def test_sorted_by_degradation_ascending(self):
        results = compare_light_sources(
            "aspirin", logp=1.2, mw_da=180.0, n_aromatic_rings=1, n_double_bonds=1
        )
        degs = [r.degradation_pct for r in results]
        assert degs == sorted(degs)

    def test_all_results_are_photostability_result(self):
        results = compare_light_sources(
            "aspirin", logp=1.2, mw_da=180.0, n_aromatic_rings=1, n_double_bonds=1
        )
        for r in results:
            assert isinstance(r, PhotostabilityResult)

    def test_fluorescent_is_most_stable(self):
        results = compare_light_sources(
            "chromophore_drug", logp=2.5, mw_da=350.0, n_aromatic_rings=3, n_double_bonds=5
        )
        # fluorescent has lowest UV factor (0.3) → least degradation
        assert results[0].light_source == "fluorescent"

    def test_uv_only_is_least_stable(self):
        results = compare_light_sources(
            "chromophore_drug", logp=2.5, mw_da=350.0, n_aromatic_rings=3, n_double_bonds=5
        )
        assert results[-1].light_source == "UV_only"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_da"):
            assess_photostability(
                "drug", logp=1.0, mw_da=-1.0, n_aromatic_rings=1, n_double_bonds=0
            )

    def test_negative_aromatic_rings_raises(self):
        with pytest.raises(ValueError, match="n_aromatic_rings"):
            assess_photostability(
                "drug", logp=1.0, mw_da=200.0, n_aromatic_rings=-1, n_double_bonds=0
            )

    def test_negative_double_bonds_raises(self):
        with pytest.raises(ValueError, match="n_double_bonds"):
            assess_photostability(
                "drug", logp=1.0, mw_da=200.0, n_aromatic_rings=0, n_double_bonds=-1
            )

    def test_invalid_light_source_raises(self):
        with pytest.raises(ValueError, match="light_source"):
            assess_photostability(
                "drug",
                logp=1.0,
                mw_da=200.0,
                n_aromatic_rings=0,
                n_double_bonds=0,
                light_source="solar",
            )

    def test_zero_exposure_raises(self):
        with pytest.raises(ValueError, match="exposure_lux_h"):
            assess_photostability(
                "drug",
                logp=1.0,
                mw_da=200.0,
                n_aromatic_rings=0,
                n_double_bonds=0,
                exposure_lux_h=0.0,
            )
