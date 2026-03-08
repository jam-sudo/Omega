"""Tests for drug nanotoxicology prediction module (Phase 1022)."""

import pytest

from omega_pbpk.prediction.drug_nanotoxicology import (
    NanotoxicologyResult,
    compare_nanoparticles,
    predict_nanotoxicology,
)

_VALID_RISK_CATEGORIES = {"low", "moderate", "high", "very_high"}


def _make(**kwargs):
    params = dict(
        particle_name="TestNP",
        size_nm=50.0,
        surface_charge_mV=0.0,
        material="gold",
        surface_coating="none",
        concentration_ug_mL=100.0,
    )
    params.update(kwargs)
    return predict_nanotoxicology(**params)


class TestReturnType:
    def test_returns_frozen_dataclass(self):
        result = _make()
        assert isinstance(result, NanotoxicologyResult)

    def test_is_frozen(self):
        result = _make()
        with pytest.raises((AttributeError, TypeError)):
            result.cytotoxicity_score = 50.0  # type: ignore[misc]


class TestScoreBounds:
    def test_cytotoxicity_score_in_range(self):
        result = _make()
        assert 0.0 <= result.cytotoxicity_score <= 100.0

    def test_oxidative_stress_score_in_range(self):
        result = _make()
        assert 0.0 <= result.oxidative_stress_score <= 100.0

    def test_inflammatory_score_in_range(self):
        result = _make()
        assert 0.0 <= result.inflammatory_score <= 100.0

    def test_genotoxicity_score_in_range(self):
        result = _make()
        assert 0.0 <= result.genotoxicity_score <= 100.0

    def test_overall_risk_score_in_range(self):
        result = _make()
        assert 0.0 <= result.overall_risk_score <= 100.0

    def test_scores_clamped_at_100_high_risk(self):
        result = _make(material="silver", size_nm=5.0, concentration_ug_mL=5000.0)
        assert result.cytotoxicity_score <= 100.0
        assert result.overall_risk_score <= 100.0


class TestRiskCategory:
    def test_risk_category_is_valid_string(self):
        result = _make()
        assert result.risk_category in _VALID_RISK_CATEGORIES

    def test_low_risk_gold_large(self):
        result = _make(material="gold", size_nm=500.0, concentration_ug_mL=1.0)
        # gold + large + low conc should be low/moderate
        assert result.risk_category in {"low", "moderate"}

    def test_high_risk_silver_small(self):
        result = _make(material="silver", size_nm=5.0, concentration_ug_mL=1000.0)
        assert result.risk_category in {"high", "very_high"}


class TestMaterialComparison:
    def test_silver_higher_cytotoxicity_than_gold(self):
        silver = _make(material="silver", size_nm=50.0, surface_charge_mV=0.0)
        gold = _make(material="gold", size_nm=50.0, surface_charge_mV=0.0)
        assert silver.cytotoxicity_score > gold.cytotoxicity_score

    def test_silver_higher_overall_than_polymer(self):
        silver = _make(material="silver")
        polymer = _make(material="polymer")
        assert silver.overall_risk_score > polymer.overall_risk_score


class TestSizeEffect:
    def test_small_particles_higher_genotoxicity(self):
        small = _make(material="silver", size_nm=5.0)
        large = _make(material="silver", size_nm=200.0)
        assert small.genotoxicity_score > large.genotoxicity_score

    def test_small_particles_higher_cytotoxicity(self):
        small = _make(material="gold", size_nm=5.0)
        large = _make(material="gold", size_nm=100.0)
        assert small.cytotoxicity_score > large.cytotoxicity_score


class TestCoatingEffect:
    def test_peg_reduces_cytotoxicity(self):
        peg = _make(material="silver", surface_coating="PEG")
        none = _make(material="silver", surface_coating="none")
        assert peg.cytotoxicity_score < none.cytotoxicity_score

    def test_amine_increases_genotoxicity(self):
        amine = _make(material="gold", surface_coating="amine")
        none_coat = _make(material="gold", surface_coating="none")
        assert amine.genotoxicity_score > none_coat.genotoxicity_score


class TestConcentrationEffect:
    def test_high_concentration_increases_cytotoxicity(self):
        low = _make(concentration_ug_mL=10.0)
        high = _make(concentration_ug_mL=1000.0)
        assert high.cytotoxicity_score > low.cytotoxicity_score


class TestCompareNanoparticles:
    def _sample_particles(self):
        return [
            {
                "particle_name": "SilverNP",
                "size_nm": 10.0,
                "surface_charge_mV": 20.0,
                "material": "silver",
            },
            {
                "particle_name": "GoldNP",
                "size_nm": 100.0,
                "surface_charge_mV": 0.0,
                "material": "gold",
            },
            {
                "particle_name": "PolymerNP",
                "size_nm": 200.0,
                "surface_charge_mV": -5.0,
                "material": "polymer",
            },
        ]

    def test_returns_list(self):
        results = compare_nanoparticles(self._sample_particles())
        assert isinstance(results, list)

    def test_length_matches_input(self):
        particles = self._sample_particles()
        results = compare_nanoparticles(particles)
        assert len(results) == len(particles)

    def test_sorted_ascending_by_overall_risk(self):
        results = compare_nanoparticles(self._sample_particles())
        scores = [r.overall_risk_score for r in results]
        assert scores == sorted(scores)

    def test_all_results_are_nanotoxicology_result(self):
        results = compare_nanoparticles(self._sample_particles())
        for r in results:
            assert isinstance(r, NanotoxicologyResult)


class TestValidation:
    def test_size_zero_raises(self):
        with pytest.raises(ValueError, match="size_nm"):
            _make(size_nm=0.0)

    def test_size_negative_raises(self):
        with pytest.raises(ValueError):
            _make(size_nm=-10.0)

    def test_size_over_1000_raises(self):
        with pytest.raises(ValueError):
            _make(size_nm=1001.0)

    def test_invalid_material_raises(self):
        with pytest.raises(ValueError, match="material"):
            _make(material="titanium")

    def test_invalid_coating_raises(self):
        with pytest.raises(ValueError, match="surface_coating"):
            _make(surface_coating="chitosan")

    def test_zero_concentration_raises(self):
        with pytest.raises(ValueError, match="concentration_ug_mL"):
            _make(concentration_ug_mL=0.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError):
            _make(concentration_ug_mL=-5.0)
