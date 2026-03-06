"""Tests for supersaturation and precipitation kinetics (Phase 110)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.supersaturation import (
    SupersaturationResult,
    compare_formulations_supersaturation,
    simulate_supersaturation,
)


def _default(**kw):
    defaults = dict(
        drug_name="BCS-II Drug",
        dose_mg=100.0,
        volume_gut_mL=250.0,
        solubility_mg_mL=0.01,
        logP=3.5,
        t_end_h=6.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_supersaturation(**defaults)


class TestSupersaturationResult:
    def test_returns_result_type(self):
        assert isinstance(_default(), SupersaturationResult)

    def test_conc_dissolved_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.conc_dissolved_mg_mL)

    def test_conc_precipitate_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.conc_precipitate_mg_mL)

    def test_times_and_conc_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.conc_dissolved_mg_mL) == len(r.conc_precipitate_mg_mL)

    def test_auc_dissolved_positive(self):
        r = _default()
        assert r.auc_dissolved > 0.0

    def test_supersaturation_ratio_above_1_for_amorphous(self):
        r = _default(logP=4.0)
        # Amorphous solubility >> crystalline → supersaturation at t=0
        assert r.supersaturation_ratio >= 1.0

    def test_precipitation_complete_is_bool(self):
        r = _default()
        assert isinstance(r.precipitation_complete, bool)

    def test_t50_within_simulation_time(self):
        r = _default()
        assert 0.0 <= r.t50_precipitation_h <= 6.0

    def test_polymer_parachute_increases_auc(self):
        r_no_poly = _default(polymer_parachute=False)
        r_poly = _default(polymer_parachute=True)
        assert r_poly.spring_and_parachute_auc >= r_no_poly.auc_dissolved

    def test_amorphous_sol_geq_crystalline(self):
        r = _default()
        assert r.solubility_amorphous_mg_mL >= r.solubility_crystalline_mg_mL


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            _default(solubility_mg_mL=0.0)

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError):
            _default(volume_gut_mL=0.0)


class TestCompareFormulations:
    def test_returns_dict_with_three_keys(self):
        result = compare_formulations_supersaturation("Drug", 100.0, 0.01)
        assert "crystalline" in result
        assert "amorphous" in result
        assert "amorphous_polymer" in result

    def test_all_result_types(self):
        result = compare_formulations_supersaturation("Drug", 100.0, 0.01)
        for v in result.values():
            assert isinstance(v, SupersaturationResult)

    def test_amorphous_higher_auc_than_crystalline(self):
        result = compare_formulations_supersaturation("Drug", 100.0, 0.01, logP=4.0)
        assert result["amorphous"].auc_dissolved >= result["crystalline"].auc_dissolved - 1e-6
