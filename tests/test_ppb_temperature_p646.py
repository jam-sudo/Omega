"""Tests for Phase 646 — Protein Binding Temperature Dependence."""

import pytest

from omega_pbpk.prediction.ppb_temperature_p646 import (
    PPBTemperatureResult,
    compare_temperatures,
    predict_ppb_temperature,
)

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULTS = dict(
    drug_name="TestDrug",
    logP=2.0,
    mw_Da=300.0,
    reference_fup=0.1,
    reference_temp_C=37.0,
    target_temp_C=25.0,
)


def _run(**overrides):
    kw = {**_DEFAULTS, **overrides}
    return predict_ppb_temperature(**kw)


# ── Return type ──────────────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, PPBTemperatureResult)

    def test_frozen(self):
        r = _run()
        with pytest.raises(AttributeError):
            r.drug_name = "Other"

    def test_frozen_mutation_fup(self):
        r = _run()
        with pytest.raises(AttributeError):
            r.fup_at_temp = 0.5

    def test_drug_name_stored(self):
        r = _run(drug_name="Warfarin")
        assert r.drug_name == "Warfarin"


# ── Temperature effects (exothermic binding) ─────────────────────────────────


class TestTemperatureEffects:
    def test_warming_increases_fup(self):
        """For exothermic binding, warming decreases K → higher fup."""
        r = _run(reference_temp_C=37.0, target_temp_C=45.0)
        assert r.fup_at_temp > r.reference_fup

    def test_cooling_decreases_fup(self):
        """For exothermic binding, cooling increases K → lower fup."""
        r = _run(reference_temp_C=37.0, target_temp_C=25.0)
        assert r.fup_at_temp < r.reference_fup

    def test_same_temp_returns_same_fup(self):
        r = _run(reference_temp_C=37.0, target_temp_C=37.0)
        assert abs(r.fup_at_temp - r.reference_fup) < 1e-6

    def test_large_warming_fup_increases_more(self):
        r_small = _run(target_temp_C=40.0)
        r_large = _run(target_temp_C=50.0)
        assert r_large.fup_at_temp > r_small.fup_at_temp


# ── fup bounds ───────────────────────────────────────────────────────────────


class TestFupBounds:
    def test_fup_in_range(self):
        r = _run()
        assert 0.0 < r.fup_at_temp < 1.0

    def test_fup_at_extreme_cold(self):
        r = _run(target_temp_C=-20.0)
        assert r.fup_at_temp >= 0.001

    def test_fup_at_extreme_hot(self):
        r = _run(target_temp_C=80.0)
        assert r.fup_at_temp <= 0.999


# ── delta_fup_pct sign ──────────────────────────────────────────────────────


class TestDeltaFupPct:
    def test_positive_when_warming(self):
        r = _run(target_temp_C=45.0)
        assert r.delta_fup_pct > 0.0

    def test_negative_when_cooling(self):
        r = _run(target_temp_C=25.0)
        assert r.delta_fup_pct < 0.0

    def test_zero_at_reference(self):
        r = _run(target_temp_C=37.0)
        assert abs(r.delta_fup_pct) < 1e-6


# ── Van't Hoff factor ───────────────────────────────────────────────────────


class TestVantHoffFactor:
    def test_warming_decreases_k(self):
        """For exothermic binding, warming → K2 < K1 → factor < 1."""
        r = _run(target_temp_C=45.0)
        assert r.van_t_hoff_factor < 1.0

    def test_cooling_increases_k(self):
        """For exothermic binding, cooling → K2 > K1 → factor > 1."""
        r = _run(target_temp_C=25.0)
        assert r.van_t_hoff_factor > 1.0

    def test_same_temp_factor_one(self):
        r = _run(target_temp_C=37.0)
        assert abs(r.van_t_hoff_factor - 1.0) < 1e-6


# ── Protein conformation factor ─────────────────────────────────────────────


class TestConformationFactor:
    def test_normal_range(self):
        r = _run(target_temp_C=30.0)
        assert r.protein_conformation_factor == 1.0

    def test_cold_denaturation(self):
        r = _run(target_temp_C=10.0)
        assert r.protein_conformation_factor == 0.85

    def test_heat_denaturation(self):
        r = _run(target_temp_C=45.0)
        assert r.protein_conformation_factor == 0.90


# ── Clinical relevance ──────────────────────────────────────────────────────


class TestClinicalRelevance:
    def test_minor(self):
        r = _run(target_temp_C=37.0)
        assert r.clinical_relevance == "minor"

    def test_significant_exists(self):
        # Large temperature change should produce significant
        r = _run(target_temp_C=5.0, logP=4.0, reference_fup=0.05)
        assert r.clinical_relevance in {"significant", "moderate"}


# ── Binding enthalpy ────────────────────────────────────────────────────────


class TestBindingEnthalpy:
    def test_enthalpy_negative(self):
        r = _run(logP=2.0)
        assert r.binding_enthalpy_kJ_mol < 0.0

    def test_enthalpy_clamped(self):
        r = _run(logP=100.0)
        assert r.binding_enthalpy_kJ_mol >= -60.0

    def test_enthalpy_clamped_low_logp(self):
        r = _run(logP=-100.0)
        assert r.binding_enthalpy_kJ_mol <= -5.0


# ── compare_temperatures ────────────────────────────────────────────────────


class TestCompareTemperatures:
    def test_returns_sorted_list(self):
        results = compare_temperatures("Drug", 2.0, 300.0, 0.1, [40.0, 20.0, 30.0])
        temps = [r.temp_C for r in results]
        assert temps == sorted(temps)

    def test_length_matches(self):
        temps_in = [10.0, 20.0, 30.0, 40.0, 50.0]
        results = compare_temperatures("Drug", 2.0, 300.0, 0.1, temps_in)
        assert len(results) == len(temps_in)

    def test_all_ppb_result(self):
        results = compare_temperatures("Drug", 2.0, 300.0, 0.1, [25.0, 37.0])
        for r in results:
            assert isinstance(r, PPBTemperatureResult)


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidation:
    def test_fup_zero_raises(self):
        with pytest.raises(ValueError, match="reference_fup"):
            _run(reference_fup=0.0)

    def test_fup_one_raises(self):
        with pytest.raises(ValueError, match="reference_fup"):
            _run(reference_fup=1.0)

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError, match="reference_fup"):
            _run(reference_fup=-0.1)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=-100.0)
