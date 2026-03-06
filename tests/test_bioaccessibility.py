"""Tests for GI bioaccessibility (Phase 118)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.bioaccessibility import (
    BioaccessibilityResult,
    pH_solubility_profile,
    simulate_bioaccessibility,
)


def _default(**kw):
    defaults = dict(
        drug_name="Drug",
        dose_mg=100.0,
        intrinsic_solubility_mg_mL=1.0,
    )
    defaults.update(kw)
    return simulate_bioaccessibility(**defaults)


class TestBioaccessibility:
    def test_returns_result_type(self):
        assert isinstance(_default(), BioaccessibilityResult)

    def test_fa_total_between_0_and_1(self):
        r = _default()
        assert 0.0 <= r.fa_total <= 1.0

    def test_fa_intestine_between_0_and_1(self):
        r = _default()
        assert 0.0 <= r.fa_intestine <= 1.0

    def test_fa_stomach_between_0_and_1(self):
        r = _default()
        assert 0.0 <= r.fa_stomach <= 1.0

    def test_solubility_values_positive(self):
        r = _default()
        assert r.solubility_stomach_mg_mL > 0.0
        assert r.solubility_jejunum_mg_mL > 0.0

    def test_dose_number_positive(self):
        r = _default()
        assert r.dose_number > 0.0

    def test_high_dose_dose_limited(self):
        r = _default(dose_mg=10000.0, intrinsic_solubility_mg_mL=0.001)
        assert r.dose_limited is True

    def test_low_dose_not_dose_limited(self):
        r = _default(dose_mg=1.0, intrinsic_solubility_mg_mL=10.0)
        assert r.dose_limited is False

    def test_acid_higher_sol_at_high_pH(self):
        r = simulate_bioaccessibility("AcidicDrug", 100.0, 0.1, pka=4.5, drug_type="acid")
        assert r.solubility_ileum_mg_mL > r.solubility_stomach_mg_mL

    def test_base_higher_sol_at_low_pH(self):
        r = simulate_bioaccessibility("BasicDrug", 100.0, 0.1, pka=9.0, drug_type="base")
        assert r.solubility_stomach_mg_mL > r.solubility_jejunum_mg_mL

    def test_base_has_precipitation_risk(self):
        r = simulate_bioaccessibility("BasicDrug", 100.0, 0.001, pka=10.0, drug_type="base")
        assert r.precipitation_risk in ("moderate", "high")

    def test_precipitation_risk_valid(self):
        r = _default()
        assert r.precipitation_risk in ("none", "low", "moderate", "high")

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError):
            _default(intrinsic_solubility_mg_mL=0.0)


class TestPHSolubilityProfile:
    def test_returns_dict_with_keys(self):
        result = pH_solubility_profile("Drug", 1.0, pka=5.0, drug_type="acid")
        assert "pH" in result
        assert "solubility_mg_mL" in result
        assert "logS" in result

    def test_acid_increasing_with_pH(self):
        result = pH_solubility_profile("Acid", 0.1, pka=5.0, drug_type="acid")
        sols = result["solubility_mg_mL"]
        # Last pH (8.0) should have higher solubility than first (1.0)
        assert sols[-1] > sols[0]

    def test_neutral_constant_solubility(self):
        result = pH_solubility_profile("Neutral", 1.0, drug_type="neutral")
        sols = result["solubility_mg_mL"]
        assert abs(sols[-1] - sols[0]) < 0.01
