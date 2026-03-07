"""Tests for Phase 351 — Spray Drying Process Parameter Predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.spray_drying_predictor import (
    SprayDryingPredictionResult,
    optimize_drug_load,
    predict_spray_drying,
)

# Shared baseline kwargs for convenience
_BASE = dict(
    drug_name="TestDrug",
    drug_load_pct=30.0,
    polymer_type="HPMC-AS",
    tg_drug_C=80.0,
    inlet_temp_C=120.0,
    feed_conc_mg_mL=20.0,
    solvent="ethanol",
    drug_logP=2.5,
)


def _predict(**kwargs):
    params = {**_BASE, **kwargs}
    return predict_spray_drying(**params)


# ── Return type ────────────────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_frozen_dataclass(self):
        result = _predict()
        assert isinstance(result, SprayDryingPredictionResult)

    def test_result_is_immutable(self):
        result = _predict()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Changed"  # type: ignore[misc]

    def test_fields_present(self):
        result = _predict()
        assert hasattr(result, "tg_asd_C")
        assert hasattr(result, "outlet_temp_C")
        assert hasattr(result, "stability_margin_C")
        assert hasattr(result, "residual_solvent_ppm")
        assert hasattr(result, "particle_d50_um")
        assert hasattr(result, "process_yield_pct")
        assert hasattr(result, "crystallization_risk")
        assert hasattr(result, "ice_formation_risk")


# ── Gordon-Taylor Tg_ASD ──────────────────────────────────────────────────────


class TestGordonTaylorTg:
    def test_tg_asd_between_drug_and_polymer(self):
        """Tg_ASD must lie between tg_drug and tg_polymer for typical k_gt."""
        result = _predict(tg_drug_C=60.0, polymer_type="HPMC-AS", drug_load_pct=30.0)
        # HPMC-AS Tg = 120 C, drug Tg = 60 C → Tg_ASD between 60 and 120
        assert 60.0 <= result.tg_asd_C <= 120.0

    def test_tg_asd_lower_at_high_drug_load(self):
        """Higher drug load (lower Tg drug) → lower Tg_ASD."""
        result_low = _predict(tg_drug_C=60.0, drug_load_pct=20.0)
        result_high = _predict(tg_drug_C=60.0, drug_load_pct=70.0)
        # More low-Tg drug → Tg_ASD drops
        assert result_high.tg_asd_C < result_low.tg_asd_C

    def test_tg_asd_at_100pct_drug_equals_drug_tg(self):
        """At 100% drug load, Tg_ASD should approach tg_drug."""
        # 80% is max allowed; use 80 and verify direction
        result = _predict(tg_drug_C=50.0, drug_load_pct=80.0, polymer_type="HPMC-AS")
        # w1=0.8, w2=0.2; Tg_ASD should be closer to 50 than to 120
        assert abs(result.tg_asd_C - 50.0) < abs(result.tg_asd_C - 120.0)

    def test_different_polymers_give_different_tg(self):
        result_hpmc = _predict(polymer_type="HPMC-AS", drug_load_pct=50.0)
        result_pvpva = _predict(polymer_type="PVP-VA", drug_load_pct=50.0)
        # HPMC-AS Tg=120, PVP-VA Tg=106 → HPMC-AS blend should have higher Tg
        assert result_hpmc.tg_asd_C > result_pvpva.tg_asd_C


# ── Outlet Temperature and Stability Margin ───────────────────────────────────


class TestOutletAndMargin:
    def test_outlet_temp_formula(self):
        result = _predict(inlet_temp_C=100.0)
        expected_outlet = 100.0 * 0.6 - 10.0
        assert abs(result.outlet_temp_C - expected_outlet) < 1e-6

    def test_stability_margin_equals_tg_minus_outlet(self):
        result = _predict()
        expected_margin = result.tg_asd_C - result.outlet_temp_C
        assert abs(result.stability_margin_C - expected_margin) < 1e-6

    def test_higher_inlet_increases_outlet(self):
        result_low = _predict(inlet_temp_C=100.0)
        result_high = _predict(inlet_temp_C=180.0)
        assert result_high.outlet_temp_C > result_low.outlet_temp_C

    def test_ice_formation_false_for_normal_inlet(self):
        result = _predict(inlet_temp_C=120.0)
        assert result.ice_formation_risk is False

    def test_ice_formation_true_when_outlet_below_zero(self):
        # outlet = inlet * 0.6 - 10; outlet < 0 when inlet < 16.67
        result = _predict(inlet_temp_C=10.0)
        assert result.ice_formation_risk is True


# ── Residual Solvent ──────────────────────────────────────────────────────────


class TestResidualSolvent:
    def test_higher_inlet_lower_residual(self):
        result_low = _predict(inlet_temp_C=80.0)
        result_high = _predict(inlet_temp_C=160.0)
        assert result_high.residual_solvent_ppm < result_low.residual_solvent_ppm

    def test_residual_minimum_100ppm(self):
        result = _predict(inlet_temp_C=300.0)
        assert result.residual_solvent_ppm >= 100.0

    def test_residual_non_negative(self):
        result = _predict()
        assert result.residual_solvent_ppm >= 0.0


# ── Yield ─────────────────────────────────────────────────────────────────────


class TestYield:
    def test_yield_in_range(self):
        result = _predict()
        assert 0.0 <= result.process_yield_pct <= 100.0

    def test_yield_max_90pct(self):
        result = _predict(inlet_temp_C=300.0)
        assert result.process_yield_pct <= 90.0

    def test_higher_inlet_higher_yield(self):
        result_low = _predict(inlet_temp_C=80.0)
        result_high = _predict(inlet_temp_C=150.0)
        assert result_high.process_yield_pct >= result_low.process_yield_pct


# ── Crystallization Risk ──────────────────────────────────────────────────────


class TestCrystallizationRisk:
    def test_valid_risk_values(self):
        valid = {"low", "moderate", "high"}
        for polymer in ("HPMC-AS", "PVP-VA", "HPMC", "Eudragit"):
            result = _predict(polymer_type=polymer)
            assert result.crystallization_risk in valid

    def test_low_risk_when_margin_above_20(self):
        # Use drug with high tg_drug so margin is large
        result = _predict(tg_drug_C=200.0, inlet_temp_C=80.0, polymer_type="HPMC-AS")
        # outlet = 80*0.6-10=38; tg_asd >> 38 → margin > 20
        assert result.crystallization_risk == "low"

    def test_high_risk_when_margin_below_10(self):
        # Very high inlet pushes outlet up, very low tg_drug lowers tg_asd
        result = _predict(
            tg_drug_C=30.0, inlet_temp_C=200.0, drug_load_pct=80.0, polymer_type="PVP-VA"
        )
        # outlet = 200*0.6-10=110; tg_asd low → margin < 10
        assert result.crystallization_risk == "high"


# ── optimize_drug_load ────────────────────────────────────────────────────────


class TestOptimizeDrugLoad:
    def test_returns_list(self):
        results = optimize_drug_load(
            drug_name="DrugX",
            polymer_type="HPMC-AS",
            tg_drug_C=80.0,
            inlet_temp_C=120.0,
            feed_conc=20.0,
            solvent="ethanol",
            logP=2.0,
            drug_loads=[20.0, 40.0, 60.0],
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_stability_margin_descending(self):
        results = optimize_drug_load(
            drug_name="DrugX",
            polymer_type="HPMC-AS",
            tg_drug_C=80.0,
            inlet_temp_C=120.0,
            feed_conc=20.0,
            solvent="ethanol",
            logP=2.0,
            drug_loads=[20.0, 40.0, 60.0, 80.0],
        )
        margins = [r.stability_margin_C for r in results]
        assert margins == sorted(margins, reverse=True)

    def test_returns_all_types_correctly(self):
        results = optimize_drug_load(
            drug_name="DrugY",
            polymer_type="PVP-VA",
            tg_drug_C=100.0,
            inlet_temp_C=130.0,
            feed_conc=15.0,
            solvent="acetone",
            logP=1.5,
            drug_loads=[10.0, 50.0],
        )
        for r in results:
            assert isinstance(r, SprayDryingPredictionResult)


# ── Validation Errors ─────────────────────────────────────────────────────────


class TestValidationErrors:
    def test_drug_load_zero_raises(self):
        with pytest.raises(ValueError, match="drug_load_pct"):
            _predict(drug_load_pct=0.0)

    def test_drug_load_above_80_raises(self):
        with pytest.raises(ValueError, match="drug_load_pct"):
            _predict(drug_load_pct=81.0)

    def test_negative_drug_load_raises(self):
        with pytest.raises(ValueError, match="drug_load_pct"):
            _predict(drug_load_pct=-5.0)

    def test_invalid_polymer_raises(self):
        with pytest.raises(ValueError, match="polymer_type"):
            _predict(polymer_type="Unknown")

    def test_inlet_temp_zero_raises(self):
        with pytest.raises(ValueError, match="inlet_temp_C"):
            _predict(inlet_temp_C=0.0)

    def test_inlet_temp_negative_raises(self):
        with pytest.raises(ValueError, match="inlet_temp_C"):
            _predict(inlet_temp_C=-10.0)

    def test_feed_conc_zero_raises(self):
        with pytest.raises(ValueError, match="feed_conc_mg_mL"):
            _predict(feed_conc_mg_mL=0.0)

    def test_feed_conc_negative_raises(self):
        with pytest.raises(ValueError, match="feed_conc_mg_mL"):
            _predict(feed_conc_mg_mL=-1.0)
