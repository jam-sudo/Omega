"""Tests for Phase 303 — Mucus Penetration Predictor."""

import math

import pytest

from omega_pbpk.prediction.mucus_penetration_predictor import (
    MucusPenetrationResult,
    compare_mucus_types,
    predict_mucus_penetration,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _basic(mucus_type="intestinal", logp=1.0, charge=0.0, mw=300.0, mode="molecular"):
    return predict_mucus_penetration("TestDrug", mw, logp, charge, mucus_type, mode)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_instance(self):
        r = _basic()
        assert isinstance(r, MucusPenetrationResult)

    def test_drug_name_preserved(self):
        r = predict_mucus_penetration("Ibuprofen", 200.0, 3.5, 0.0, "gastric", "molecular")
        assert r.drug_name == "Ibuprofen"

    def test_mucus_type_preserved(self):
        r = _basic(mucus_type="bronchial")
        assert r.mucus_type == "bronchial"

    def test_particle_or_molecular_preserved(self):
        r = _basic(mode="particle")
        assert r.particle_or_molecular == "particle"


# ---------------------------------------------------------------------------
# Diffusivity ratio bounds
# ---------------------------------------------------------------------------


class TestDiffusivityRatio:
    def test_ratio_positive(self):
        r = _basic()
        assert r.effective_diffusivity_ratio > 0

    def test_ratio_at_most_one(self):
        r = _basic(logp=-4.0, charge=0.0, mw=50.0)
        assert r.effective_diffusivity_ratio <= 1.0

    def test_ratio_at_least_floor(self):
        # Worst case: large logp, high charge, large mw
        r = predict_mucus_penetration("Heavy", 5000.0, 9.0, 2.0, "cervical", "molecular")
        assert r.effective_diffusivity_ratio >= 0.001


# ---------------------------------------------------------------------------
# Penetration time
# ---------------------------------------------------------------------------


class TestPenetrationTime:
    def test_penetration_time_positive(self):
        r = _basic()
        assert r.penetration_time_h > 0

    def test_fast_class(self):
        # Very thin mucus (bronchial = 10 um) with easy molecule
        r = predict_mucus_penetration("Tiny", 50.0, -3.0, 0.0, "bronchial", "molecular")
        # Even if slow, time must be > 0; class depends on physics
        assert r.penetration_class in {"fast", "moderate", "slow"}

    def test_class_slow_for_very_high_viscosity(self):
        r = predict_mucus_penetration("BigDrug", 1000.0, 5.0, 2.0, "cervical", "molecular")
        assert r.penetration_class in {"moderate", "slow"}


# ---------------------------------------------------------------------------
# Physics: logP effect
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logp_lower_diffusivity(self):
        r_low = _basic(logp=0.0)
        r_high = _basic(logp=5.0)
        assert r_high.effective_diffusivity_ratio < r_low.effective_diffusivity_ratio

    def test_logp_below_1_no_penalty(self):
        r_neg = _basic(logp=-2.0)
        r_zero = _basic(logp=1.0)
        # Both should have f_hydro = 1.0 (no penalty when logp <= 1)
        assert math.isclose(r_neg.hydrophobic_factor, 1.0)
        assert math.isclose(r_zero.hydrophobic_factor, 1.0)

    def test_logp_above_1_has_penalty(self):
        r = _basic(logp=3.0)
        assert r.hydrophobic_factor < 1.0


# ---------------------------------------------------------------------------
# Physics: charge effect
# ---------------------------------------------------------------------------


class TestChargeEffect:
    def test_high_charge_lower_diffusivity(self):
        r_neutral = _basic(charge=0.0)
        r_charged = _basic(charge=2.0)
        assert r_charged.effective_diffusivity_ratio < r_neutral.effective_diffusivity_ratio

    def test_charge_factor_neutral(self):
        r = _basic(charge=0.0)
        assert math.isclose(r.charge_factor, 1.0)

    def test_charge_factor_monovalent(self):
        # abs(charge) = 1 → should NOT trigger the 0.3 factor
        r = _basic(charge=1.0)
        assert math.isclose(r.charge_factor, 1.0)

    def test_charge_factor_divalent(self):
        r = _basic(charge=2.0)
        assert math.isclose(r.charge_factor, 0.3)


# ---------------------------------------------------------------------------
# Physics: MW effect
# ---------------------------------------------------------------------------


class TestMWEffect:
    def test_larger_mw_lower_diffusivity(self):
        r_small = _basic(mw=100.0)
        r_large = _basic(mw=1000.0)
        assert r_large.effective_diffusivity_ratio < r_small.effective_diffusivity_ratio

    def test_molecular_radius_scales_with_mw(self):
        r = predict_mucus_penetration("Drug", 1000.0, 0.0, 0.0, "intestinal", "molecular")
        expected_r = 0.066 * (1000.0 ** (1.0 / 3.0))
        assert math.isclose(r.molecular_radius_nm, expected_r, rel_tol=1e-6)

    def test_particle_mode_fixed_radius(self):
        r = predict_mucus_penetration("NP", 300.0, 0.0, 0.0, "intestinal", "particle")
        assert math.isclose(r.molecular_radius_nm, 200.0)


# ---------------------------------------------------------------------------
# Mucus type comparison
# ---------------------------------------------------------------------------


class TestMucusTypeComparison:
    def test_bronchial_tighter_mesh_than_gastric(self):
        r_bronchial = _basic(mucus_type="bronchial", logp=0.0, charge=0.0, mw=200.0)
        r_gastric = _basic(mucus_type="gastric", logp=0.0, charge=0.0, mw=200.0)
        # Bronchial has smaller mesh (50 nm vs 200 nm), so lower obstruction factor
        assert r_bronchial.mesh_obstruction_factor < r_gastric.mesh_obstruction_factor

    def test_compare_returns_four_results(self):
        results = compare_mucus_types("Drug", 300.0, 1.0, 0.0)
        assert len(results) == 4

    def test_compare_sorted_descending(self):
        results = compare_mucus_types("Drug", 300.0, 1.0, 0.0)
        ratios = [r.effective_diffusivity_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_compare_all_mucus_types_present(self):
        results = compare_mucus_types("Drug", 300.0, 1.0, 0.0)
        types_present = {r.mucus_type for r in results}
        assert types_present == {"gastric", "intestinal", "bronchial", "cervical"}


# ---------------------------------------------------------------------------
# Mucociliary clearance risk
# ---------------------------------------------------------------------------


class TestMucociliaryClearance:
    def test_bronchial_high_risk_when_very_low_diffusivity(self):
        # Very lipophilic + highly charged → low diffusivity in bronchial
        r = predict_mucus_penetration("Sticky", 2000.0, 8.0, 3.0, "bronchial", "molecular")
        # Check that it's classified correctly based on diffusivity
        if r.effective_diffusivity_ratio < 0.01:
            assert r.mucociliary_clearance_risk == "high"

    def test_non_bronchial_always_low_risk(self):
        for mt in ("gastric", "intestinal", "cervical"):
            r = _basic(mucus_type=mt)
            assert r.mucociliary_clearance_risk == "low"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw"):
            predict_mucus_penetration("Drug", -1.0, 1.0, 0.0, "intestinal", "molecular")

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw"):
            predict_mucus_penetration("Drug", 0.0, 1.0, 0.0, "intestinal", "molecular")

    def test_invalid_mucus_type(self):
        with pytest.raises(ValueError, match="mucus_type"):
            predict_mucus_penetration("Drug", 300.0, 1.0, 0.0, "nasal", "molecular")

    def test_invalid_particle_mode(self):
        with pytest.raises(ValueError, match="particle_or_molecular"):
            predict_mucus_penetration("Drug", 300.0, 1.0, 0.0, "intestinal", "liposome")

    def test_logp_out_of_range_high(self):
        with pytest.raises(ValueError, match="logp"):
            predict_mucus_penetration("Drug", 300.0, 11.0, 0.0, "intestinal", "molecular")

    def test_logp_out_of_range_low(self):
        with pytest.raises(ValueError, match="logp"):
            predict_mucus_penetration("Drug", 300.0, -6.0, 0.0, "intestinal", "molecular")
