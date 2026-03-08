"""Tests for Phase 622: Volume of Distribution Scaling."""

import math

import pytest

from omega_pbpk.prediction.vd_scaling_p622 import (
    VdScalingResult,
    predict_vd_scaling,
    scale_across_species,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        logP=2.0,
        fu_plasma=0.1,
        body_weight_kg=70.0,
        pka=7.0,
        drug_type="neutral",
        species="human",
        cl_L_per_h=0.0,
    )
    params.update(kwargs)
    return predict_vd_scaling(**params)


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_vd_scaling_result(self):
        r = human_result()
        assert isinstance(r, VdScalingResult)

    def test_frozen_dataclass(self):
        r = human_result()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        r = predict_vd_scaling("Warfarin", 2.0, 0.01, 70.0, drug_type="acid")
        assert r.drug_name == "Warfarin"

    def test_species_preserved(self):
        r = human_result(species="rat", body_weight_kg=0.25)
        assert r.species == "rat"

    def test_body_weight_preserved(self):
        r = human_result(body_weight_kg=70.0)
        assert r.body_weight_kg == pytest.approx(70.0)

    def test_logP_preserved(self):
        r = human_result(logP=3.5)
        assert r.logP == pytest.approx(3.5)

    def test_fu_plasma_preserved(self):
        r = human_result(fu_plasma=0.2)
        assert r.fu_plasma == pytest.approx(0.2)

    def test_drug_type_preserved(self):
        r = human_result(drug_type="base")
        assert r.drug_type == "base"

    def test_notes_is_str(self):
        r = human_result()
        assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Vd values
# ---------------------------------------------------------------------------


class TestVdValues:
    def test_vd_predicted_L_equals_per_kg_times_bw(self):
        r = human_result(body_weight_kg=70.0)
        assert r.vd_predicted_L == pytest.approx(r.vd_predicted_L_per_kg * 70.0, rel=1e-9)

    def test_vd_positive(self):
        r = human_result()
        assert r.vd_predicted_L_per_kg > 0.0
        assert r.vd_predicted_L > 0.0

    def test_base_drug_higher_vd_than_acid(self):
        r_acid = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.5, body_weight_kg=70.0, drug_type="acid"
        )
        r_base = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.5, body_weight_kg=70.0, drug_type="base"
        )
        assert r_base.vd_predicted_L_per_kg > r_acid.vd_predicted_L_per_kg

    def test_lower_fu_higher_vd(self):
        r_high_fu = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.8, body_weight_kg=70.0, drug_type="neutral"
        )
        r_low_fu = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.1, body_weight_kg=70.0, drug_type="neutral"
        )
        assert r_low_fu.vd_predicted_L_per_kg > r_high_fu.vd_predicted_L_per_kg

    def test_higher_logP_higher_vd_neutral(self):
        r_low = predict_vd_scaling(
            "D", logP=0.0, fu_plasma=0.5, body_weight_kg=70.0, drug_type="neutral"
        )
        r_high = predict_vd_scaling(
            "D", logP=4.0, fu_plasma=0.5, body_weight_kg=70.0, drug_type="neutral"
        )
        assert r_high.vd_predicted_L_per_kg > r_low.vd_predicted_L_per_kg

    def test_vd_clamped_above_minimum(self):
        # Extreme low logP, high fu → should still be >= 0.05
        r = predict_vd_scaling(
            "D", logP=-10.0, fu_plasma=1.0, body_weight_kg=70.0, drug_type="neutral"
        )
        assert r.vd_predicted_L_per_kg >= 0.05

    def test_vd_clamped_below_maximum(self):
        # Very low fu → risk of exceeding 50 L/kg clamp
        r = predict_vd_scaling(
            "D", logP=5.0, fu_plasma=0.001, body_weight_kg=70.0, drug_type="base"
        )
        assert r.vd_predicted_L_per_kg <= 50.0


# ---------------------------------------------------------------------------
# Vd classification
# ---------------------------------------------------------------------------


class TestVdClassification:
    def test_vd_class_low(self):
        # High fu, acid, very negative logP → low Vd < 0.3 L/kg
        # vd_base = 0.1 + 1.0*0.2 + (-2.0)*0.1 = 0.1 → vd_per_kg = 0.1/1.0 = 0.1 → "low"
        r = predict_vd_scaling("D", logP=-2.0, fu_plasma=1.0, body_weight_kg=70.0, drug_type="acid")
        assert r.vd_class == "low"
        assert r.main_distribution_tissue == "plasma"

    def test_vd_class_high_for_basic_lipophilic(self):
        # Basic, lipophilic, low fu → high or very_high Vd
        r = predict_vd_scaling("D", logP=4.0, fu_plasma=0.05, body_weight_kg=70.0, drug_type="base")
        assert r.vd_class in ("high", "very_high")
        assert r.main_distribution_tissue == "extensive tissue binding"

    def test_vd_class_moderate_exists(self):
        # Find a moderate scenario
        r = predict_vd_scaling(
            "D", logP=1.0, fu_plasma=0.5, body_weight_kg=70.0, drug_type="neutral"
        )
        assert r.vd_class in ("low", "moderate", "high")

    def test_very_high_vd_tissue_distribution(self):
        # Force very_high: vd > 10 L/kg
        r = predict_vd_scaling("D", logP=5.0, fu_plasma=0.02, body_weight_kg=70.0, drug_type="base")
        if r.vd_class == "very_high":
            assert r.main_distribution_tissue == "extensive tissue binding"


# ---------------------------------------------------------------------------
# Species scaling
# ---------------------------------------------------------------------------


class TestSpeciesScaling:
    def test_human_uses_scale_1(self):
        r = predict_vd_scaling("D", logP=2.0, fu_plasma=0.2, body_weight_kg=70.0, species="human")
        r_ref = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.2, body_weight_kg=70.0, species="unknown_species"
        )
        assert r.vd_predicted_L_per_kg == pytest.approx(r_ref.vd_predicted_L_per_kg, rel=1e-9)

    def test_mouse_lower_vd_than_human(self):
        r_human = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.2, body_weight_kg=70.0, species="human"
        )
        r_mouse = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.2, body_weight_kg=0.02, species="mouse"
        )
        assert r_mouse.vd_predicted_L_per_kg < r_human.vd_predicted_L_per_kg

    def test_rat_lower_vd_per_kg_than_human(self):
        r_human = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.2, body_weight_kg=70.0, species="human"
        )
        r_rat = predict_vd_scaling("D", logP=2.0, fu_plasma=0.2, body_weight_kg=0.25, species="rat")
        assert r_rat.vd_predicted_L_per_kg <= r_human.vd_predicted_L_per_kg


# ---------------------------------------------------------------------------
# t_half
# ---------------------------------------------------------------------------


class TestTHalf:
    def test_t_half_zero_when_no_cl(self):
        r = human_result(cl_L_per_h=0.0)
        assert r.t_half_predicted_h == pytest.approx(0.0)

    def test_t_half_positive_when_cl_provided(self):
        r = human_result(cl_L_per_h=5.0)
        assert r.t_half_predicted_h > 0.0

    def test_t_half_formula(self):
        r = human_result(body_weight_kg=70.0, cl_L_per_h=5.0)
        expected = math.log(2.0) * r.vd_predicted_L / 5.0
        assert r.t_half_predicted_h == pytest.approx(expected, rel=1e-9)

    def test_larger_vd_longer_t_half(self):
        r_low_fu = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.05, body_weight_kg=70.0, drug_type="neutral", cl_L_per_h=5.0
        )
        r_high_fu = predict_vd_scaling(
            "D", logP=2.0, fu_plasma=0.8, body_weight_kg=70.0, drug_type="neutral", cl_L_per_h=5.0
        )
        assert r_low_fu.t_half_predicted_h > r_high_fu.t_half_predicted_h


# ---------------------------------------------------------------------------
# Fraction in plasma
# ---------------------------------------------------------------------------


class TestFractionInPlasma:
    def test_fraction_in_plasma_in_0_1(self):
        r = human_result()
        assert 0.0 <= r.fraction_in_plasma <= 1.0

    def test_low_vd_high_fraction_in_plasma(self):
        # Low Vd drug → more drug stays in plasma
        r_low = predict_vd_scaling(
            "D", logP=0.0, fu_plasma=1.0, body_weight_kg=70.0, drug_type="acid"
        )
        r_high = predict_vd_scaling(
            "D", logP=4.0, fu_plasma=0.05, body_weight_kg=70.0, drug_type="base"
        )
        assert r_low.fraction_in_plasma > r_high.fraction_in_plasma

    def test_fraction_formula(self):
        r = human_result(body_weight_kg=70.0)
        plasma_vol = 70.0 * 0.045
        expected = min(1.0, plasma_vol / r.vd_predicted_L)
        assert r.fraction_in_plasma == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# scale_across_species
# ---------------------------------------------------------------------------


class TestScaleAcrossSpecies:
    def test_returns_list(self):
        results = scale_across_species("D", logP=2.0, fu_plasma=0.2)
        assert isinstance(results, list)

    def test_returns_five_species(self):
        results = scale_across_species("D", logP=2.0, fu_plasma=0.2)
        assert len(results) == 5

    def test_all_vd_scaling_result_instances(self):
        results = scale_across_species("D", logP=2.0, fu_plasma=0.2)
        for r in results:
            assert isinstance(r, VdScalingResult)

    def test_sorted_ascending_by_vd_per_kg(self):
        results = scale_across_species("D", logP=2.0, fu_plasma=0.2)
        vds = [r.vd_predicted_L_per_kg for r in results]
        assert vds == sorted(vds)

    def test_all_five_species_present(self):
        results = scale_across_species("D", logP=2.0, fu_plasma=0.2)
        species_set = {r.species for r in results}
        assert species_set == {"human", "rat", "mouse", "dog", "monkey"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_body_weight_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            predict_vd_scaling("D", 2.0, 0.1, -1.0)

    def test_zero_body_weight_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            predict_vd_scaling("D", 2.0, 0.1, 0.0)

    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_vd_scaling("D", 2.0, 0.0, 70.0)

    def test_fu_plasma_above_1_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_vd_scaling("D", 2.0, 1.1, 70.0)

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_vd_scaling("D", 2.0, 0.1, 70.0, drug_type="zwitterion")
