"""Tests for Phase 346 — Physiological Time Scaling (Species Translation)."""

import math

import pytest

from omega_pbpk.core.physiological_time_scaling import (
    _BODY_WEIGHTS,
    _CL_EXPONENT,
    PhysiologicalTimeScalingResult,
    translate_across_species,
    translate_species,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rat_to_human(**kwargs):
    """Rat → human with sensible defaults."""
    defaults = dict(
        drug_name="TestDrug",
        species_from="rat",
        species_to="human",
        cl_from_L_per_h=0.5,
        vd_from_L=1.0,
        t_half_from_h=1.0,
    )
    defaults.update(kwargs)
    return translate_species(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_frozen_dataclass(self):
        r = _rat_to_human()
        assert isinstance(r, PhysiologicalTimeScalingResult)

    def test_result_is_immutable(self):
        r = _rat_to_human()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        r = _rat_to_human(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_species_fields(self):
        r = _rat_to_human()
        assert r.species_from == "rat"
        assert r.species_to == "human"


# ---------------------------------------------------------------------------
# Scaling direction (small species → human should give larger values)
# ---------------------------------------------------------------------------


class TestScalingDirection:
    def test_cl_predicted_larger_than_cl_from_for_rat(self):
        r = _rat_to_human(cl_from_L_per_h=0.5, vd_from_L=1.0)
        assert r.cl_predicted_L_per_h > r.cl_from_L_per_h

    def test_vd_predicted_larger_than_vd_from_for_rat(self):
        r = _rat_to_human(cl_from_L_per_h=0.5, vd_from_L=1.0)
        assert r.vd_predicted_L > r.vd_from_L

    def test_cl_predicted_larger_for_mouse(self):
        r = translate_species("Drug", "mouse", "human", 0.01, 0.05)
        assert r.cl_predicted_L_per_h > r.cl_from_L_per_h

    def test_vd_predicted_larger_for_mouse(self):
        r = translate_species("Drug", "mouse", "human", 0.01, 0.05)
        assert r.vd_predicted_L > r.vd_from_L

    def test_t_half_predicted_positive(self):
        r = _rat_to_human()
        assert r.t_half_predicted_h > 0.0


# ---------------------------------------------------------------------------
# Allometric scaling accuracy
# ---------------------------------------------------------------------------


class TestScalingAccuracy:
    def test_cl_rat_to_human_exponent_075(self):
        """CL_human/CL_rat should equal (70/0.25)^0.75."""
        cl_rat = 0.5
        r = translate_species("Drug", "rat", "human", cl_rat, 1.0)
        bw_rat = _BODY_WEIGHTS["rat"]
        bw_human = _BODY_WEIGHTS["human"]
        expected_ratio = (bw_human / bw_rat) ** _CL_EXPONENT
        actual_ratio = r.cl_predicted_L_per_h / cl_rat
        assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-6)

    def test_vd_rat_to_human_linear(self):
        """Vd_human/Vd_rat should equal 70/0.25 (linear scaling)."""
        vd_rat = 1.0
        r = translate_species("Drug", "rat", "human", 0.5, vd_rat)
        bw_rat = _BODY_WEIGHTS["rat"]
        bw_human = _BODY_WEIGHTS["human"]
        expected_ratio = bw_human / bw_rat
        actual_ratio = r.vd_predicted_L / vd_rat
        assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-6)

    def test_scaling_exponents_stored(self):
        r = _rat_to_human()
        assert math.isclose(r.cl_scaling_exponent, 0.75)
        assert math.isclose(r.vd_scaling_exponent, 1.0)


# ---------------------------------------------------------------------------
# MLP correction
# ---------------------------------------------------------------------------


class TestMLPCorrection:
    def test_mlp_correction_applied_for_mouse(self):
        r = translate_species("Drug", "mouse", "human", 0.01, 0.05)
        assert r.mlp_correction_applied is True

    def test_mlp_correction_applied_for_rat(self):
        r = _rat_to_human()
        assert r.mlp_correction_applied is True

    def test_mlp_correction_applied_for_dog(self):
        r = translate_species("Drug", "dog", "human", 0.1, 1.0)
        assert r.mlp_correction_applied is True

    def test_no_mlp_correction_for_monkey(self):
        # Monkey has MLP = 30 years (not < 30)
        r = translate_species("Drug", "monkey", "human", 0.2, 2.0)
        assert r.mlp_correction_applied is False


# ---------------------------------------------------------------------------
# Confidence class
# ---------------------------------------------------------------------------


class TestConfidenceClass:
    def test_rat_confidence_high(self):
        r = _rat_to_human()
        assert r.confidence_class == "high"

    def test_dog_confidence_high(self):
        r = translate_species("Drug", "dog", "human", 0.1, 1.0)
        assert r.confidence_class == "high"

    def test_monkey_confidence_high(self):
        r = translate_species("Drug", "monkey", "human", 0.2, 2.0)
        assert r.confidence_class == "high"

    def test_mouse_confidence_moderate(self):
        r = translate_species("Drug", "mouse", "human", 0.01, 0.05)
        assert r.confidence_class == "moderate"

    def test_human_confidence_reference(self):
        r = translate_species("Drug", "human", "human", 5.0, 50.0)
        assert r.confidence_class == "reference"

    def test_confidence_class_in_valid_set(self):
        valid = {"high", "moderate", "reference"}
        r = _rat_to_human()
        assert r.confidence_class in valid


# ---------------------------------------------------------------------------
# Trivial case: same species
# ---------------------------------------------------------------------------


class TestTrivialCase:
    def test_same_species_works(self):
        r = translate_species("Drug", "rat", "rat", 0.5, 1.0)
        assert isinstance(r, PhysiologicalTimeScalingResult)
        assert math.isclose(r.cl_predicted_L_per_h, 0.5, rel_tol=1e-9)
        assert math.isclose(r.vd_predicted_L, 1.0, rel_tol=1e-9)

    def test_human_to_human_identity(self):
        r = translate_species("Drug", "human", "human", 5.0, 50.0)
        assert math.isclose(r.cl_predicted_L_per_h, 5.0, rel_tol=1e-9)
        assert math.isclose(r.vd_predicted_L, 50.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# translate_across_species
# ---------------------------------------------------------------------------


class TestTranslateAcrossSpecies:
    def test_returns_correct_count(self):
        cl_dict = {"rat": 0.5, "dog": 0.8, "monkey": 0.3}
        vd_dict = {"rat": 1.0, "dog": 5.0, "monkey": 2.0}
        results = translate_across_species("Drug", cl_dict, vd_dict)
        assert len(results) == 3

    def test_all_results_are_correct_type(self):
        cl_dict = {"rat": 0.5, "dog": 0.8}
        vd_dict = {"rat": 1.0, "dog": 5.0}
        results = translate_across_species("Drug", cl_dict, vd_dict)
        for r in results:
            assert isinstance(r, PhysiologicalTimeScalingResult)

    def test_species_to_default_human(self):
        cl_dict = {"rat": 0.5}
        vd_dict = {"rat": 1.0}
        results = translate_across_species("Drug", cl_dict, vd_dict)
        assert results[0].species_to == "human"

    def test_missing_vd_entry_skipped(self):
        # rat has vd, mouse does not
        cl_dict = {"rat": 0.5, "mouse": 0.01}
        vd_dict = {"rat": 1.0}
        results = translate_across_species("Drug", cl_dict, vd_dict)
        assert len(results) == 1
        assert results[0].species_from == "rat"

    def test_single_species_result(self):
        results = translate_across_species("Drug", {"rat": 0.5}, {"rat": 1.0})
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_species_from_raises(self):
        with pytest.raises(ValueError, match="species_from"):
            translate_species("Drug", "rabbit", "human", 0.5, 1.0)

    def test_invalid_species_to_raises(self):
        with pytest.raises(ValueError, match="species_to"):
            translate_species("Drug", "rat", "pig", 0.5, 1.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_from"):
            translate_species("Drug", "rat", "human", 0.0, 1.0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_from"):
            translate_species("Drug", "rat", "human", -0.5, 1.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_from"):
            translate_species("Drug", "rat", "human", 0.5, 0.0)

    def test_vd_negative_raises(self):
        with pytest.raises(ValueError, match="vd_from"):
            translate_species("Drug", "rat", "human", 0.5, -1.0)
