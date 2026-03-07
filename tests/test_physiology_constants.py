"""Tests for physiology_constants module (Phase 357)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.physiology_constants import (
    PhysiologyProfile,
    get_physiology,
    list_available_profiles,
    scale_physiology,
)


class TestGetPhysiologyHumanAdultMale:
    def test_returns_profile_type(self):
        p = get_physiology("human", "male", "adult")
        assert isinstance(p, PhysiologyProfile)

    def test_plasma_volume_70kg(self):
        p = get_physiology("human", "male", "adult")
        assert abs(p.plasma_volume_L - 3.0) < 0.01

    def test_cardiac_output_70kg(self):
        p = get_physiology("human", "male", "adult")
        assert abs(p.cardiac_output_L_per_h - 390.0) < 1.0

    def test_plasma_ph(self):
        p = get_physiology("human", "male", "adult")
        assert abs(p.plasma_ph - 7.4) < 0.01

    def test_gastric_ph_acidic(self):
        p = get_physiology("human", "male", "adult")
        assert p.gastric_ph < 3.0

    def test_all_volumes_positive(self):
        p = get_physiology("human", "male", "adult")
        assert p.plasma_volume_L > 0
        assert p.blood_volume_L > 0
        assert p.interstitial_volume_L > 0
        assert p.intracellular_volume_L > 0
        assert p.total_body_water_L > 0
        assert p.liver_volume_L > 0
        assert p.kidney_volume_L > 0
        assert p.lung_volume_L > 0
        assert p.brain_volume_L > 0

    def test_all_flows_positive(self):
        p = get_physiology("human", "male", "adult")
        assert p.cardiac_output_L_per_h > 0
        assert p.hepatic_blood_flow_L_per_h > 0
        assert p.renal_blood_flow_L_per_h > 0
        assert p.pulmonary_blood_flow_L_per_h > 0

    def test_pulmonary_equals_cardiac(self):
        p = get_physiology("human", "male", "adult")
        assert abs(p.pulmonary_blood_flow_L_per_h - p.cardiac_output_L_per_h) < 1e-6

    def test_liver_weight_g(self):
        p = get_physiology("human", "male", "adult")
        assert p.liver_weight_g > 1000  # should be ~1500 g

    def test_species_sex_age_stored(self):
        p = get_physiology("human", "male", "adult")
        assert p.species == "human"
        assert p.sex == "male"
        assert p.age_category == "adult"


class TestGetPhysiologyRat:
    def test_rat_cardiac_output_smaller_than_human(self):
        rat = get_physiology("rat", "male", "adult")
        human = get_physiology("human", "male", "adult")
        assert rat.cardiac_output_L_per_h < human.cardiac_output_L_per_h

    def test_rat_plasma_volume_small(self):
        rat = get_physiology("rat", "male", "adult")
        assert rat.plasma_volume_L < 0.1

    def test_rat_gastric_ph_higher_than_human(self):
        rat = get_physiology("rat", "male", "adult")
        human = get_physiology("human", "male", "adult")
        assert rat.gastric_ph > human.gastric_ph

    def test_rat_all_volumes_positive(self):
        rat = get_physiology("rat", "male", "adult")
        assert rat.liver_volume_L > 0
        assert rat.kidney_volume_L > 0


class TestWeightScaling:
    def test_heavier_larger_volumes(self):
        p_light = get_physiology("human", "male", "adult", weight_kg=50.0)
        p_heavy = get_physiology("human", "male", "adult", weight_kg=100.0)
        assert p_heavy.plasma_volume_L > p_light.plasma_volume_L
        assert p_heavy.liver_volume_L > p_light.liver_volume_L

    def test_volume_proportional_to_weight(self):
        p_ref = get_physiology("human", "male", "adult", weight_kg=70.0)
        p_140 = get_physiology("human", "male", "adult", weight_kg=140.0)
        # Volumes scale linearly
        ratio = p_140.plasma_volume_L / p_ref.plasma_volume_L
        assert abs(ratio - 2.0) < 0.01

    def test_blood_flow_sublinear_scaling(self):
        p_ref = get_physiology("human", "male", "adult", weight_kg=70.0)
        p_140 = get_physiology("human", "male", "adult", weight_kg=140.0)
        flow_ratio = p_140.cardiac_output_L_per_h / p_ref.cardiac_output_L_per_h
        # Should be 2^0.75 ≈ 1.68
        assert 1.5 < flow_ratio < 1.9


class TestAgeCategories:
    def test_neonatal_smaller_than_adult(self):
        adult = get_physiology("human", "male", "adult")
        neonatal = get_physiology("human", "male", "neonatal")
        assert neonatal.plasma_volume_L < adult.plasma_volume_L
        assert neonatal.liver_volume_L < adult.liver_volume_L

    def test_elderly_close_to_adult(self):
        adult = get_physiology("human", "male", "adult")
        elderly = get_physiology("human", "male", "elderly")
        ratio = elderly.plasma_volume_L / adult.plasma_volume_L
        # Should be ~0.9 (10% smaller)
        assert 0.8 < ratio < 1.0

    def test_pediatric_between_neonatal_and_adult(self):
        adult = get_physiology("human", "male", "adult")
        pediatric = get_physiology("human", "male", "pediatric")
        neonatal = get_physiology("human", "male", "neonatal")
        assert neonatal.plasma_volume_L < pediatric.plasma_volume_L < adult.plasma_volume_L


class TestListAvailableProfiles:
    def test_returns_non_empty_list(self):
        profiles = list_available_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) > 0

    def test_contains_human_male_adult(self):
        profiles = list_available_profiles()
        assert "human/male/adult" in profiles

    def test_contains_rat(self):
        profiles = list_available_profiles()
        assert any("rat" in p for p in profiles)


class TestScalePhysiology:
    def test_scale_proportional_volumes(self):
        base = get_physiology("human", "male", "adult")
        scaled = scale_physiology(base, 35.0)  # half the weight
        ratio = scaled.plasma_volume_L / base.plasma_volume_L
        assert abs(ratio - 0.5) < 0.01

    def test_scale_sublinear_flows(self):
        base = get_physiology("human", "male", "adult")
        scaled = scale_physiology(base, 35.0)
        flow_ratio = scaled.cardiac_output_L_per_h / base.cardiac_output_L_per_h
        # (35/70)^0.75 = 0.5^0.75 ≈ 0.595
        assert 0.5 < flow_ratio < 0.7

    def test_scale_returns_profile(self):
        base = get_physiology("human", "male", "adult")
        scaled = scale_physiology(base, 90.0)
        assert isinstance(scaled, PhysiologyProfile)

    def test_scale_weight_stored(self):
        base = get_physiology("rat", "male", "adult")
        scaled = scale_physiology(base, 0.5)
        assert abs(scaled.weight_kg - 0.5) < 1e-6

    def test_scale_invalid_weight_raises(self):
        base = get_physiology("human", "male", "adult")
        with pytest.raises(ValueError):
            scale_physiology(base, -1.0)


class TestInputValidation:
    def test_invalid_species_raises(self):
        with pytest.raises(ValueError, match="species"):
            get_physiology("pig", "male", "adult")

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            get_physiology("human", "unknown_sex", "adult")

    def test_invalid_age_category_raises(self):
        with pytest.raises(ValueError, match="age_category"):
            get_physiology("human", "male", "teenager")

    def test_mouse_profiles_accessible(self):
        p = get_physiology("mouse", "male", "adult")
        assert p.weight_kg < 0.1
        assert p.cardiac_output_L_per_h > 0

    def test_dog_profiles_accessible(self):
        p = get_physiology("dog", "male", "adult")
        assert p.weight_kg > 5.0

    def test_monkey_profiles_accessible(self):
        p = get_physiology("monkey", "male", "adult")
        assert p.plasma_ph == pytest.approx(7.4, abs=0.01)
