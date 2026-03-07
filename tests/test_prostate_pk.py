"""Tests for Phase 1143: core/prostate_pk.py"""

import math
import pytest

from omega_pbpk.core.prostate_pk import (
    ProstatePKResult,
    simulate_prostate_pk,
    _compute_logp_factor,
    _compute_ionization_factor,
    _penetration_class,
)


# ------------------------------------------------------------------ helpers --

def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.0,
        ionization_type="neutral",
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_prostate_pk(**params)


# ---------------------------------------------------------------- unit helpers -

class TestLogPFactor:
    def test_negative_logp_returns_minimum(self):
        assert _compute_logp_factor(-3) == 0.2

    def test_zero_logp_returns_minimum(self):
        assert _compute_logp_factor(0) == 0.2

    def test_logp_2_gives_1(self):
        assert math.isclose(_compute_logp_factor(2.0), 1.0)

    def test_logp_6_clamps_to_3(self):
        assert _compute_logp_factor(6.0) == 3.0

    def test_logp_1_gives_0p5(self):
        assert math.isclose(_compute_logp_factor(1.0), 0.5)


class TestIonizationFactor:
    def test_base_is_1p5(self):
        assert _compute_ionization_factor("base") == 1.5

    def test_acid_is_0p7(self):
        assert _compute_ionization_factor("acid") == 0.7

    def test_neutral_is_1p0(self):
        assert _compute_ionization_factor("neutral") == 1.0


class TestPenetrationClass:
    def test_poor_below_0p3(self):
        assert _penetration_class(0.1) == "poor"

    def test_moderate_at_0p3(self):
        assert _penetration_class(0.3) == "moderate"

    def test_moderate_at_1p0(self):
        assert _penetration_class(1.0) == "moderate"

    def test_good_above_1p0(self):
        assert _penetration_class(1.5) == "good"


# --------------------------------------------------------- simulation basics -

class TestSimulationBasics:
    def test_returns_result_type(self):
        r = default_result()
        assert isinstance(r, ProstatePKResult)

    def test_c_prostate_starts_at_zero(self):
        r = default_result()
        assert r.c_prostate_mg_L[0] == 0.0

    def test_c_plasma_starts_at_dose_over_vd(self):
        r = default_result(dose_mg=100.0, vd_L=50.0)
        assert math.isclose(r.c_plasma_mg_L[0], 2.0)

    def test_times_h_length_matches_profiles(self):
        r = default_result()
        assert len(r.times_h) == len(r.c_plasma_mg_L)
        assert len(r.times_h) == len(r.c_prostate_mg_L)

    def test_times_h_starts_at_zero(self):
        r = default_result()
        assert r.times_h[0] == 0.0

    def test_times_h_ends_near_t_end(self):
        r = default_result(t_end_h=12.0, dt_h=0.05)
        assert math.isclose(r.times_h[-1], 12.0, rel_tol=1e-6)

    def test_c_plasma_declines_over_time(self):
        r = default_result()
        assert r.c_plasma_mg_L[-1] < r.c_plasma_mg_L[0]

    def test_concentrations_non_negative(self):
        r = default_result()
        assert all(c >= 0 for c in r.c_plasma_mg_L)
        assert all(c >= 0 for c in r.c_prostate_mg_L)

    def test_auc_prostate_positive(self):
        r = default_result()
        assert r.auc_prostate_mg_h_per_L > 0

    def test_auc_plasma_positive(self):
        r = default_result()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_ratio_positive(self):
        r = default_result()
        assert r.prostate_to_plasma_auc_ratio > 0


# ---------------------------------------------------------- ionization effect -

class TestIonizationEffect:
    def test_base_higher_prostate_auc_than_acid(self):
        r_base = default_result(ionization_type="base", logP=2.0)
        r_acid = default_result(ionization_type="acid", logP=2.0)
        assert r_base.auc_prostate_mg_h_per_L > r_acid.auc_prostate_mg_h_per_L

    def test_base_higher_ratio_than_neutral(self):
        r_base = default_result(ionization_type="base", logP=2.0)
        r_neutral = default_result(ionization_type="neutral", logP=2.0)
        assert r_base.prostate_to_plasma_auc_ratio > r_neutral.prostate_to_plasma_auc_ratio

    def test_acid_lower_ratio_than_neutral(self):
        r_acid = default_result(ionization_type="acid", logP=2.0)
        r_neutral = default_result(ionization_type="neutral", logP=2.0)
        assert r_acid.prostate_to_plasma_auc_ratio < r_neutral.prostate_to_plasma_auc_ratio

    def test_base_notes_contain_ion_trapping(self):
        r = default_result(ionization_type="base")
        assert "ion trapping" in r.notes

    def test_neutral_no_ion_trapping_note(self):
        r = default_result(ionization_type="neutral")
        assert "ion trapping" not in r.notes


# ----------------------------------------------------------- logP effect -----

class TestLogPEffect:
    def test_higher_logp_more_prostate_penetration(self):
        r_high = default_result(logP=4.0)
        r_low = default_result(logP=0.5)
        assert r_high.prostate_to_plasma_auc_ratio > r_low.prostate_to_plasma_auc_ratio

    def test_negative_logp_uses_minimum_factor(self):
        r = default_result(logP=-2.0)
        # logP_factor should be 0.2, result should still be valid
        assert r.prostate_to_plasma_auc_ratio > 0

    def test_kp_prostate_bounded_above(self):
        r = default_result(logP=10.0, ionization_type="base")
        assert r.kp_prostate <= 10.0

    def test_kp_prostate_bounded_below(self):
        r = default_result(logP=-5.0, ionization_type="acid")
        assert r.kp_prostate >= 0.1


# ---------------------------------------------------------- penetration class -

class TestPenetrationClassAssignment:
    def test_low_logp_acid_lower_ratio_than_high_logp_base(self):
        # Low logP acid should have worse penetration than high logP base
        r_low = default_result(logP=0.2, ionization_type="acid", cl_L_per_h=20.0)
        r_high = default_result(logP=4.0, ionization_type="base", cl_L_per_h=20.0)
        assert r_low.prostate_to_plasma_auc_ratio < r_high.prostate_to_plasma_auc_ratio

    def test_good_class_for_high_logp_base(self):
        r = default_result(logP=5.0, ionization_type="base")
        assert r.prostate_penetration_class in ("good", "moderate")

    def test_clinical_utility_recommended_for_good(self):
        r = default_result(logP=5.0, ionization_type="base")
        if r.prostate_penetration_class == "good":
            assert r.clinical_utility == "recommended"

    def test_clinical_utility_not_recommended_for_poor(self):
        r = default_result(logP=0.1, ionization_type="acid", cl_L_per_h=50.0)
        if r.prostate_penetration_class == "poor":
            assert r.clinical_utility == "not_recommended"


# ----------------------------------------------------------- metadata fields -

class TestResultMetadata:
    def test_drug_name_preserved(self):
        r = default_result(drug_name="Ciprofloxacin")
        assert r.drug_name == "Ciprofloxacin"

    def test_dose_mg_preserved(self):
        r = default_result(dose_mg=250.0)
        assert r.dose_mg == 250.0

    def test_logP_preserved(self):
        r = default_result(logP=1.5)
        assert r.logP == 1.5

    def test_ionization_type_preserved(self):
        r = default_result(ionization_type="base")
        assert r.ionization_type == "base"

    def test_tmax_prostate_within_range(self):
        r = default_result()
        assert 0.0 <= r.tmax_prostate_h <= r.times_h[-1] + 1e-9


# ----------------------------------------------------------- validation ------

class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_prostate_pk("X", dose_mg=0.0, logP=2.0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_prostate_pk("X", dose_mg=-10.0, logP=2.0)

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_prostate_pk("X", dose_mg=100.0, logP=2.0, cl_L_per_h=0.0)

    def test_invalid_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_prostate_pk("X", dose_mg=100.0, logP=2.0, vd_L=-1.0)

    def test_logP_too_low(self):
        with pytest.raises(ValueError, match="logP"):
            simulate_prostate_pk("X", dose_mg=100.0, logP=-6.0)

    def test_logP_too_high(self):
        with pytest.raises(ValueError, match="logP"):
            simulate_prostate_pk("X", dose_mg=100.0, logP=11.0)

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            simulate_prostate_pk("X", dose_mg=100.0, logP=2.0, ionization_type="zwitterion")
