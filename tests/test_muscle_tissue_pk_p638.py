"""
Tests for Phase 638 — Drug Muscle Tissue Distribution
"""

import pytest

from omega_pbpk.core.muscle_tissue_pk_p638 import (
    MuscleTissuePKResult,
    simulate_muscle_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.0,
        mw_Da=400.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        muscle_mass_kg=28.0,
        sex="male",
        t_end_h=48.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_muscle_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        r = run_default()
        assert isinstance(r, MuscleTissuePKResult)

    def test_drug_name_preserved(self):
        r = run_default(drug_name="Statin")
        assert r.drug_name == "Statin"

    def test_dose_preserved(self):
        r = run_default(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_muscle_mass_preserved(self):
        r = run_default(muscle_mass_kg=30.0)
        assert r.muscle_mass_kg == 30.0

    def test_times_is_list(self):
        r = run_default()
        assert isinstance(r.times_h, list)

    def test_c_plasma_is_list(self):
        r = run_default()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_c_muscle_is_list(self):
        r = run_default()
        assert isinstance(r.c_muscle_mg_g, list)

    def test_lists_same_length(self):
        r = run_default(t_end_h=24.0, dt_h=0.5)
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_muscle_mg_g) == n

    def test_times_start_at_zero(self):
        r = run_default()
        assert r.times_h[0] == 0.0

    def test_times_end_near_t_end(self):
        r = run_default(t_end_h=24.0, dt_h=0.1)
        assert abs(r.times_h[-1] - 24.0) < 0.2


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_plasma_initial_zero(self):
        r = run_default()
        assert r.c_plasma_mg_L[0] == 0.0

    def test_muscle_initial_zero(self):
        r = run_default()
        assert r.c_muscle_mg_g[0] == 0.0


# ---------------------------------------------------------------------------
# Physical realism
# ---------------------------------------------------------------------------


class TestPhysicalRealism:
    def test_all_concentrations_non_negative(self):
        r = run_default()
        assert all(v >= 0.0 for v in r.c_plasma_mg_L)
        assert all(v >= 0.0 for v in r.c_muscle_mg_g)

    def test_cmax_plasma_positive(self):
        r = run_default()
        assert r.cmax_plasma_mg_L > 0.0

    def test_cmax_muscle_positive(self):
        r = run_default()
        assert r.cmax_muscle_mg_g > 0.0

    def test_auc_plasma_positive(self):
        r = run_default()
        assert r.auc_plasma_mg_h_per_L > 0.0

    def test_auc_muscle_positive(self):
        r = run_default()
        assert r.auc_muscle_mg_h_per_g > 0.0

    def test_plasma_eventually_declines(self):
        r = run_default(t_end_h=72.0)
        # Concentration at end should be much lower than peak
        assert r.c_plasma_mg_L[-1] < r.cmax_plasma_mg_L


# ---------------------------------------------------------------------------
# Sex effect
# ---------------------------------------------------------------------------


class TestSexEffect:
    def test_female_lower_effective_muscle_mass(self):
        _r_male = run_default(sex="male", muscle_mass_kg=28.0)
        r_female = run_default(sex="female", muscle_mass_kg=28.0)
        # Female has 15% less effective muscle mass → lower muscle Vd → different PK
        # Kp is same; female Vd_muscle = 0.85 * male Vd_muscle
        # Female muscle concentration (mg/g) should be slightly different
        # We check notes contain different effective mass
        assert "23.8" in r_female.notes or "23" in r_female.notes  # 28 * 0.85 ≈ 23.8

    def test_male_female_same_kp(self):
        r_male = run_default(sex="male")
        r_female = run_default(sex="female")
        assert r_male.kp_muscle == r_female.kp_muscle

    def test_female_label(self):
        r = run_default(sex="female")
        assert isinstance(r, MuscleTissuePKResult)

    def test_male_label(self):
        r = run_default(sex="male")
        assert isinstance(r, MuscleTissuePKResult)


# ---------------------------------------------------------------------------
# logP effects on Kp
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logP_increases_kp(self):
        r_low = simulate_muscle_pk(
            "D", 100.0, logP=-1.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        r_high = simulate_muscle_pk(
            "D", 100.0, logP=4.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        assert r_high.kp_muscle > r_low.kp_muscle

    def test_kp_clamped_to_minimum(self):
        r = simulate_muscle_pk(
            "D", 100.0, logP=-10.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        assert r.kp_muscle >= 0.1

    def test_kp_clamped_to_maximum(self):
        r = simulate_muscle_pk(
            "D", 100.0, logP=100.0, mw_Da=10.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        assert r.kp_muscle <= 8.0

    def test_higher_logP_increases_myotoxicity(self):
        r_low = simulate_muscle_pk(
            "D", 100.0, logP=1.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        r_high = simulate_muscle_pk(
            "D", 100.0, logP=4.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        assert r_high.myotoxicity_index > r_low.myotoxicity_index


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_double_dose_doubles_cmax_plasma(self):
        r1 = run_default(dose_mg=100.0)
        r2 = run_default(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_auc(self):
        r1 = run_default(dose_mg=100.0)
        r2 = run_default(dose_mg=200.0)
        ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


# ---------------------------------------------------------------------------
# t_equilibration
# ---------------------------------------------------------------------------


class TestEquilibration:
    def test_t_equilibration_positive(self):
        r = run_default()
        assert r.t_equilibration_h > 0.0

    def test_higher_kp_longer_equilibration(self):
        # Higher Kp → larger Vd_muscle → slower k_out → longer equilibration
        r_low = simulate_muscle_pk(
            "D", 100.0, logP=0.5, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        r_high = simulate_muscle_pk(
            "D", 100.0, logP=4.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        assert r_high.t_equilibration_h > r_low.t_equilibration_h


# ---------------------------------------------------------------------------
# AUC ratio
# ---------------------------------------------------------------------------


class TestAUCRatio:
    def test_muscle_to_plasma_ratio_positive(self):
        r = run_default()
        assert r.muscle_to_plasma_ratio > 0.0

    def test_higher_kp_higher_muscle_to_plasma(self):
        r_low = simulate_muscle_pk(
            "D", 100.0, logP=0.5, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        r_high = simulate_muscle_pk(
            "D", 100.0, logP=4.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
        )
        assert r_high.muscle_to_plasma_ratio >= r_low.muscle_to_plasma_ratio


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_muscle_pk("D", -1.0, logP=1.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_muscle_pk("D", 0.0, logP=1.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            simulate_muscle_pk("D", 100.0, logP=1.0, mw_Da=300.0, cl_sys_L_per_h=0.0, vd_sys_L=50.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            simulate_muscle_pk(
                "D", 100.0, logP=1.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=-10.0
            )

    def test_zero_muscle_mass_raises(self):
        with pytest.raises(ValueError, match="muscle_mass_kg"):
            simulate_muscle_pk(
                "D",
                100.0,
                logP=1.0,
                mw_Da=300.0,
                cl_sys_L_per_h=5.0,
                vd_sys_L=50.0,
                muscle_mass_kg=0.0,
            )

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            simulate_muscle_pk(
                "D", 100.0, logP=1.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0, sex="other"
            )

    def test_notes_is_string(self):
        r = run_default()
        assert isinstance(r.notes, str)
