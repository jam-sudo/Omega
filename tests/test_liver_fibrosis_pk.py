"""Tests for clinical/liver_fibrosis_pk.py — Phase 574."""

import pytest

from omega_pbpk.clinical.liver_fibrosis_pk import (
    albumin_binding_fibrosis,
    dose_recommendation_cirrhosis,
    fibrosis_cl_scaling,
    portal_hypertension_bioavailability,
    simulate_cirrhosis_pk,
)

# ---------------------------------------------------------------------------
# fibrosis_cl_scaling
# ---------------------------------------------------------------------------


class TestFibrosisCLScaling:
    def test_child_pugh_a_cl_ratio(self):
        result = fibrosis_cl_scaling(10.0, child_pugh_score=6)
        assert abs(result["cl_ratio"] - 0.75) < 1e-9

    def test_child_pugh_b_cl_ratio(self):
        result = fibrosis_cl_scaling(10.0, child_pugh_score=8)
        assert abs(result["cl_ratio"] - 0.50) < 1e-9

    def test_child_pugh_c_cl_ratio(self):
        result = fibrosis_cl_scaling(10.0, child_pugh_score=12)
        assert abs(result["cl_ratio"] - 0.25) < 1e-9

    def test_child_pugh_c_cl_is_25pct(self):
        cl_normal = 20.0
        result = fibrosis_cl_scaling(cl_normal, child_pugh_score=15)
        assert abs(result["cl_fibrosis_L_per_h"] - cl_normal * 0.25) < 1e-9

    def test_higher_score_lower_cl_ratio(self):
        a = fibrosis_cl_scaling(10.0, child_pugh_score=5)
        b = fibrosis_cl_scaling(10.0, child_pugh_score=9)
        c = fibrosis_cl_scaling(10.0, child_pugh_score=14)
        assert a["cl_ratio"] > b["cl_ratio"] > c["cl_ratio"]

    def test_class_a_label(self):
        result = fibrosis_cl_scaling(10.0, child_pugh_score=5)
        assert result["child_pugh_class"] == "A"

    def test_class_b_label(self):
        result = fibrosis_cl_scaling(10.0, child_pugh_score=7)
        assert result["child_pugh_class"] == "B"

    def test_class_c_label(self):
        result = fibrosis_cl_scaling(10.0, child_pugh_score=10)
        assert result["child_pugh_class"] == "C"

    def test_score_below_5_raises(self):
        with pytest.raises(ValueError):
            fibrosis_cl_scaling(10.0, child_pugh_score=4)

    def test_score_above_15_raises(self):
        with pytest.raises(ValueError):
            fibrosis_cl_scaling(10.0, child_pugh_score=16)

    def test_return_keys(self):
        result = fibrosis_cl_scaling(10.0, child_pugh_score=7)
        assert "cl_fibrosis_L_per_h" in result
        assert "cl_ratio" in result
        assert "child_pugh_class" in result
        assert "dose_adjustment_factor" in result
        assert "notes" in result


# ---------------------------------------------------------------------------
# albumin_binding_fibrosis
# ---------------------------------------------------------------------------


class TestAlbuminBindingFibrosis:
    def test_low_albumin_higher_fu(self):
        fu_low = albumin_binding_fibrosis(0.1, albumin_g_dL=2.0, albumin_normal_g_dL=4.0)
        fu_normal = albumin_binding_fibrosis(0.1, albumin_g_dL=4.0, albumin_normal_g_dL=4.0)
        assert fu_low > fu_normal

    def test_normal_albumin_returns_fu_normal(self):
        fu = albumin_binding_fibrosis(0.1, albumin_g_dL=4.0, albumin_normal_g_dL=4.0)
        assert abs(fu - 0.1) < 1e-9

    def test_clamped_to_one(self):
        fu = albumin_binding_fibrosis(0.5, albumin_g_dL=0.5, albumin_normal_g_dL=4.0)
        assert fu <= 1.0

    def test_result_positive(self):
        fu = albumin_binding_fibrosis(0.05, albumin_g_dL=3.0)
        assert fu > 0.0

    def test_formula_correctness(self):
        fu_normal = 0.1
        albumin = 2.0
        albumin_ref = 4.0
        expected = fu_normal * (albumin_ref / albumin) ** 0.5
        result = albumin_binding_fibrosis(fu_normal, albumin, albumin_ref)
        assert abs(result - min(expected, 1.0)) < 1e-9


# ---------------------------------------------------------------------------
# portal_hypertension_bioavailability
# ---------------------------------------------------------------------------


class TestPortalHypertensionBioavailability:
    def test_increases_bioavailability(self):
        f_normal = 0.5
        f_fibrosis = portal_hypertension_bioavailability(f_normal, 0.3)
        assert f_fibrosis > f_normal

    def test_f_normal_1_stays_1(self):
        f = portal_hypertension_bioavailability(1.0, 0.3)
        assert abs(f - 1.0) < 1e-9

    def test_f_normal_0_equals_shunt_fraction(self):
        f = portal_hypertension_bioavailability(0.0, 0.3)
        assert abs(f - 0.3) < 1e-9

    def test_clamped_to_one(self):
        f = portal_hypertension_bioavailability(0.95, 0.9)
        assert f <= 1.0

    def test_formula_correctness(self):
        f_normal = 0.4
        shunt = 0.25
        expected = f_normal + (1 - f_normal) * shunt
        result = portal_hypertension_bioavailability(f_normal, shunt)
        assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# dose_recommendation_cirrhosis
# ---------------------------------------------------------------------------


class TestDoseRecommendationCirrhosis:
    def test_return_keys(self):
        result = dose_recommendation_cirrhosis("Drug A", 100.0, cl_ratio=0.5, fu_ratio=1.2)
        assert "drug_name" in result
        assert "recommended_dose_mg" in result
        assert "rationale" in result
        assert "warnings" in result

    def test_drug_name_preserved(self):
        result = dose_recommendation_cirrhosis("Warfarin", 5.0, cl_ratio=0.75, fu_ratio=1.0)
        assert result["drug_name"] == "Warfarin"

    def test_dose_adjustment_formula(self):
        normal_dose = 100.0
        cl_ratio = 0.5
        fu_ratio = 2.0
        result = dose_recommendation_cirrhosis("Drug", normal_dose, cl_ratio, fu_ratio)
        expected = normal_dose * cl_ratio / fu_ratio
        assert abs(result["recommended_dose_mg"] - expected) < 1e-6

    def test_warnings_list(self):
        result = dose_recommendation_cirrhosis("Drug", 100.0, cl_ratio=0.5, fu_ratio=1.2)
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# simulate_cirrhosis_pk
# ---------------------------------------------------------------------------


class TestSimulateCirrhotisPK:
    def test_return_keys(self):
        result = simulate_cirrhosis_pk(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_normal_L_per_h=5.0,
            vd_L=50.0,
            child_pugh_score=8,
        )
        assert "times_h" in result
        assert "c_plasma_mg_L" in result
        assert "cmax" in result
        assert "auc" in result
        assert "cl_fibrosis" in result
        assert "f_fibrosis" in result
        assert "notes" in result

    def test_positive_cmax(self):
        result = simulate_cirrhosis_pk(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_normal_L_per_h=5.0,
            vd_L=50.0,
            child_pugh_score=7,
        )
        assert result["cmax"] > 0.0

    def test_cp_c_cl_is_25pct(self):
        cl_normal = 10.0
        result = simulate_cirrhosis_pk(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_normal_L_per_h=cl_normal,
            vd_L=50.0,
            child_pugh_score=12,
        )
        assert abs(result["cl_fibrosis"] - cl_normal * 0.25) < 1e-9

    def test_f_fibrosis_greater_than_f_oral(self):
        """Portal shunting should increase bioavailability."""
        f_oral = 0.6
        result = simulate_cirrhosis_pk(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_normal_L_per_h=5.0,
            vd_L=50.0,
            child_pugh_score=8,
            f_oral_normal=f_oral,
        )
        assert result["f_fibrosis"] > f_oral

    def test_invalid_child_pugh_score_raises(self):
        with pytest.raises(ValueError):
            simulate_cirrhosis_pk(
                drug_name="TestDrug",
                dose_mg=100.0,
                cl_normal_L_per_h=5.0,
                vd_L=50.0,
                child_pugh_score=4,
            )

    def test_auc_positive(self):
        result = simulate_cirrhosis_pk(
            drug_name="TestDrug",
            dose_mg=200.0,
            cl_normal_L_per_h=3.0,
            vd_L=30.0,
            child_pugh_score=9,
        )
        assert result["auc"] > 0.0

    def test_concentration_eventually_declines(self):
        """After Cmax, concentrations should eventually decrease."""
        result = simulate_cirrhosis_pk(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_normal_L_per_h=5.0,
            vd_L=50.0,
            child_pugh_score=6,
            t_end_h=24.0,
        )
        c = result["c_plasma_mg_L"]
        # Last concentration should be less than Cmax
        assert c[-1] < result["cmax"]
