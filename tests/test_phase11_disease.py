"""Phase 11 — Disease state PBPK tests.

Tests cover:
- RenalImpairment: eGFR scaling, albumin correction, on-dialysis
- HepaticImpairment: CYP activity, hepatic flow, shunt fractions, albumin
- HeartFailure: CO fraction, hepatic flow
- ObesityState: BMI, IBW, ABW, adipose scaling
- PregnancyState: CO, GFR, CYP factors per trimester
- DiseaseState composite
- apply_disease_scaling(): correct parameter propagation
- Integration with WholeBodyPBPK: severe RI → higher AUC for renally-cleared drug
"""

import pytest

from omega_pbpk.clinical.disease import (
    ChildPughClass,
    DiseaseState,
    HeartFailure,
    HepaticImpairment,
    NyhaClass,
    ObesityState,
    PregnancyState,
    RenalImpairment,
    RenalImpairmentStage,
    TrimesterStage,
    apply_disease_scaling,
)
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# RenalImpairment
# ---------------------------------------------------------------------------


class TestRenalImpairment:
    def test_normal_clr_fraction_one(self):
        ri = RenalImpairment(stage=RenalImpairmentStage.NORMAL)
        assert ri.clr_fraction == pytest.approx(1.0, rel=0.05)

    def test_esrd_clr_fraction_very_low(self):
        ri = RenalImpairment(stage=RenalImpairmentStage.ESRD)
        assert ri.clr_fraction < 0.15

    def test_moderate_clr_fraction_intermediate(self):
        ri = RenalImpairment(stage=RenalImpairmentStage.MODERATE)
        assert 0.25 < ri.clr_fraction < 0.75

    def test_dialysis_gives_zero_clr(self):
        ri = RenalImpairment(on_dialysis=True)
        assert ri.clr_fraction == 0.0

    def test_custom_egfr_overrides_stage(self):
        ri = RenalImpairment(stage=RenalImpairmentStage.NORMAL, egfr_mL_min=30.0)
        # eGFR 30 / 110 ≈ 0.27
        assert ri.clr_fraction == pytest.approx(30.0 / 110.0, rel=0.01)

    def test_clr_fraction_bounded_by_one(self):
        ri = RenalImpairment(egfr_mL_min=200.0)  # supranormal
        assert ri.clr_fraction <= 1.0

    def test_low_albumin_increases_fup_scale(self):
        ri_low = RenalImpairment(albumin_g_dL=2.0)
        ri_norm = RenalImpairment(albumin_g_dL=4.0)
        p_low = apply_disease_scaling(disease=DiseaseState(renal=ri_low))
        p_norm = apply_disease_scaling(disease=DiseaseState(renal=ri_norm))
        # Lower albumin → more unbound drug → higher fup_scale
        assert p_low.fup_scale > p_norm.fup_scale

    def test_normal_albumin_no_fup_change(self):
        ri = RenalImpairment(albumin_g_dL=4.0)  # normal
        params = apply_disease_scaling(disease=DiseaseState(renal=ri))
        assert params.fup_scale == pytest.approx(1.0, abs=1e-9)

    def test_clr_fraction_monotone_decreasing(self):
        stages = [
            RenalImpairmentStage.NORMAL,
            RenalImpairmentStage.MILD,
            RenalImpairmentStage.MODERATE,
            RenalImpairmentStage.SEVERE,
            RenalImpairmentStage.ESRD,
        ]
        fracs = [RenalImpairment(stage=s).clr_fraction for s in stages]
        for i in range(len(fracs) - 1):
            assert fracs[i] >= fracs[i + 1]


# ---------------------------------------------------------------------------
# HepaticImpairment
# ---------------------------------------------------------------------------


class TestHepaticImpairment:
    def test_normal_cyp_full_activity(self):
        hi = HepaticImpairment(cp_class=ChildPughClass.NORMAL)
        assert hi.cyp_activity_fraction == pytest.approx(1.0)

    def test_child_pugh_c_severe_activity_reduction(self):
        hi = HepaticImpairment(cp_class=ChildPughClass.C)
        assert hi.cyp_activity_fraction <= 0.30

    def test_cyp_activity_monotone_decreasing(self):
        a = HepaticImpairment(cp_class=ChildPughClass.A).cyp_activity_fraction
        b = HepaticImpairment(cp_class=ChildPughClass.B).cyp_activity_fraction
        c = HepaticImpairment(cp_class=ChildPughClass.C).cyp_activity_fraction
        assert a > b > c

    def test_hepatic_flow_reduced_in_cp_c(self):
        hi_norm = HepaticImpairment(cp_class=ChildPughClass.NORMAL)
        hi_c = HepaticImpairment(cp_class=ChildPughClass.C)
        assert hi_c.hepatic_flow_fraction < hi_norm.hepatic_flow_fraction

    def test_portal_shunt_increases_with_severity(self):
        shunt_a = HepaticImpairment(cp_class=ChildPughClass.A).effective_shunt_fraction
        shunt_c = HepaticImpairment(cp_class=ChildPughClass.C).effective_shunt_fraction
        assert shunt_c > shunt_a

    def test_custom_shunt_overrides_cp_class(self):
        hi = HepaticImpairment(cp_class=ChildPughClass.A, portal_shunt_fraction=0.50)
        assert hi.effective_shunt_fraction == pytest.approx(0.50)

    def test_low_albumin_increases_fup_in_scaling(self):
        hi = HepaticImpairment(cp_class=ChildPughClass.C, albumin_g_dL=2.0)
        params = apply_disease_scaling(disease=DiseaseState(hepatic=hi))
        assert params.fup_scale > 1.0


# ---------------------------------------------------------------------------
# HeartFailure
# ---------------------------------------------------------------------------


class TestHeartFailure:
    def test_normal_co_fraction_one(self):
        hf = HeartFailure(nyha_class=NyhaClass.NORMAL)
        assert hf.cardiac_output_fraction == pytest.approx(1.0)

    def test_nyha_iv_co_fraction_low(self):
        hf = HeartFailure(nyha_class=NyhaClass.CLASS_IV)
        assert hf.cardiac_output_fraction <= 0.55

    def test_co_fraction_monotone_decreasing(self):
        classes = [
            NyhaClass.NORMAL,
            NyhaClass.CLASS_I,
            NyhaClass.CLASS_II,
            NyhaClass.CLASS_III,
            NyhaClass.CLASS_IV,
        ]
        fracs = [HeartFailure(nyha_class=c).cardiac_output_fraction for c in classes]
        for i in range(len(fracs) - 1):
            assert fracs[i] >= fracs[i + 1]

    def test_hepatic_flow_reduced_in_severe_hf(self):
        hf_norm = HeartFailure(nyha_class=NyhaClass.NORMAL)
        hf_iv = HeartFailure(nyha_class=NyhaClass.CLASS_IV)
        assert hf_iv.hepatic_flow_fraction < hf_norm.hepatic_flow_fraction

    def test_scaled_params_co_reduced(self):
        hf = HeartFailure(nyha_class=NyhaClass.CLASS_III)
        params = apply_disease_scaling(disease=DiseaseState(heart_failure=hf))
        assert params.cardiac_output_scale < 1.0


# ---------------------------------------------------------------------------
# ObesityState
# ---------------------------------------------------------------------------


class TestObesityState:
    def test_normal_bmi_not_obese(self):
        ob = ObesityState(body_weight_kg=70.0, height_cm=175.0)
        assert not ob.is_obese
        assert ob.bmi == pytest.approx(70.0 / 1.75**2, rel=0.01)

    def test_high_bmi_is_obese(self):
        ob = ObesityState(body_weight_kg=120.0, height_cm=170.0)
        assert ob.is_obese
        assert ob.bmi >= 30.0

    def test_ibw_male_formula(self):
        ob = ObesityState(body_weight_kg=80.0, height_cm=170.0, sex="male")
        # Devine: IBW = 50 + 0.906 × (170 - 152.4) ≈ 50 + 15.96 ≈ 65.96
        expected = 50.0 + 0.906 * (170.0 - 152.4)
        assert ob.ibw_kg == pytest.approx(expected, rel=0.01)

    def test_ibw_female_formula(self):
        ob = ObesityState(body_weight_kg=80.0, height_cm=160.0, sex="female")
        expected = 45.5 + 0.906 * (160.0 - 152.4)
        assert ob.ibw_kg == pytest.approx(expected, rel=0.01)

    def test_abw_normal_weight_equals_tbw(self):
        ob = ObesityState(body_weight_kg=65.0, height_cm=175.0)
        # TBW ≤ IBW → ABW = TBW
        assert ob.adjusted_bw_kg == pytest.approx(ob.body_weight_kg, rel=0.01)

    def test_abw_obese_between_ibw_and_tbw(self):
        ob = ObesityState(body_weight_kg=120.0, height_cm=170.0)
        assert ob.ibw_kg < ob.adjusted_bw_kg < ob.body_weight_kg

    def test_adipose_scale_greater_one_for_obese(self):
        ob = ObesityState(body_weight_kg=120.0, height_cm=170.0)
        assert ob.adipose_volume_scale > 1.0

    def test_adipose_scale_near_one_for_normal(self):
        ob = ObesityState(body_weight_kg=70.0, height_cm=175.0)
        assert ob.adipose_volume_scale == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------------------------
# PregnancyState
# ---------------------------------------------------------------------------


class TestPregnancyState:
    def test_non_pregnant_all_factors_normal(self):
        pg = PregnancyState(trimester=TrimesterStage.NON_PREGNANT)
        assert pg.cardiac_output_factor == pytest.approx(1.0)
        assert pg.gfr_factor == pytest.approx(1.0)
        assert pg.cyp3a4_factor == pytest.approx(1.0)
        assert pg.cyp1a2_factor == pytest.approx(1.0)

    def test_t3_co_elevated(self):
        pg = PregnancyState(trimester=TrimesterStage.T3)
        assert pg.cardiac_output_factor > 1.3

    def test_t3_gfr_elevated(self):
        pg = PregnancyState(trimester=TrimesterStage.T3)
        assert pg.gfr_factor >= 1.4

    def test_cyp3a4_induced_in_pregnancy(self):
        pg = PregnancyState(trimester=TrimesterStage.T3)
        assert pg.cyp3a4_factor > 1.5

    def test_cyp1a2_suppressed_in_pregnancy(self):
        pg = PregnancyState(trimester=TrimesterStage.T3)
        assert pg.cyp1a2_factor < 1.0

    def test_co_increases_through_trimesters(self):
        t1 = PregnancyState(trimester=TrimesterStage.T1).cardiac_output_factor
        t2 = PregnancyState(trimester=TrimesterStage.T2).cardiac_output_factor
        t3 = PregnancyState(trimester=TrimesterStage.T3).cardiac_output_factor
        assert t1 <= t2 <= t3

    def test_pregnancy_notes_populated(self):
        pg = PregnancyState(trimester=TrimesterStage.T2)
        params = apply_disease_scaling(disease=DiseaseState(pregnancy=pg))
        assert any("Pregnancy" in n for n in params.notes)


# ---------------------------------------------------------------------------
# apply_disease_scaling — combined effects
# ---------------------------------------------------------------------------


class TestApplyDiseaseScaling:
    def test_healthy_returns_defaults(self):
        params = apply_disease_scaling()
        assert params.clr_scale == pytest.approx(1.0)
        assert params.clint_scale == pytest.approx(1.0)
        assert params.fup_scale == pytest.approx(1.0)
        assert params.cardiac_output_scale == pytest.approx(1.0)

    def test_none_disease_returns_defaults(self):
        params = apply_disease_scaling(disease=None)
        assert params.clr_scale == pytest.approx(1.0)

    def test_ri_reduces_clr(self):
        ri = RenalImpairment(stage=RenalImpairmentStage.SEVERE)
        params = apply_disease_scaling(disease=DiseaseState(renal=ri))
        assert params.clr_scale < 0.5

    def test_hi_reduces_clint(self):
        hi = HepaticImpairment(cp_class=ChildPughClass.C)
        params = apply_disease_scaling(disease=DiseaseState(hepatic=hi))
        assert params.clint_scale <= 0.30

    def test_hf_reduces_co(self):
        hf = HeartFailure(nyha_class=NyhaClass.CLASS_IV)
        params = apply_disease_scaling(disease=DiseaseState(heart_failure=hf))
        assert params.cardiac_output_scale < 0.6

    def test_combined_ri_hi_stacks(self):
        disease = DiseaseState(
            renal=RenalImpairment(stage=RenalImpairmentStage.MODERATE),
            hepatic=HepaticImpairment(cp_class=ChildPughClass.B),
        )
        params = apply_disease_scaling(disease=disease)
        assert params.clr_scale < 0.6
        assert params.clint_scale < 0.6
        assert len(params.notes) == 2

    def test_pregnancy_scales_co_and_clr(self):
        pg = PregnancyState(trimester=TrimesterStage.T3)
        params = apply_disease_scaling(disease=DiseaseState(pregnancy=pg))
        assert params.cardiac_output_scale > 1.3
        assert params.clr_scale > 1.3

    def test_pregnancy_sets_cyp_per_enzyme(self):
        pg = PregnancyState(trimester=TrimesterStage.T3)
        params = apply_disease_scaling(disease=DiseaseState(pregnancy=pg))
        assert "CYP3A4" in params.clint_per_enzyme
        assert params.clint_per_enzyme["CYP3A4"] > 1.0
        assert params.clint_per_enzyme["CYP1A2"] < 1.0

    def test_obesity_uses_abw(self):
        ob = ObesityState(body_weight_kg=120.0, height_cm=170.0)
        params = apply_disease_scaling(body_weight=120.0, disease=DiseaseState(obesity=ob))
        assert params.body_weight_kg < 120.0  # ABW < TBW
        assert params.body_weight_kg > ob.ibw_kg  # ABW > IBW

    def test_notes_populated_for_each_condition(self):
        disease = DiseaseState(
            renal=RenalImpairment(stage=RenalImpairmentStage.MODERATE),
            hepatic=HepaticImpairment(cp_class=ChildPughClass.B),
            heart_failure=HeartFailure(nyha_class=NyhaClass.CLASS_II),
        )
        params = apply_disease_scaling(disease=disease)
        assert len(params.notes) == 3


# ---------------------------------------------------------------------------
# PBPK integration — renal impairment raises AUC of renally-cleared drug
# ---------------------------------------------------------------------------


class TestDiseaseStatePBPKIntegration:
    def _run_iv(self, clr_fraction=1.0, dose=100.0):
        """Run IV simulation with scaled renal clearance."""
        drug = Drug(
            name="RenalDrug",
            mw=300.0,
            logP=-1.0,
            fup=0.8,
            rbp=1.0,
            clint_hepatic_L_per_h=0.1,  # minimal hepatic
            clr_L_per_h=10.0 * clr_fraction,  # mainly renal
        )
        model = WholeBodyPBPK(drug, body_weight=70.0)
        model.setup_iv(dose_mg=dose)
        return model.simulate(t_end_h=24.0)

    def test_severe_ri_increases_auc(self):
        ri = RenalImpairment(stage=RenalImpairmentStage.SEVERE)
        params = apply_disease_scaling(disease=DiseaseState(renal=ri))
        result_normal = self._run_iv(clr_fraction=1.0)
        result_ri = self._run_iv(clr_fraction=params.clr_scale)
        assert result_ri.pk_summary()["AUC_mg_h_L"] > result_normal.pk_summary()["AUC_mg_h_L"]

    def test_esrd_substantially_higher_auc(self):
        ri = RenalImpairment(stage=RenalImpairmentStage.ESRD)
        params = apply_disease_scaling(disease=DiseaseState(renal=ri))
        result_normal = self._run_iv(clr_fraction=1.0)
        result_esrd = self._run_iv(clr_fraction=params.clr_scale)
        auc_ratio = (
            result_esrd.pk_summary()["AUC_mg_h_L"] / result_normal.pk_summary()["AUC_mg_h_L"]
        )
        assert auc_ratio > 3.0  # expect > 3× increase

    def test_hi_increases_auc_hepatic_drug(self):
        """Hepatic impairment → lower CLint → higher AUC for CYP3A4 substrate."""

        def run(clint_scale=1.0):
            drug = Drug(
                name="HepaticDrug",
                mw=400.0,
                logP=2.5,
                fup=0.2,
                rbp=1.0,
                clint_hepatic_L_per_h=8.0 * clint_scale,
                clr_L_per_h=0.1,
            )
            model = WholeBodyPBPK(drug)
            model.setup_iv(100.0)
            return model.simulate(t_end_h=24.0)

        hi = HepaticImpairment(cp_class=ChildPughClass.C)
        params = apply_disease_scaling(disease=DiseaseState(hepatic=hi))
        r_normal = run(clint_scale=1.0)
        r_hi = run(clint_scale=params.clint_scale)
        assert r_hi.pk_summary()["AUC_mg_h_L"] > r_normal.pk_summary()["AUC_mg_h_L"]
