"""Tests for Phase 908 — Drug Vitreous Humor Distribution."""

import pytest

from omega_pbpk.prediction.vitreous_humor_distribution import (
    compare_intravitreal_doses,
    predict_vitreous_distribution,
)


class TestVitreousHumorDistributionResult:
    def test_is_frozen_dataclass(self):
        result = predict_vitreous_distribution("TestDrug")
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_fields_present(self):
        result = predict_vitreous_distribution("DrugA")
        assert hasattr(result, "drug_name")
        assert hasattr(result, "logp")
        assert hasattr(result, "mw_Da")
        assert hasattr(result, "route")
        assert hasattr(result, "dose_mg")
        assert hasattr(result, "predicted_vh_cmax_ug_mL")
        assert hasattr(result, "predicted_vh_t_half_days")
        assert hasattr(result, "brb_plasma_to_vh_ratio")
        assert hasattr(result, "monthly_dosing_feasible")
        assert hasattr(result, "anti_vegf_suitable")
        assert hasattr(result, "retinal_penetration_fraction")
        assert hasattr(result, "notes")


class TestIntravitreousCmax:
    def test_1mg_dose(self):
        # 1 mg * 1000 / 4.0 mL = 250 µg/mL
        result = predict_vitreous_distribution("Drug", dose_mg=1.0, route="intravitreal")
        assert result.predicted_vh_cmax_ug_mL == pytest.approx(250.0, rel=1e-6)

    def test_2mg_dose(self):
        result = predict_vitreous_distribution("Drug", dose_mg=2.0, route="intravitreal")
        assert result.predicted_vh_cmax_ug_mL == pytest.approx(500.0, rel=1e-6)

    def test_05mg_dose(self):
        result = predict_vitreous_distribution("Drug", dose_mg=0.5, route="intravitreal")
        assert result.predicted_vh_cmax_ug_mL == pytest.approx(125.0, rel=1e-6)

    def test_intravitreal_brb_ratio_zero(self):
        result = predict_vitreous_distribution("Drug", route="intravitreal")
        assert result.brb_plasma_to_vh_ratio == 0.0


class TestSystemicRoute:
    def test_small_mw_brb_ratio(self):
        # mw < 300 → brb_ratio = 0.1, plasma=1.0 → cmax = 100 µg/mL
        result = predict_vitreous_distribution(
            "Small", mw_Da=200.0, route="systemic", plasma_conc_mg_L=1.0
        )
        assert result.predicted_vh_cmax_ug_mL == pytest.approx(100.0, rel=1e-6)
        assert result.brb_plasma_to_vh_ratio == pytest.approx(0.1, rel=1e-6)

    def test_medium_mw_brb_ratio(self):
        # mw between 300-1000 → brb_ratio = 0.05, plasma=2.0 → cmax = 100 µg/mL
        result = predict_vitreous_distribution(
            "Medium", mw_Da=500.0, route="systemic", plasma_conc_mg_L=2.0
        )
        assert result.predicted_vh_cmax_ug_mL == pytest.approx(100.0, rel=1e-6)
        assert result.brb_plasma_to_vh_ratio == pytest.approx(0.05, rel=1e-6)

    def test_large_mw_brb_ratio(self):
        # mw > 1000 → brb_ratio = 0.01, plasma=1.0 → cmax = 10 µg/mL
        result = predict_vitreous_distribution(
            "Large", mw_Da=150000.0, route="systemic", plasma_conc_mg_L=1.0
        )
        assert result.predicted_vh_cmax_ug_mL == pytest.approx(10.0, rel=1e-6)
        assert result.brb_plasma_to_vh_ratio == pytest.approx(0.01, rel=1e-6)

    def test_systemic_default_plasma(self):
        # default plasma = 1.0 if None
        result_explicit = predict_vitreous_distribution(
            "Drug", mw_Da=500.0, route="systemic", plasma_conc_mg_L=1.0
        )
        result_default = predict_vitreous_distribution(
            "Drug", mw_Da=500.0, route="systemic", plasma_conc_mg_L=None
        )
        assert result_explicit.predicted_vh_cmax_ug_mL == pytest.approx(
            result_default.predicted_vh_cmax_ug_mL, rel=1e-6
        )


class TestHalfLife:
    def test_small_mol_t_half_basic(self):
        # logp=2.0 → 3.0 + 2.0*0.5 = 4.0 days
        result = predict_vitreous_distribution("Drug", logp=2.0, mw_Da=300.0)
        assert result.predicted_vh_t_half_days == pytest.approx(4.0, rel=1e-6)

    def test_small_mol_clamp_min(self):
        # logp=-10 → 3.0 + (-10)*0.5 = -2 → clamped to 0.5
        result = predict_vitreous_distribution("Drug", logp=-10.0, mw_Da=300.0)
        assert result.predicted_vh_t_half_days == pytest.approx(0.5, rel=1e-6)

    def test_small_mol_clamp_max(self):
        # logp=20 → 3.0 + 10 = 13 → clamped to 10
        result = predict_vitreous_distribution("Drug", logp=20.0, mw_Da=300.0)
        assert result.predicted_vh_t_half_days == pytest.approx(10.0, rel=1e-6)

    def test_large_mol_t_half_antibody(self):
        # mw=150000 Da → 20 + (150000/150000)*10 = 30.0
        result = predict_vitreous_distribution("mAb", mw_Da=150000.0)
        assert result.predicted_vh_t_half_days == pytest.approx(30.0, rel=1e-6)

    def test_large_mol_clamp_min(self):
        # mw = 1001 → 20 + (1001/150000)*10 ≈ 20.067 → above 10
        result = predict_vitreous_distribution("BigDrug", mw_Da=1001.0)
        assert result.predicted_vh_t_half_days >= 10.0

    def test_large_mol_clamp_max(self):
        # very large mw
        result = predict_vitreous_distribution("Huge", mw_Da=999999.0)
        assert result.predicted_vh_t_half_days <= 60.0


class TestRetinalPenetration:
    def test_small_mol_retinal_fraction(self):
        result = predict_vitreous_distribution("Small", mw_Da=500.0)
        assert result.retinal_penetration_fraction == pytest.approx(0.3, rel=1e-6)

    def test_large_mol_retinal_fraction(self):
        result = predict_vitreous_distribution("Large", mw_Da=150000.0)
        assert result.retinal_penetration_fraction == pytest.approx(0.1, rel=1e-6)


class TestBooleanFlags:
    def test_monthly_dosing_feasible_true(self):
        # mAb with t_half > 15
        result = predict_vitreous_distribution("mAb", mw_Da=150000.0)
        assert result.monthly_dosing_feasible is True

    def test_monthly_dosing_feasible_false(self):
        # small mol logp=2 → t_half=4 < 15
        result = predict_vitreous_distribution("Small", logp=2.0, mw_Da=300.0)
        assert result.monthly_dosing_feasible is False

    def test_anti_vegf_suitable_true(self):
        # mw > 10000, t_half > 20 (mAb)
        result = predict_vitreous_distribution("Ranibizumab", mw_Da=48000.0)
        assert result.anti_vegf_suitable is True

    def test_anti_vegf_suitable_false_small_mol(self):
        result = predict_vitreous_distribution("Small", mw_Da=300.0)
        assert result.anti_vegf_suitable is False


class TestNotes:
    def test_notes_anti_vegf(self):
        result = predict_vitreous_distribution("Bevacizumab", mw_Da=149000.0)
        assert "anti-vegf" in result.notes.lower()

    def test_notes_monthly(self):
        # mw = 1500, t_half = 20 + (1500/150000)*10 = 20.1 > 15 → monthly
        result = predict_vitreous_distribution("Peptide", mw_Da=1500.0)
        assert result.monthly_dosing_feasible
        # anti_vegf: mw > 10000? No (1500 < 10000) so monthly note
        assert "monthly" in result.notes.lower()

    def test_notes_frequent(self):
        result = predict_vitreous_distribution("SmallMol", logp=2.0, mw_Da=300.0)
        assert "frequent" in result.notes.lower() or "sustained" in result.notes.lower()


class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_vitreous_distribution("Drug", dose_mg=0.0)

    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_vitreous_distribution("Drug", mw_Da=-1.0)

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            predict_vitreous_distribution("Drug", route="oral")


class TestCompareIntravitreousDoses:
    def test_sorted_by_cmax_descending(self):
        results = compare_intravitreal_doses("Drug", doses_mg=[0.5, 1.0, 2.0])
        cmaxes = [r.predicted_vh_cmax_ug_mL for r in results]
        assert cmaxes == sorted(cmaxes, reverse=True)

    def test_all_intravitreal_route(self):
        results = compare_intravitreal_doses("Drug", doses_mg=[1.0, 2.0])
        for r in results:
            assert r.route == "intravitreal"

    def test_dose_values_preserved(self):
        doses = [0.1, 0.5, 1.0, 2.0, 4.0]
        results = compare_intravitreal_doses("Drug", doses_mg=doses)
        result_doses = sorted([r.dose_mg for r in results])
        assert result_doses == pytest.approx(sorted(doses), rel=1e-6)

    def test_single_dose(self):
        results = compare_intravitreal_doses("Drug", doses_mg=[1.0])
        assert len(results) == 1
        assert results[0].predicted_vh_cmax_ug_mL == pytest.approx(250.0, rel=1e-6)
