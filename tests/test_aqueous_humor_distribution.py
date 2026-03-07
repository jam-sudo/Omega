"""Tests for Phase 907 — Drug Aqueous Humor Distribution."""

import pytest

from omega_pbpk.prediction.aqueous_humor_distribution import (
    AqueousHumorDistributionResult,
    predict_aqueous_humor_distribution,
    screen_glaucoma_candidates,
)


class TestAqueousHumorDistributionResult:
    def test_is_frozen_dataclass(self):
        result = predict_aqueous_humor_distribution("TestDrug")
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_fields_present(self):
        result = predict_aqueous_humor_distribution("DrugA")
        assert hasattr(result, "drug_name")
        assert hasattr(result, "logp")
        assert hasattr(result, "mw_Da")
        assert hasattr(result, "ah_plasma_ratio")
        assert hasattr(result, "predicted_ah_conc_ug_mL")
        assert hasattr(result, "plasma_conc_input_mg_L")
        assert hasattr(result, "ah_turnover_h")
        assert hasattr(result, "bab_permeability_class")
        assert hasattr(result, "anterior_penetration_score")
        assert hasattr(result, "iop_lowering_potential")
        assert hasattr(result, "notes")


class TestBabPermeabilityClass:
    def test_high_permeability(self):
        # bab_score = logp - mw_Da/500 = 4.0 - 200/500 = 3.6 > 1.0
        result = predict_aqueous_humor_distribution("HighPerm", logp=4.0, mw_Da=200.0)
        assert result.bab_permeability_class == "high"
        assert result.ah_plasma_ratio == pytest.approx(0.5, rel=1e-6)

    def test_moderate_permeability(self):
        # bab_score = 1.0 - 300/500 = 0.4 → moderate
        result = predict_aqueous_humor_distribution("ModPerm", logp=1.0, mw_Da=300.0)
        assert result.bab_permeability_class == "moderate"
        assert result.ah_plasma_ratio == pytest.approx(0.15, rel=1e-6)

    def test_low_permeability(self):
        # bab_score = 0.0 - 800/500 = -1.6 < -0.5 → low
        result = predict_aqueous_humor_distribution("LowPerm", logp=0.0, mw_Da=800.0)
        assert result.bab_permeability_class == "low"
        assert result.ah_plasma_ratio == pytest.approx(0.02, rel=1e-6)

    def test_boundary_moderate_to_high(self):
        # bab_score exactly 1.0: logp - mw/500 = 1.0, e.g. logp=2.5, mw=750: 2.5-1.5=1.0 → NOT > 1.0
        result = predict_aqueous_humor_distribution("Boundary", logp=2.5, mw_Da=750.0)
        assert result.bab_permeability_class == "moderate"

    def test_boundary_low_to_moderate(self):
        # bab_score exactly -0.5: logp=0.0, mw=250: 0 - 0.5 = -0.5 → NOT > -0.5
        result = predict_aqueous_humor_distribution("LowBound", logp=0.0, mw_Da=250.0)
        assert result.bab_permeability_class == "low"


class TestAhPlasmaRatioAndConcentration:
    def test_transport_multiplier(self):
        no_trans = predict_aqueous_humor_distribution("Drug", logp=2.0, mw_Da=300.0)
        with_trans = predict_aqueous_humor_distribution(
            "Drug", logp=2.0, mw_Da=300.0, has_bab_transport=True
        )
        assert with_trans.ah_plasma_ratio == pytest.approx(no_trans.ah_plasma_ratio * 3.0, rel=1e-6)

    def test_ah_ratio_clamped_high(self):
        # high perm (0.5) * 3 = 1.5, still within [0.001, 5.0]
        result = predict_aqueous_humor_distribution(
            "HighTrans", logp=4.0, mw_Da=200.0, has_bab_transport=True
        )
        assert result.ah_plasma_ratio <= 5.0
        assert result.ah_plasma_ratio == pytest.approx(1.5, rel=1e-6)

    def test_ah_conc_calculation(self):
        # moderate: ratio=0.15, plasma=2.0, fu=0.5 → 0.15*2.0*0.5*1000=150
        result = predict_aqueous_humor_distribution(
            "Drug", logp=1.0, mw_Da=300.0, plasma_conc_mg_L=2.0, fu_plasma=0.5
        )
        assert result.predicted_ah_conc_ug_mL == pytest.approx(150.0, rel=1e-6)

    def test_ah_conc_with_fu(self):
        # low perm: ratio=0.02, plasma=1.0, fu=1.0 → 0.02*1.0*1.0*1000=20
        result = predict_aqueous_humor_distribution(
            "Drug", logp=0.0, mw_Da=800.0, plasma_conc_mg_L=1.0, fu_plasma=1.0
        )
        assert result.predicted_ah_conc_ug_mL == pytest.approx(20.0, rel=1e-6)


class TestAhTurnoverAndScore:
    def test_ah_turnover_fixed(self):
        result = predict_aqueous_humor_distribution("Drug")
        assert result.ah_turnover_h == pytest.approx(1.5, rel=1e-6)

    def test_anterior_penetration_score_no_transport(self):
        # high perm: ratio=0.5 → score = min(100, 50)=50, no bonus
        result = predict_aqueous_humor_distribution("Drug", logp=4.0, mw_Da=200.0)
        assert result.anterior_penetration_score == pytest.approx(50.0, rel=1e-6)

    def test_anterior_penetration_score_with_transport(self):
        # high perm + transport: ratio=1.5 → 150+20=170 → clamped to 100
        result = predict_aqueous_humor_distribution(
            "Drug", logp=4.0, mw_Da=200.0, has_bab_transport=True
        )
        assert result.anterior_penetration_score == pytest.approx(100.0, rel=1e-6)

    def test_iop_lowering_potential_true(self):
        # transport bonus pushes above 50
        result = predict_aqueous_humor_distribution(
            "Drug", logp=4.0, mw_Da=200.0, has_bab_transport=True
        )
        assert result.iop_lowering_potential is True

    def test_iop_lowering_potential_false(self):
        # low perm: score=0.02*100=2 < 50
        result = predict_aqueous_humor_distribution("Drug", logp=0.0, mw_Da=800.0)
        assert result.iop_lowering_potential is False


class TestNotes:
    def test_notes_iop_lowering(self):
        result = predict_aqueous_humor_distribution(
            "Drug", logp=4.0, mw_Da=200.0, has_bab_transport=True
        )
        assert "glaucoma" in result.notes.lower()

    def test_notes_low_perm(self):
        result = predict_aqueous_humor_distribution("Drug", logp=0.0, mw_Da=800.0)
        assert "topical" in result.notes.lower()

    def test_notes_high_bab_no_iop(self):
        # high perm (score=50) but NOT > 50 → iop_lowering=False, bab_class=high
        result = predict_aqueous_humor_distribution("Drug", logp=4.0, mw_Da=200.0)
        assert result.iop_lowering_potential is False
        assert result.bab_permeability_class == "high"
        assert "high bab" in result.notes.lower()


class TestValidation:
    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_aqueous_humor_distribution("Drug", mw_Da=0.0)

    def test_invalid_plasma_conc(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            predict_aqueous_humor_distribution("Drug", plasma_conc_mg_L=0.0)

    def test_negative_mw(self):
        with pytest.raises(ValueError):
            predict_aqueous_humor_distribution("Drug", mw_Da=-1.0)


class TestScreenGlaucomaCandidates:
    def test_sorted_by_score_descending(self):
        candidates = [
            {"name": "LowPerm", "logp": 0.0, "mw_Da": 800.0},
            {"name": "HighPerm", "logp": 4.0, "mw_Da": 200.0},
            {"name": "Transport", "logp": 4.0, "mw_Da": 200.0, "has_bab_transport": True},
        ]
        results = screen_glaucoma_candidates(candidates)
        scores = [r.anterior_penetration_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_default_mw_applied(self):
        candidates = [{"name": "NoDrugMW", "logp": 2.0}]
        results = screen_glaucoma_candidates(candidates)
        assert results[0].mw_Da == 300.0

    def test_empty_candidates(self):
        results = screen_glaucoma_candidates([])
        assert results == []

    def test_returns_result_objects(self):
        candidates = [{"name": "Timolol", "logp": 1.8, "mw_Da": 316.4}]
        results = screen_glaucoma_candidates(candidates)
        assert isinstance(results[0], AqueousHumorDistributionResult)
