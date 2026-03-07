"""Tests for drug potency and selectivity analysis (Phase 327)."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.potency_selectivity import (
    LEResult,
    SelectivityResult,
    compute_selectivity_ratio,
    ligand_efficiency,
    potency_optimization_score,
    selectivity_profile,
)

# ---------------------------------------------------------------------------
# ligand_efficiency tests
# ---------------------------------------------------------------------------


class TestLigandEfficiency:
    def test_returns_le_result(self):
        result = ligand_efficiency(pIC50=7.0, mw=350.0, logP=2.0)
        assert isinstance(result, LEResult)

    def test_le_calculation_explicit_heavy_atoms(self):
        result = ligand_efficiency(pIC50=9.0, mw=350.0, logP=2.0, n_heavy_atoms=25)
        assert result.le == pytest.approx(9.0 / 25, rel=1e-6)

    def test_le_high_category(self):
        # LE = 9/25 = 0.36 > 0.3 → high
        result = ligand_efficiency(pIC50=9.0, mw=350.0, logP=2.0, n_heavy_atoms=25)
        assert result.le_category == "high"

    def test_le_moderate_category(self):
        # LE = 5/20 = 0.25 → moderate
        result = ligand_efficiency(pIC50=5.0, mw=260.0, logP=1.0, n_heavy_atoms=20)
        assert result.le_category == "moderate"

    def test_le_low_category(self):
        # LE = 5/40 = 0.125 → low
        result = ligand_efficiency(pIC50=5.0, mw=520.0, logP=4.0, n_heavy_atoms=40)
        assert result.le_category == "low"

    def test_lipe_calculation(self):
        result = ligand_efficiency(pIC50=8.0, mw=300.0, logP=3.0, n_heavy_atoms=20)
        assert result.lipe == pytest.approx(8.0 - 3.0, rel=1e-6)

    def test_lelp_calculation(self):
        result = ligand_efficiency(pIC50=8.0, mw=300.0, logP=3.0, n_heavy_atoms=20)
        le = 8.0 / 20
        expected_lelp = 3.0 / le
        assert result.lelp == pytest.approx(expected_lelp, rel=1e-6)

    def test_n_heavy_atoms_estimated_from_mw(self):
        # mw=390 → n_heavy = round(390/13) = 30
        result = ligand_efficiency(pIC50=9.0, mw=390.0, logP=2.0)
        assert result.n_heavy_atoms == 30
        assert result.le == pytest.approx(9.0 / 30, rel=1e-6)

    def test_n_heavy_atoms_explicit_overrides_mw(self):
        result = ligand_efficiency(pIC50=9.0, mw=390.0, logP=2.0, n_heavy_atoms=35)
        assert result.n_heavy_atoms == 35

    def test_result_is_frozen(self):
        result = ligand_efficiency(pIC50=7.0, mw=350.0, logP=2.0)
        with pytest.raises((AttributeError, TypeError)):
            result.le = 0.99  # type: ignore[misc]

    def test_invalid_pic50_zero(self):
        with pytest.raises(ValueError, match="pIC50 must be > 0"):
            ligand_efficiency(pIC50=0.0, mw=300.0, logP=2.0)

    def test_invalid_pic50_negative(self):
        with pytest.raises(ValueError, match="pIC50 must be > 0"):
            ligand_efficiency(pIC50=-1.0, mw=300.0, logP=2.0)

    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw must be > 0"):
            ligand_efficiency(pIC50=7.0, mw=0.0, logP=2.0)

    def test_notes_not_empty(self):
        result = ligand_efficiency(pIC50=7.0, mw=350.0, logP=2.0)
        assert len(result.notes) > 0

    def test_stored_fields_match(self):
        result = ligand_efficiency(pIC50=7.5, mw=320.0, logP=2.5, n_heavy_atoms=22)
        assert result.pIC50 == 7.5
        assert result.mw == 320.0
        assert result.logP == 2.5
        assert result.n_heavy_atoms == 22


# ---------------------------------------------------------------------------
# compute_selectivity_ratio tests
# ---------------------------------------------------------------------------


class TestComputeSelectivityRatio:
    def test_basic_ratio(self):
        ratio = compute_selectivity_ratio(1.0, 100.0)
        assert ratio == pytest.approx(100.0)

    def test_selectivity_100x(self):
        ratio = compute_selectivity_ratio(10.0, 1000.0)
        assert ratio == pytest.approx(100.0)

    def test_nonselective_below_10(self):
        ratio = compute_selectivity_ratio(10.0, 50.0)
        assert ratio == pytest.approx(5.0)

    def test_invalid_target_ic50_zero(self):
        with pytest.raises(ValueError, match="ic50_target_nM must be > 0"):
            compute_selectivity_ratio(0.0, 100.0)

    def test_invalid_offtarget_ic50_negative(self):
        with pytest.raises(ValueError, match="ic50_off_target_nM must be > 0"):
            compute_selectivity_ratio(10.0, -1.0)


# ---------------------------------------------------------------------------
# selectivity_profile tests
# ---------------------------------------------------------------------------


class TestSelectivityProfile:
    def _make_panel(self):
        return {
            "CYP3A4": 1.0,
            "CYP2D6": 500.0,
            "hERG": 50.0,
            "COX2": 2000.0,
        }

    def test_returns_selectivity_result(self):
        result = selectivity_profile("CompA", self._make_panel())
        assert isinstance(result, SelectivityResult)

    def test_primary_target_is_first_key(self):
        result = selectivity_profile("CompA", self._make_panel())
        assert result.primary_target == "CYP3A4"

    def test_primary_ic50_correct(self):
        result = selectivity_profile("CompA", self._make_panel())
        assert result.primary_ic50_nM == pytest.approx(1.0)

    def test_selectivity_ratios_computed(self):
        result = selectivity_profile("CompA", self._make_panel())
        assert "hERG" in result.selectivity_ratios
        assert result.selectivity_ratios["hERG"] == pytest.approx(50.0)

    def test_most_selective_off_target(self):
        result = selectivity_profile("CompA", self._make_panel())
        assert result.most_selective_off_target == "COX2"

    def test_worst_selectivity_ratio(self):
        result = selectivity_profile("CompA", self._make_panel())
        # Lowest ratio is hERG: 50/1 = 50
        assert result.worst_selectivity_ratio == pytest.approx(50.0)

    def test_overall_class_selective(self):
        # worst ratio = 50 -> moderate
        result = selectivity_profile("CompA", self._make_panel())
        assert result.overall_selectivity_class == "moderate"

    def test_overall_class_nonselective(self):
        panel = {"TargetA": 1.0, "OffTarget": 5.0}
        result = selectivity_profile("CompX", panel)
        assert result.overall_selectivity_class == "nonselective"

    def test_overall_class_selective_high(self):
        panel = {"TargetA": 1.0, "OffTarget1": 200.0, "OffTarget2": 500.0}
        result = selectivity_profile("CompX", panel)
        assert result.overall_selectivity_class == "selective"

    def test_compound_name_stored(self):
        result = selectivity_profile("DrugX", self._make_panel())
        assert result.compound_name == "DrugX"

    def test_invalid_ic50_zero(self):
        with pytest.raises(ValueError, match="IC50 for"):
            selectivity_profile("Bad", {"TargetA": 0.0, "OffTarget": 100.0})

    def test_single_target_returns_result(self):
        result = selectivity_profile("OnlyOne", {"TargetA": 10.0})
        assert result.selectivity_ratios == {}
        assert result.overall_selectivity_class == "selective"

    def test_result_is_frozen(self):
        result = selectivity_profile("CompA", self._make_panel())
        with pytest.raises((AttributeError, TypeError)):
            result.compound_name = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# potency_optimization_score tests
# ---------------------------------------------------------------------------


class TestPotencyOptimizationScore:
    def test_returns_float_in_0_1(self):
        score = potency_optimization_score(pIC50=8.0, logP=2.0, mw=350.0, hbd=2, psa=90.0)
        assert 0.0 <= score <= 1.0

    def test_ideal_compound_scores_high(self):
        # High LE, moderate LipE, low MW, low HBD, moderate PSA
        score = potency_optimization_score(
            pIC50=9.0, logP=2.0, mw=250.0, hbd=1, psa=90.0, n_heavy_atoms=20
        )
        assert score > 0.5

    def test_poor_compound_scores_low(self):
        # Low LE, high logP, high MW, high HBD, high PSA
        score = potency_optimization_score(
            pIC50=4.0, logP=7.0, mw=600.0, hbd=8, psa=180.0, n_heavy_atoms=45
        )
        assert score < 0.4

    def test_n_heavy_atoms_can_be_provided(self):
        s1 = potency_optimization_score(
            pIC50=8.0, logP=2.0, mw=350.0, hbd=2, psa=90.0, n_heavy_atoms=25
        )
        s2 = potency_optimization_score(
            pIC50=8.0, logP=2.0, mw=350.0, hbd=2, psa=90.0, n_heavy_atoms=30
        )
        # Different n_heavy gives different LE, hence different scores
        assert s1 != s2

    def test_invalid_pic50(self):
        with pytest.raises(ValueError, match="pIC50 must be > 0"):
            potency_optimization_score(pIC50=0.0, logP=2.0, mw=350.0, hbd=2, psa=90.0)

    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw must be > 0"):
            potency_optimization_score(pIC50=7.0, logP=2.0, mw=0.0, hbd=2, psa=90.0)
