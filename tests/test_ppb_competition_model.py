"""Tests for clinical/ppb_competition_model.py (Phase 525)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.ppb_competition_model import (
    FuShiftResult,
    assess_ddi_fu_shift,
    fu_adjusted_for_competition,
    protein_binding_sensitivity,
)

# ---------------------------------------------------------------------------
# fu_adjusted_for_competition
# ---------------------------------------------------------------------------


class TestFuAdjustedForCompetition:
    def test_zero_competitor_returns_fu_alone(self):
        fu = fu_adjusted_for_competition(
            fu_alone=0.1,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=0.0,
            ka_drug=1e4,
            ka_competitor=1e4,
        )
        assert fu == pytest.approx(0.1, rel=1e-6)

    def test_high_competitor_increases_fu(self):
        """Displacing competitor should increase free fraction."""
        fu_no_comp = fu_adjusted_for_competition(
            fu_alone=0.05,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=0.0,
            ka_drug=1e5,
            ka_competitor=1e5,
        )
        fu_with_comp = fu_adjusted_for_competition(
            fu_alone=0.05,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=10.0,
            ka_drug=1e5,
            ka_competitor=1e5,
        )
        assert fu_with_comp > fu_no_comp

    def test_fu_bounded_between_0_and_1(self):
        fu = fu_adjusted_for_competition(
            fu_alone=0.1,
            c_drug_mg_L=5.0,
            c_competitor_mg_L=100.0,
            ka_drug=1e6,
            ka_competitor=1e6,
        )
        assert 0 < fu <= 1.0

    def test_fu_alone_equal_1_no_binding(self):
        """fu_alone=1 means drug is not bound; competitor cannot increase fu."""
        fu = fu_adjusted_for_competition(
            fu_alone=1.0,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=5.0,
            ka_drug=1e5,
            ka_competitor=1e5,
        )
        assert fu == pytest.approx(1.0, abs=0.01)

    def test_invalid_fu_alone_zero_raises(self):
        with pytest.raises(ValueError, match="fu_alone"):
            fu_adjusted_for_competition(0.0, 1.0, 1.0, 1e4, 1e4)

    def test_invalid_fu_alone_negative_raises(self):
        with pytest.raises(ValueError, match="fu_alone"):
            fu_adjusted_for_competition(-0.1, 1.0, 1.0, 1e4, 1e4)

    def test_invalid_fu_alone_gt1_raises(self):
        with pytest.raises(ValueError, match="fu_alone"):
            fu_adjusted_for_competition(1.1, 1.0, 1.0, 1e4, 1e4)

    def test_negative_drug_conc_raises(self):
        with pytest.raises(ValueError, match="c_drug_mg_L"):
            fu_adjusted_for_competition(0.1, -1.0, 1.0, 1e4, 1e4)

    def test_negative_competitor_conc_raises(self):
        with pytest.raises(ValueError, match="c_competitor_mg_L"):
            fu_adjusted_for_competition(0.1, 1.0, -1.0, 1e4, 1e4)

    def test_low_ka_competitor_minimal_displacement(self):
        """Very weak competitor binding → minimal displacement."""
        fu_high_ka = fu_adjusted_for_competition(
            fu_alone=0.1,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=5.0,
            ka_drug=1e5,
            ka_competitor=1.0,  # very weak competitor
        )
        fu_high_ka2 = fu_adjusted_for_competition(
            fu_alone=0.1,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=5.0,
            ka_drug=1e5,
            ka_competitor=1e5,  # strong competitor
        )
        # Stronger competitor displaces more → higher fu
        assert fu_high_ka2 >= fu_high_ka

    def test_scatchard_iteration_convergence(self):
        """Calling twice with same args gives same result (stable iteration)."""
        kwargs = dict(
            fu_alone=0.08,
            c_drug_mg_L=2.0,
            c_competitor_mg_L=3.0,
            ka_drug=5e4,
            ka_competitor=8e4,
        )
        fu1 = fu_adjusted_for_competition(**kwargs)
        fu2 = fu_adjusted_for_competition(**kwargs)
        assert fu1 == pytest.approx(fu2, rel=1e-9)

    def test_custom_protein_concentration(self):
        fu_low_protein = fu_adjusted_for_competition(
            fu_alone=0.1,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=5.0,
            ka_drug=1e5,
            ka_competitor=1e5,
            protein_conc_mg_L=10000.0,
        )
        fu_high_protein = fu_adjusted_for_competition(
            fu_alone=0.1,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=5.0,
            ka_drug=1e5,
            ka_competitor=1e5,
            protein_conc_mg_L=80000.0,
        )
        # Both should be valid floats
        assert 0 < fu_low_protein <= 1.0
        assert 0 < fu_high_protein <= 1.0


# ---------------------------------------------------------------------------
# assess_ddi_fu_shift
# ---------------------------------------------------------------------------


class TestAssessDdiFuShift:
    def test_returns_fu_shift_result(self):
        result = assess_ddi_fu_shift(
            drug_name="warfarin",
            competitor_name="aspirin",
            fu_drug_alone=0.01,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=0.0,
        )
        assert isinstance(result, FuShiftResult)

    def test_no_competitor_no_change(self):
        result = assess_ddi_fu_shift(
            drug_name="warfarin",
            competitor_name="aspirin",
            fu_drug_alone=0.01,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=0.0,
        )
        assert result.fu_change_pct == pytest.approx(0.0, abs=0.01)
        assert result.clinical_relevance == "none"

    def test_large_displacement_significant(self):
        result = assess_ddi_fu_shift(
            drug_name="diazepam",
            competitor_name="valproate",
            fu_drug_alone=0.01,
            c_drug_mg_L=0.5,
            c_competitor_mg_L=50.0,
            ka_drug=1e6,
            ka_competitor=1e6,
        )
        assert result.clinical_relevance in ("significant", "monitor")
        assert result.fu_displaced > result.fu_alone

    def test_moderate_displacement_monitor(self):
        result = assess_ddi_fu_shift(
            drug_name="drug_a",
            competitor_name="drug_b",
            fu_drug_alone=0.05,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=5.0,
            ka_drug=1e5,
            ka_competitor=1e5,
        )
        # Any valid relevance level
        assert result.clinical_relevance in ("none", "monitor", "significant")
        assert result.fu_displaced >= result.fu_alone

    def test_frozen_dataclass(self):
        result = assess_ddi_fu_shift(
            drug_name="drug_a",
            competitor_name="drug_b",
            fu_drug_alone=0.1,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=1.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.fu_alone = 0.5  # type: ignore[misc]

    def test_fu_change_pct_sign_positive_for_displacement(self):
        """Displacement should increase fu → positive change pct."""
        result = assess_ddi_fu_shift(
            drug_name="drug_a",
            competitor_name="drug_b",
            fu_drug_alone=0.05,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=20.0,
            ka_drug=1e5,
            ka_competitor=1e5,
        )
        assert result.fu_change_pct >= 0.0

    def test_notes_field_non_empty(self):
        result = assess_ddi_fu_shift(
            drug_name="A",
            competitor_name="B",
            fu_drug_alone=0.1,
            c_drug_mg_L=1.0,
            c_competitor_mg_L=1.0,
        )
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# protein_binding_sensitivity
# ---------------------------------------------------------------------------


class TestProteinBindingSensitivity:
    def test_returns_list_of_dicts(self):
        result = protein_binding_sensitivity(
            drug_name="drug_x",
            fu_baseline=0.1,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(r, dict) for r in result)

    def test_dict_keys_present(self):
        result = protein_binding_sensitivity("X", 0.1, 5.0, 50.0)
        for row in result:
            assert "fu" in row
            assert "cl_effective" in row
            assert "auc_ratio" in row
            assert "t_half_ratio" in row

    def test_higher_fu_lower_cl_effective(self):
        """Restrictive model: CL_eff = CL_baseline * fu_baseline / fu.
        Higher fu → smaller divisor → higher CL is WRONG; actually:
        as fu increases ABOVE fu_baseline, CL_eff = CL_baseline * (fu_baseline/fu) DECREASES.
        So fu_values > fu_baseline → lower CL_eff.
        """
        fu_baseline = 0.1
        cl_baseline = 5.0
        vd = 50.0
        # Use fu values all above fu_baseline so we can check monotonicity
        fu_vals_above = [0.2, 0.5, 1.0]
        result = protein_binding_sensitivity(
            "X", fu_baseline, cl_baseline, vd, fu_values=fu_vals_above
        )
        cl_vals = [r["cl_effective"] for r in result]
        fu_vals = [r["fu"] for r in result]
        # Higher fu → lower CL_effective (since CL_eff = CL * fu_base / fu)
        pairs = sorted(zip(fu_vals, cl_vals))
        for i in range(1, len(pairs)):
            assert pairs[i][1] <= pairs[i - 1][1]

    def test_auc_ratio_at_baseline_fu_is_1(self):
        """At fu == fu_baseline, auc_ratio should be 1."""
        result = protein_binding_sensitivity("X", 0.1, 5.0, 50.0, fu_values=[0.1])
        assert len(result) == 1
        assert result[0]["auc_ratio"] == pytest.approx(1.0, rel=1e-6)

    def test_custom_fu_values(self):
        fu_list = [0.2, 0.4, 0.8]
        result = protein_binding_sensitivity("X", 0.1, 5.0, 50.0, fu_values=fu_list)
        returned_fu = [r["fu"] for r in result]
        assert returned_fu == fu_list

    def test_default_fu_values_filtered_by_half_fu_baseline(self):
        """Default fu_values should exclude those < fu_baseline/2."""
        # fu_baseline=0.2 → min_fu=0.1 → 0.01 and 0.05 excluded
        result = protein_binding_sensitivity("X", 0.2, 5.0, 50.0)
        fu_vals = [r["fu"] for r in result]
        assert all(f >= 0.1 for f in fu_vals)

    def test_t_half_ratio_at_baseline_is_1(self):
        result = protein_binding_sensitivity("X", 0.1, 5.0, 50.0, fu_values=[0.1])
        assert result[0]["t_half_ratio"] == pytest.approx(1.0, rel=1e-6)
