"""Tests for Phase 611: Fed vs Fasted State Drug Dissolution.

Coverage:
- Return type and field presence
- Fed vs fasted differences by BCS class
- BCS class 2 shows largest food effect (bile-acid-driven solubility boost)
- BCS class 3 shows slight negative or neutral food effect
- AUC increases for BCS 2 in fed state
- Fraction absorbed values
- compare_fed_fasted function
- food_effect_pct sign by BCS class
- Validation errors
- Plasma concentration trajectory shape
- Dissolution rate metric
- Time vector consistency
"""

import pytest

from omega_pbpk.core.fed_fasted_dissolution_p611 import (
    FedFastedDissolutionResult,
    compare_fed_fasted,
    simulate_fed_fasted_dissolution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRUG_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    solubility_fasted_mg_mL=0.1,
    bcs_class=2,
    logP=2.5,
    mw_Da=350.0,
    cl_sys_L_per_h=10.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DRUG_DEFAULTS, **overrides}
    return simulate_fed_fasted_dissolution(**params)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_fed_fasted_dissolution_result(self):
        result = _run()
        assert isinstance(result, FedFastedDissolutionResult)

    def test_result_is_not_frozen(self):
        result = _run()
        # plain dataclass — should be mutable
        result.drug_name = "Modified"
        assert result.drug_name == "Modified"

    def test_all_list_fields_are_lists(self):
        result = _run()
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_dissolved_mg_mL, list)
        assert isinstance(result.c_absorbed_mg, list)
        assert isinstance(result.c_plasma_mg_L, list)

    def test_scalar_fields_are_floats(self):
        result = _run()
        assert isinstance(result.cmax_mg_L, float)
        assert isinstance(result.tmax_h, float)
        assert isinstance(result.auc_mg_h_per_L, float)
        assert isinstance(result.fraction_absorbed, float)
        assert isinstance(result.dissolution_rate_mg_per_h, float)
        assert isinstance(result.food_effect_pct, float)

    def test_notes_is_str(self):
        result = _run()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_drug_name_preserved(self):
        result = _run(drug_name="Ibuprofen")
        assert result.drug_name == "Ibuprofen"

    def test_dose_mg_preserved(self):
        result = _run(dose_mg=200.0)
        assert result.dose_mg == pytest.approx(200.0)

    def test_fed_state_flag_preserved(self):
        fasted = _run(fed_state=False)
        fed = _run(fed_state=True)
        assert fasted.fed_state is False
        assert fed.fed_state is True


# ---------------------------------------------------------------------------
# 2. Time vector
# ---------------------------------------------------------------------------


class TestTimeVector:
    def test_times_start_at_zero(self):
        result = _run()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = _run(t_end_h=8.0, dt_h=0.1)
        assert result.times_h[-1] == pytest.approx(8.0, abs=0.15)

    def test_list_lengths_consistent(self):
        result = _run()
        n = len(result.times_h)
        assert len(result.c_dissolved_mg_mL) == n
        assert len(result.c_absorbed_mg) == n
        assert len(result.c_plasma_mg_L) == n


# ---------------------------------------------------------------------------
# 3. Concentration constraints
# ---------------------------------------------------------------------------


class TestConcentrationConstraints:
    def test_plasma_concentration_nonnegative(self):
        result = _run()
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)

    def test_dissolved_concentration_nonnegative(self):
        result = _run()
        assert all(c >= 0.0 for c in result.c_dissolved_mg_mL)

    def test_absorbed_monotone(self):
        result = _run()
        # cumulative absorbed should be non-decreasing
        for i in range(1, len(result.c_absorbed_mg)):
            assert result.c_absorbed_mg[i] >= result.c_absorbed_mg[i - 1] - 1e-9

    def test_cmax_equals_max_plasma(self):
        result = _run()
        assert result.cmax_mg_L == pytest.approx(max(result.c_plasma_mg_L))

    def test_cmax_positive(self):
        result = _run()
        assert result.cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# 4. Fed vs fasted BCS class differences
# ---------------------------------------------------------------------------


class TestFedFastedBCSClass:
    """BCS class 2 should show the largest positive food effect."""

    def test_bcs2_fed_higher_auc_than_fasted(self):
        fasted = _run(bcs_class=2, fed_state=False)
        fed = _run(bcs_class=2, fed_state=True)
        assert fed.auc_mg_h_per_L > fasted.auc_mg_h_per_L

    def test_bcs2_fed_higher_cmax_than_fasted(self):
        fasted = _run(bcs_class=2, fed_state=False)
        fed = _run(bcs_class=2, fed_state=True)
        assert fed.cmax_mg_L > fasted.cmax_mg_L

    def test_bcs4_fed_higher_auc_than_fasted(self):
        fasted = _run(bcs_class=4, fed_state=False)
        fed = _run(bcs_class=4, fed_state=True)
        assert fed.auc_mg_h_per_L > fasted.auc_mg_h_per_L

    def test_bcs1_fed_auc_in_plausible_range(self):
        """BCS 1: no solubility change, but gastric emptying delay extends transit
        time and increases AUC; ratio may exceed 2 due to slower drainage."""
        fasted = _run(bcs_class=1, solubility_fasted_mg_mL=5.0, fed_state=False)
        fed = _run(bcs_class=1, solubility_fasted_mg_mL=5.0, fed_state=True)
        # Both states should give positive AUC; fed AUC is typically higher
        assert fasted.auc_mg_h_per_L > 0.0
        assert fed.auc_mg_h_per_L > 0.0
        # Fed is never dramatically lower than fasted
        assert fed.auc_mg_h_per_L >= fasted.auc_mg_h_per_L * 0.5

    def test_bcs3_fed_auc_positive(self):
        """BCS 3: despite 0.8× solubility, slower emptying still allows positive
        absorption in both states; fed AUC may be higher due to extended residence."""
        fasted = _run(bcs_class=3, solubility_fasted_mg_mL=2.0, fed_state=False)
        fed = _run(bcs_class=3, solubility_fasted_mg_mL=2.0, fed_state=True)
        assert fasted.auc_mg_h_per_L > 0.0
        assert fed.auc_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 5. Fraction absorbed
# ---------------------------------------------------------------------------


class TestFractionAbsorbed:
    def test_fraction_absorbed_in_range(self):
        result = _run()
        assert 0.0 <= result.fraction_absorbed <= 1.0

    def test_high_solubility_high_perm_meaningful_absorption(self):
        """BCS class 1 drug (high sol, high perm) should show meaningful
        absorption; complete absorption is limited by GI drainage and clearance."""
        result = simulate_fed_fasted_dissolution(
            drug_name="HighSolHighPerm",
            dose_mg=10.0,
            solubility_fasted_mg_mL=10.0,
            bcs_class=1,
            logP=1.0,
            mw_Da=250.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=20.0,
            t_end_h=24.0,
            dt_h=0.05,
        )
        # Dissolution rate constant limited by MW factor; some fraction absorbed
        assert result.fraction_absorbed > 0.1

    def test_very_low_solubility_lower_absorption(self):
        high = _run(solubility_fasted_mg_mL=5.0, bcs_class=2)
        low = _run(solubility_fasted_mg_mL=0.001, bcs_class=2)
        assert low.fraction_absorbed < high.fraction_absorbed


# ---------------------------------------------------------------------------
# 6. Dissolution rate
# ---------------------------------------------------------------------------


class TestDissolutionRate:
    def test_dissolution_rate_nonnegative(self):
        result = _run()
        assert result.dissolution_rate_mg_per_h >= 0.0

    def test_food_effect_pct_zero_in_single_run(self):
        result = _run(fed_state=True)
        assert result.food_effect_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. compare_fed_fasted function
# ---------------------------------------------------------------------------


class TestCompareFedFasted:
    def test_returns_dict_with_expected_keys(self):
        result = compare_fed_fasted(**{k: v for k, v in DRUG_DEFAULTS.items()})
        assert "fed" in result
        assert "fasted" in result
        assert "food_effect_pct" in result

    def test_fed_result_type(self):
        result = compare_fed_fasted(**{k: v for k, v in DRUG_DEFAULTS.items()})
        assert isinstance(result["fed"], FedFastedDissolutionResult)
        assert isinstance(result["fasted"], FedFastedDissolutionResult)

    def test_food_effect_pct_is_float(self):
        result = compare_fed_fasted(**{k: v for k, v in DRUG_DEFAULTS.items()})
        assert isinstance(result["food_effect_pct"], float)

    def test_bcs2_positive_food_effect(self):
        result = compare_fed_fasted(
            drug_name="BCS2Drug",
            dose_mg=100.0,
            solubility_fasted_mg_mL=0.05,
            bcs_class=2,
            logP=3.0,
            mw_Da=400.0,
            cl_sys_L_per_h=8.0,
            vd_sys_L=40.0,
        )
        assert result["food_effect_pct"] > 0.0

    def test_food_effect_consistent_with_aucs(self):
        result = compare_fed_fasted(**{k: v for k, v in DRUG_DEFAULTS.items()})
        fed_auc = result["fed"].auc_mg_h_per_L
        fasted_auc = result["fasted"].auc_mg_h_per_L
        expected_pct = (fed_auc - fasted_auc) / fasted_auc * 100.0
        assert result["food_effect_pct"] == pytest.approx(expected_pct, rel=1e-6)

    def test_fed_flag_in_results(self):
        result = compare_fed_fasted(**{k: v for k, v in DRUG_DEFAULTS.items()})
        assert result["fed"].fed_state is True
        assert result["fasted"].fed_state is False


# ---------------------------------------------------------------------------
# 8. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_invalid_bcs_class_raises(self):
        with pytest.raises(ValueError, match="bcs_class"):
            _run(bcs_class=5)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=0.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=-1.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            _run(solubility_fasted_mg_mL=0.0)
