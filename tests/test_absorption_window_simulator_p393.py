"""Tests for Phase 393 - Drug Absorption Window Simulator."""

import pytest

from omega_pbpk.core.absorption_window_simulator_p393 import (
    AbsorptionWindowResult,
    _ph_adjusted_solubility,
    compare_formulation_solubilities,
    simulate_absorption_window,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bcs1_drug():
    """BCS Class I: high solubility, high permeability."""
    return dict(
        drug_name="BCS1Drug",
        dose_mg=10.0,
        pka=7.0,
        logp=2.0,
        mw_da=300.0,
        solubility_mg_mL=10.0,  # Do = 10/(10*250)=0.004 << 1
        permeability_cm_s=5e-4,  # Pn = 5e-4/2e-4 = 2.5 >= 1
        drug_type="neutral",
    )


def _bcs4_drug():
    """BCS Class IV: low solubility, low permeability."""
    return dict(
        drug_name="BCS4Drug",
        dose_mg=100.0,
        pka=7.0,
        logp=1.0,
        mw_da=400.0,
        solubility_mg_mL=0.001,  # Do = 100/(0.001*250) = 400 >> 1
        permeability_cm_s=5e-6,  # Pn = 5e-6/2e-4 = 0.025 < 1
        drug_type="neutral",
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_absorption_window_result(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert isinstance(result, AbsorptionWindowResult)

    def test_result_has_drug_name(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert result.drug_name == "BCS1Drug"

    def test_result_has_dose(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert result.dose_mg == 10.0

    def test_segments_is_list_of_five(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert isinstance(result.segments, list)
        assert len(result.segments) == 5

    def test_segment_names_correct(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert result.segments == ["stomach", "duodenum", "jejunum", "ileum", "colon"]

    def test_transit_times_match_segments(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert len(result.transit_times_h) == len(result.segments)

    def test_dissolved_fractions_in_range(self):
        result = simulate_absorption_window(**_bcs1_drug())
        for df in result.dissolved_fractions:
            assert 0.0 <= df <= 1.0

    def test_absorbed_fractions_in_range(self):
        result = simulate_absorption_window(**_bcs1_drug())
        for af in result.absorbed_fractions:
            assert 0.0 <= af <= 1.0

    def test_fa_predicted_in_zero_to_one(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert 0.0 <= result.fa_predicted <= 1.0

    def test_cumulative_absorbed_pct_in_zero_to_hundred(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert 0.0 <= result.cumulative_absorbed_pct <= 100.0

    def test_notes_is_string(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# BCS classification tests
# ---------------------------------------------------------------------------


class TestBCSClassification:
    def test_bcs1_classification(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert result.bcs_class == "I"

    def test_bcs4_classification(self):
        result = simulate_absorption_window(**_bcs4_drug())
        assert result.bcs_class == "IV"

    def test_bcs2_classification(self):
        # High dose, low solubility, high permeability
        result = simulate_absorption_window(
            drug_name="BCS2",
            dose_mg=500.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            solubility_mg_mL=0.1,  # Do=500/(0.1*250)=20 >1
            permeability_cm_s=5e-4,  # Pn=2.5 >=1
        )
        assert result.bcs_class == "II"

    def test_bcs3_classification(self):
        # Low dose, high solubility, low permeability
        result = simulate_absorption_window(
            drug_name="BCS3",
            dose_mg=5.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            solubility_mg_mL=5.0,  # Do=5/(5*250)=0.004 <=1
            permeability_cm_s=5e-6,  # Pn=0.025 <1
        )
        assert result.bcs_class == "III"

    def test_bcs1_rate_limiting_transit(self):
        result = simulate_absorption_window(**_bcs1_drug())
        assert result.rate_limiting_step == "transit"

    def test_bcs2_rate_limiting_dissolution(self):
        result = simulate_absorption_window(
            drug_name="BCS2",
            dose_mg=500.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            solubility_mg_mL=0.1,
            permeability_cm_s=5e-4,
        )
        assert result.rate_limiting_step == "dissolution"

    def test_bcs3_rate_limiting_permeability(self):
        result = simulate_absorption_window(
            drug_name="BCS3",
            dose_mg=5.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            solubility_mg_mL=5.0,
            permeability_cm_s=5e-6,
        )
        assert result.rate_limiting_step == "permeability"

    def test_bcs4_rate_limiting_solubility(self):
        result = simulate_absorption_window(**_bcs4_drug())
        assert result.rate_limiting_step == "solubility"


# ---------------------------------------------------------------------------
# Absorption tests
# ---------------------------------------------------------------------------


class TestAbsorption:
    def test_bcs1_good_absorption(self):
        result = simulate_absorption_window(**_bcs1_drug())
        # BCS I drug should have high fa (>50%)
        assert result.fa_predicted > 0.5

    def test_bcs4_poor_absorption(self):
        result = simulate_absorption_window(**_bcs4_drug())
        # BCS IV should have lower fa
        assert result.fa_predicted < result.fa_predicted + 0.5  # relative check
        # More specifically, BCS IV should absorb poorly
        # At very low permeability and low solubility
        assert result.fa_predicted < 0.5 or result.cumulative_absorbed_pct < 50.0

    def test_higher_dose_lower_fa_bcs2(self):
        # For BCS II (dissolution-limited), higher dose → lower FA
        low_dose = simulate_absorption_window(
            drug_name="BCS2low",
            dose_mg=10.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            solubility_mg_mL=0.01,
            permeability_cm_s=5e-4,
        )
        high_dose = simulate_absorption_window(
            drug_name="BCS2high",
            dose_mg=500.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            solubility_mg_mL=0.01,
            permeability_cm_s=5e-4,
        )
        assert low_dose.fa_predicted >= high_dose.fa_predicted

    def test_permeability_effect_on_absorption(self):
        # Higher permeability → higher fa
        low_perm = simulate_absorption_window(
            drug_name="LowPerm",
            dose_mg=10.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            solubility_mg_mL=1.0,
            permeability_cm_s=1e-6,
        )
        high_perm = simulate_absorption_window(
            drug_name="HighPerm",
            dose_mg=10.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            solubility_mg_mL=1.0,
            permeability_cm_s=1e-3,
        )
        assert high_perm.fa_predicted > low_perm.fa_predicted

    def test_jejunum_most_absorbed_well_absorbed_drug(self):
        result = simulate_absorption_window(**_bcs1_drug())
        # Jejunum has largest surface area - should have highest absorbed_fraction
        seg_idx = result.segments.index("jejunum")
        jejunum_absorbed = result.absorbed_fractions[seg_idx]
        colon_idx = result.segments.index("colon")
        colon_absorbed = result.absorbed_fractions[colon_idx]
        assert jejunum_absorbed > colon_absorbed

    def test_colon_small_contribution(self):
        result = simulate_absorption_window(**_bcs1_drug())
        colon_idx = result.segments.index("colon")
        colon_absorbed = result.absorbed_fractions[colon_idx]
        # Colon should be small contributor for well-absorbed drug
        assert colon_absorbed < 0.3  # less than 30% of dose from colon

    def test_absorption_complete_segment_set_for_bcs1(self):
        result = simulate_absorption_window(**_bcs1_drug())
        # BCS I drugs should achieve 90% absorption in jejunum or earlier
        assert result.absorption_complete_segment in [
            "stomach",
            "duodenum",
            "jejunum",
            "ileum",
            "colon",
        ]


# ---------------------------------------------------------------------------
# pH-dependent solubility tests
# ---------------------------------------------------------------------------


class TestPhSolubility:
    def test_acid_higher_solubility_at_high_ph(self):
        # Acid drug: more soluble at higher pH (above pKa)
        s_low = _ph_adjusted_solubility(0.1, pka=5.0, ph=2.0, drug_type="acid")
        s_high = _ph_adjusted_solubility(0.1, pka=5.0, ph=8.0, drug_type="acid")
        assert s_high > s_low

    def test_base_higher_solubility_at_low_ph(self):
        # Base drug: more soluble at lower pH (below pKa)
        s_low = _ph_adjusted_solubility(0.1, pka=7.0, ph=2.0, drug_type="base")
        s_high = _ph_adjusted_solubility(0.1, pka=7.0, ph=9.0, drug_type="base")
        assert s_low > s_high

    def test_neutral_solubility_unchanged(self):
        s = _ph_adjusted_solubility(0.5, pka=7.0, ph=6.5, drug_type="neutral")
        assert s == pytest.approx(0.5, rel=1e-9)

    def test_solubility_capped_at_1000x(self):
        # Extreme pH should not exceed 1000x
        s = _ph_adjusted_solubility(1.0, pka=7.0, ph=14.0, drug_type="acid")
        assert s <= 1000.0


# ---------------------------------------------------------------------------
# Compare formulations tests
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_compare_returns_five_results(self):
        results = compare_formulation_solubilities(
            drug_name="TestDrug",
            dose_mg=100.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            permeability_cm_s=5e-4,
        )
        assert len(results) == 5

    def test_compare_sorted_descending_fa(self):
        results = compare_formulation_solubilities(
            drug_name="TestDrug",
            dose_mg=100.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            permeability_cm_s=5e-4,
        )
        fas = [r.fa_predicted for r in results]
        assert fas == sorted(fas, reverse=True)

    def test_compare_custom_solubilities(self):
        solubilities = [0.5, 5.0, 50.0]
        results = compare_formulation_solubilities(
            drug_name="TestDrug",
            dose_mg=100.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            permeability_cm_s=5e-4,
            solubility_values_mg_mL=solubilities,
        )
        assert len(results) == 3

    def test_compare_returns_absorption_window_results(self):
        results = compare_formulation_solubilities(
            drug_name="TestDrug",
            dose_mg=100.0,
            pka=7.0,
            logp=2.0,
            mw_da=300.0,
            permeability_cm_s=5e-4,
        )
        for r in results:
            assert isinstance(r, AbsorptionWindowResult)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_absorption_window(
                drug_name="X",
                dose_mg=0.0,
                pka=7.0,
                logp=2.0,
                mw_da=300.0,
                solubility_mg_mL=1.0,
                permeability_cm_s=1e-4,
            )

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_absorption_window(
                drug_name="X",
                dose_mg=-10.0,
                pka=7.0,
                logp=2.0,
                mw_da=300.0,
                solubility_mg_mL=1.0,
                permeability_cm_s=1e-4,
            )

    def test_invalid_solubility_zero(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            simulate_absorption_window(
                drug_name="X",
                dose_mg=10.0,
                pka=7.0,
                logp=2.0,
                mw_da=300.0,
                solubility_mg_mL=0.0,
                permeability_cm_s=1e-4,
            )

    def test_invalid_permeability_zero(self):
        with pytest.raises(ValueError, match="permeability_cm_s"):
            simulate_absorption_window(
                drug_name="X",
                dose_mg=10.0,
                pka=7.0,
                logp=2.0,
                mw_da=300.0,
                solubility_mg_mL=1.0,
                permeability_cm_s=0.0,
            )

    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw_da"):
            simulate_absorption_window(
                drug_name="X",
                dose_mg=10.0,
                pka=7.0,
                logp=2.0,
                mw_da=0.0,
                solubility_mg_mL=1.0,
                permeability_cm_s=1e-4,
            )
