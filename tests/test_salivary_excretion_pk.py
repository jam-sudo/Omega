"""Tests for core/salivary_excretion_pk.py — Drug Salivary Excretion PK Model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.salivary_excretion_pk import (
    SalivaryExcretionResult,
    assess_salivary_tdm,
    simulate_salivary_pk,
)

VALID_RELIABILITY = {"excellent", "good", "acceptable", "unreliable"}


class TestSimulateSalivaryPkReturnsResult:
    def test_returns_dataclass(self):
        result = simulate_salivary_pk("Drug A", dose_mg=100.0)
        assert isinstance(result, SalivaryExcretionResult)

    def test_drug_name_preserved(self):
        result = simulate_salivary_pk("Warfarin", dose_mg=10.0)
        assert result.drug_name == "Warfarin"

    def test_dose_mg_preserved(self):
        result = simulate_salivary_pk("Drug B", dose_mg=50.0)
        assert math.isclose(result.dose_mg, 50.0)

    def test_lists_same_length(self):
        result = simulate_salivary_pk("Drug C", dose_mg=100.0, t_end_h=6.0, dt_h=0.5)
        assert len(result.times_h) == len(result.c_plasma_mg_L) == len(result.c_saliva_mg_L)

    def test_times_start_at_zero(self):
        result = simulate_salivary_pk("Drug D", dose_mg=100.0)
        assert math.isclose(result.times_h[0], 0.0)


class TestCmaxRelationship:
    def test_cmax_plasma_positive(self):
        result = simulate_salivary_pk("Drug E", dose_mg=100.0)
        assert result.cmax_plasma > 0.0

    def test_cmax_saliva_positive(self):
        result = simulate_salivary_pk("Drug F", dose_mg=100.0)
        assert result.cmax_saliva > 0.0

    def test_cmax_plasma_gt_cmax_saliva_when_sp_lt_1(self):
        # fup=0.5, neutral → sp_ratio=0.5 < 1
        result = simulate_salivary_pk("Drug G", dose_mg=100.0, fup=0.5, compound_type="neutral")
        if result.sp_ratio < 1.0:
            assert result.cmax_plasma > result.cmax_saliva

    def test_cmax_saliva_proportional_to_sp_ratio(self):
        r1 = simulate_salivary_pk("Drug H", dose_mg=100.0, fup=0.1)
        r2 = simulate_salivary_pk("Drug H", dose_mg=100.0, fup=0.8)
        assert r2.cmax_saliva > r1.cmax_saliva


class TestAUCRelationship:
    def test_auc_plasma_positive(self):
        result = simulate_salivary_pk("Drug I", dose_mg=100.0)
        assert result.auc_plasma > 0.0

    def test_auc_saliva_positive(self):
        result = simulate_salivary_pk("Drug J", dose_mg=100.0)
        assert result.auc_saliva > 0.0

    def test_auc_saliva_proportional_to_sp_ratio(self):
        result = simulate_salivary_pk("Drug K", dose_mg=100.0, fup=0.5, compound_type="neutral")
        # auc_saliva ≈ sp_ratio × auc_plasma
        expected = result.sp_ratio * result.auc_plasma
        assert math.isclose(result.auc_saliva, expected, rel_tol=1e-3)


class TestSpRatio:
    def test_sp_ratio_positive(self):
        result = simulate_salivary_pk("Drug L", dose_mg=100.0)
        assert result.sp_ratio > 0.0

    def test_neutral_sp_ratio_equals_fup(self):
        fup = 0.6
        result = simulate_salivary_pk("Drug M", dose_mg=100.0, fup=fup, compound_type="neutral")
        assert math.isclose(result.sp_ratio, fup, rel_tol=1e-9)

    def test_high_fup_higher_sp_ratio(self):
        r_low = simulate_salivary_pk("Drug N", dose_mg=100.0, fup=0.1, compound_type="neutral")
        r_high = simulate_salivary_pk("Drug N", dose_mg=100.0, fup=0.9, compound_type="neutral")
        assert r_high.sp_ratio > r_low.sp_ratio

    def test_basic_drug_high_pka_low_sp(self):
        # Basic drug pKa=9 at pH_saliva=6.8: heavily ionized → low f_neutral → low sp_ratio
        result = simulate_salivary_pk(
            "Basic Drug", dose_mg=100.0, fup=0.8, pka=9.0, compound_type="base", ph_saliva=6.8
        )
        # f_neutral = 1/(1+10^(9-6.8)) = 1/(1+10^2.2) ≈ 1/159 ≈ 0.0063
        assert result.sp_ratio < 0.1

    def test_acid_drug_below_pka_neutral(self):
        # Acid with pKa=9, pH_saliva=6.8: pH < pKa → mostly neutral → high f_neutral
        result = simulate_salivary_pk(
            "Acid Drug", dose_mg=100.0, fup=0.8, pka=9.0, compound_type="acid", ph_saliva=6.8
        )
        # f_neutral = 1/(1+10^(6.8-9)) = 1/(1+10^-2.2) ≈ 0.994
        assert result.sp_ratio > 0.7


class TestReliabilityLabel:
    def test_reliability_is_valid_string(self):
        result = simulate_salivary_pk("Drug O", dose_mg=100.0)
        assert result.reliability_for_tdm in VALID_RELIABILITY

    def test_excellent_when_high_sp(self):
        # fup close to 1, neutral → sp_ratio ≈ 1 → excellent
        result = simulate_salivary_pk("Drug P", dose_mg=100.0, fup=0.99, compound_type="neutral")
        assert result.reliability_for_tdm == "excellent"

    def test_unreliable_when_low_sp(self):
        # Basic drug, heavily ionized → sp_ratio < 0.1
        result = simulate_salivary_pk(
            "Drug Q", dose_mg=100.0, fup=0.05, pka=9.0, compound_type="base"
        )
        assert result.reliability_for_tdm == "unreliable"


class TestFNeutral:
    def test_neutral_f_neutral_is_one(self):
        result = simulate_salivary_pk("Drug R", dose_mg=100.0, compound_type="neutral")
        assert math.isclose(result.f_neutral_at_saliva_ph, 1.0, rel_tol=1e-9)

    def test_f_neutral_between_zero_and_one(self):
        result = simulate_salivary_pk(
            "Drug S", dose_mg=100.0, pka=7.4, compound_type="acid", ph_saliva=6.8
        )
        assert 0.0 < result.f_neutral_at_saliva_ph < 1.0


class TestNotesField:
    def test_notes_nonempty(self):
        result = simulate_salivary_pk("Drug T", dose_mg=100.0)
        assert len(result.notes) > 0


class TestValidation:
    def test_raises_on_zero_dose(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("Drug", dose_mg=0.0)

    def test_raises_on_negative_dose(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("Drug", dose_mg=-10.0)

    def test_raises_on_zero_cl(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("Drug", dose_mg=100.0, cl_L_per_h=0.0)

    def test_raises_on_zero_vd(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("Drug", dose_mg=100.0, vd_L=0.0)

    def test_raises_on_invalid_fup_zero(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("Drug", dose_mg=100.0, fup=0.0)

    def test_raises_on_invalid_ph(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("Drug", dose_mg=100.0, ph_saliva=15.0)


class TestAssessSalivaryTdm:
    def test_returns_dict(self):
        result = simulate_salivary_pk("Drug U", dose_mg=100.0)
        assessment = assess_salivary_tdm(result)
        assert isinstance(assessment, dict)

    def test_has_usable_key(self):
        result = simulate_salivary_pk("Drug V", dose_mg=100.0)
        assessment = assess_salivary_tdm(result)
        assert "usable" in assessment

    def test_has_sp_ratio_key(self):
        result = simulate_salivary_pk("Drug W", dose_mg=100.0)
        assessment = assess_salivary_tdm(result)
        assert "sp_ratio" in assessment

    def test_usable_true_when_sp_above_threshold(self):
        result = simulate_salivary_pk("Drug X", dose_mg=100.0, fup=0.5, compound_type="neutral")
        assessment = assess_salivary_tdm(result)
        assert assessment["usable"] is True

    def test_usable_false_when_sp_below_threshold(self):
        result = simulate_salivary_pk(
            "Drug Y", dose_mg=100.0, fup=0.05, pka=9.0, compound_type="base"
        )
        assessment = assess_salivary_tdm(result)
        assert assessment["usable"] is False

    def test_recommendation_nonempty(self):
        result = simulate_salivary_pk("Drug Z", dose_mg=100.0)
        assessment = assess_salivary_tdm(result)
        assert len(assessment["recommendation"]) > 0

    def test_plasma_preferred_when_unusable(self):
        result = simulate_salivary_pk(
            "Drug AA", dose_mg=100.0, fup=0.05, pka=9.0, compound_type="base"
        )
        assessment = assess_salivary_tdm(result)
        assert "plasma" in assessment["recommendation"].lower()
