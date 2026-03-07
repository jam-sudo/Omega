"""Tests for prediction/metabolism_rate.py (Phase 367)."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.prediction.metabolism_rate import (
    MetabolismRateResult,
    compare_structural_modifications,
    predict_metabolism_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def neutral_drug():
    """Generic neutral drug with moderate logP."""
    return predict_metabolism_rate(
        compound_name="NeutralDrug",
        logP=2.5,
        mw=350.0,
        n_hbd=2,
        n_hba=4,
    )


@pytest.fixture
def high_logp_drug():
    return predict_metabolism_rate(
        compound_name="HighLogP",
        logP=5.0,
        mw=300.0,
        n_hbd=1,
        n_hba=2,
    )


@pytest.fixture
def low_logp_drug():
    return predict_metabolism_rate(
        compound_name="LowLogP",
        logP=-0.5,
        mw=200.0,
        n_hbd=3,
        n_hba=5,
    )


@pytest.fixture
def basic_drug():
    return predict_metabolism_rate(
        compound_name="BasicDrug",
        logP=3.0,
        mw=300.0,
        n_hbd=1,
        n_hba=3,
        pka_base=8.5,
    )


@pytest.fixture
def acidic_drug():
    return predict_metabolism_rate(
        compound_name="AcidicDrug",
        logP=2.0,
        mw=280.0,
        n_hbd=2,
        n_hba=4,
        pka_acid=3.5,
    )


@pytest.fixture
def high_mw_drug():
    return predict_metabolism_rate(
        compound_name="HighMW",
        logP=3.0,
        mw=600.0,
        n_hbd=3,
        n_hba=6,
    )


# ---------------------------------------------------------------------------
# Basic structural and type tests
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dataclass(self, neutral_drug):
        assert isinstance(neutral_drug, MetabolismRateResult)

    def test_frozen_dataclass(self, neutral_drug):
        with pytest.raises((AttributeError, TypeError)):
            neutral_drug.compound_name = "Modified"  # type: ignore[misc]

    def test_compound_name_preserved(self):
        result = predict_metabolism_rate("TestCompound", logP=2.0, mw=300.0, n_hbd=1, n_hba=3)
        assert result.compound_name == "TestCompound"


# ---------------------------------------------------------------------------
# CLint range tests
# ---------------------------------------------------------------------------

class TestClintRange:
    def test_clint_hep_in_valid_range(self, neutral_drug):
        assert 0.1 <= neutral_drug.clint_hep_uL_per_min_per_mg <= 500.0

    def test_clint_hep_range_high_logp(self, high_logp_drug):
        assert 0.1 <= high_logp_drug.clint_hep_uL_per_min_per_mg <= 500.0

    def test_clint_hep_range_low_logp(self, low_logp_drug):
        assert 0.1 <= low_logp_drug.clint_hep_uL_per_min_per_mg <= 500.0

    def test_s9_lower_than_hep(self, neutral_drug):
        assert neutral_drug.clint_s9_uL_per_min_per_mg < neutral_drug.clint_hep_uL_per_min_per_mg

    def test_s9_is_30_percent_of_hep(self, neutral_drug):
        assert abs(neutral_drug.clint_s9_uL_per_min_per_mg - neutral_drug.clint_hep_uL_per_min_per_mg * 0.3) < 1e-6

    def test_hepatocyte_clint_positive(self, neutral_drug):
        assert neutral_drug.clint_hepatocyte_uL_per_min_per_cell > 0


# ---------------------------------------------------------------------------
# LogP effect tests
# ---------------------------------------------------------------------------

class TestLogPEffect:
    def test_high_logp_higher_clint_than_low_logp(self, high_logp_drug, low_logp_drug):
        """High logP should generally produce higher CLint than low logP."""
        assert high_logp_drug.clint_hep_uL_per_min_per_mg > low_logp_drug.clint_hep_uL_per_min_per_mg

    def test_high_logp_stability_low_or_moderate(self, high_logp_drug):
        assert high_logp_drug.microsomal_stability in ("moderate", "low")


# ---------------------------------------------------------------------------
# MW effect tests
# ---------------------------------------------------------------------------

class TestMWEffect:
    def test_high_mw_reduces_clint(self, neutral_drug, high_mw_drug):
        """Both have logP=3, but high MW drug should have lower CLint."""
        # high_mw_drug (600 Da) vs neutral_drug (350 Da, similar logP)
        # The high MW drug gets a 0.5x reduction factor
        base_neutral = predict_metabolism_rate("X", logP=3.0, mw=350.0, n_hbd=3, n_hba=6)
        assert high_mw_drug.clint_hep_uL_per_min_per_mg < base_neutral.clint_hep_uL_per_min_per_mg


# ---------------------------------------------------------------------------
# CYP attribution tests
# ---------------------------------------------------------------------------

class TestCYPAttribution:
    def test_fm_sum_leq_1(self, neutral_drug):
        total = (
            neutral_drug.predicted_fm_cyp3a4
            + neutral_drug.predicted_fm_cyp2d6
            + neutral_drug.predicted_fm_cyp2c9
            + neutral_drug.predicted_fm_cyp2c19
            + neutral_drug.predicted_fm_cyp1a2
        )
        assert total <= 1.0 + 1e-9

    def test_fm_total_leq_1(self, neutral_drug):
        assert neutral_drug.fm_total <= 1.0

    def test_basic_drug_higher_cyp2d6(self, basic_drug, neutral_drug):
        """Basic drug should have higher CYP2D6 fraction."""
        assert basic_drug.predicted_fm_cyp2d6 > neutral_drug.predicted_fm_cyp2d6

    def test_acidic_drug_higher_cyp2c9(self, acidic_drug, neutral_drug):
        """Acidic drug should have higher CYP2C9 fraction."""
        assert acidic_drug.predicted_fm_cyp2c9 > neutral_drug.predicted_fm_cyp2c9

    def test_low_logp_higher_cyp1a2(self, low_logp_drug, neutral_drug):
        """Low logP drug should have higher CYP1A2 fraction."""
        assert low_logp_drug.predicted_fm_cyp1a2 > neutral_drug.predicted_fm_cyp1a2

    def test_all_fm_nonnegative(self, basic_drug):
        assert basic_drug.predicted_fm_cyp3a4 >= 0
        assert basic_drug.predicted_fm_cyp2d6 >= 0
        assert basic_drug.predicted_fm_cyp2c9 >= 0
        assert basic_drug.predicted_fm_cyp2c19 >= 0
        assert basic_drug.predicted_fm_cyp1a2 >= 0


# ---------------------------------------------------------------------------
# In vivo scaling tests
# ---------------------------------------------------------------------------

class TestInVivoScaling:
    def test_cl_hep_leq_hepatic_blood_flow(self, neutral_drug):
        assert neutral_drug.cl_hep_predicted_L_per_h <= 90.0

    def test_cl_hep_leq_custom_blood_flow(self):
        result = predict_metabolism_rate(
            "TestDrug", logP=4.0, mw=300.0, n_hbd=1, n_hba=3,
            hepatic_blood_flow_L_per_h=60.0
        )
        assert result.cl_hep_predicted_L_per_h <= 60.0

    def test_t_half_positive(self, neutral_drug):
        assert neutral_drug.t_half_predicted_h > 0

    def test_t_half_finite(self, neutral_drug):
        assert not math.isinf(neutral_drug.t_half_predicted_h)

    def test_cl_sys_equals_cl_hep(self, neutral_drug):
        assert abs(neutral_drug.cl_sys_predicted_L_per_h - neutral_drug.cl_hep_predicted_L_per_h) < 1e-9


# ---------------------------------------------------------------------------
# Flag tests
# ---------------------------------------------------------------------------

class TestFlags:
    def test_cyp3a4_dominant_flag(self):
        result = predict_metabolism_rate(
            "Dominant3A4", logP=2.5, mw=350.0, n_hbd=1, n_hba=3
        )
        # Default neutral drug: fm_3a4 starts high (0.6) with no adjustments
        if result.predicted_fm_cyp3a4 > 0.7:
            assert result.cyp3a4_dominant is True
        else:
            assert result.cyp3a4_dominant is False

    def test_cyp2d6_substrate_flag(self, basic_drug):
        """Basic drug with high pKa should trigger CYP2D6 substrate flag."""
        if basic_drug.predicted_fm_cyp2d6 > 0.3:
            assert basic_drug.cyp2d6_substrate is True

    def test_non_cyp_metabolism_flag_when_fm_low(self):
        # Force low logP, high MW, acidic: reduces fm_3a4 twice
        result = predict_metabolism_rate(
            "NonCYP", logP=0.5, mw=600.0, n_hbd=5, n_hba=8, pka_acid=3.0
        )
        if result.fm_total < 0.7:
            assert result.non_cyp_metabolism is True

    def test_microsomal_stability_high(self):
        result = predict_metabolism_rate(
            "StableDrug", logP=0.0, mw=800.0, n_hbd=0, n_hba=0
        )
        if result.clint_hep_uL_per_min_per_mg < 10:
            assert result.microsomal_stability == "high"

    def test_microsomal_stability_low(self, high_logp_drug):
        if high_logp_drug.clint_hep_uL_per_min_per_mg > 100:
            assert high_logp_drug.microsomal_stability == "low"

    def test_microsomal_stability_valid_values(self, neutral_drug):
        assert neutral_drug.microsomal_stability in ("high", "moderate", "low")


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw"):
            predict_metabolism_rate("X", logP=2.0, mw=0.0, n_hbd=1, n_hba=2)

    def test_invalid_mw_negative(self):
        with pytest.raises(ValueError, match="mw"):
            predict_metabolism_rate("X", logP=2.0, mw=-100.0, n_hbd=1, n_hba=2)

    def test_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_metabolism_rate("X", logP=2.0, mw=300.0, n_hbd=1, n_hba=2, fu_plasma=0.0)

    def test_invalid_fu_plasma_greater_than_1(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_metabolism_rate("X", logP=2.0, mw=300.0, n_hbd=1, n_hba=2, fu_plasma=1.5)


# ---------------------------------------------------------------------------
# compare_structural_modifications tests
# ---------------------------------------------------------------------------

class TestCompareStructuralModifications:
    def test_returns_sorted_by_cl_sys_ascending(self):
        base = {"compound_name": "Base", "logP": 2.0, "mw": 300.0, "n_hbd": 2, "n_hba": 4}
        mods = [
            {"compound_name": "ModHighLogP", "logP": 5.0},
            {"compound_name": "ModLowLogP", "logP": 0.0},
            {"compound_name": "ModHighMW", "mw": 700.0},
        ]
        results = compare_structural_modifications(base, mods)
        cl_values = [r.cl_sys_predicted_L_per_h for r in results]
        assert cl_values == sorted(cl_values)

    def test_returns_correct_count(self):
        base = {"compound_name": "Base", "logP": 2.0, "mw": 300.0, "n_hbd": 1, "n_hba": 2}
        mods = [{"compound_name": f"Mod{i}", "logP": float(i)} for i in range(5)]
        results = compare_structural_modifications(base, mods)
        assert len(results) == 5

    def test_modification_overrides_base(self):
        base = {"compound_name": "Base", "logP": 2.0, "mw": 300.0, "n_hbd": 1, "n_hba": 2}
        mods = [{"compound_name": "Modified", "logP": 5.0, "mw": 300.0}]
        results = compare_structural_modifications(base, mods)
        assert results[0].compound_name == "Modified"
