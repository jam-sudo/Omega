"""
Tests for Phase 510 — Drug Partitioning into Red Blood Cells
"""

import pytest

from omega_pbpk.prediction.rbc_partitioning_p510 import (
    RBCPartitioningResult,
    predict_rbc_partitioning,
    screen_rbc_partitioning,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

HCT = 0.45  # default hematocrit


def make_result(
    drug_name="TestDrug",
    logP=2.0,
    pKa=7.0,
    drug_type="neutral",
    fu_plasma=0.1,
    hematocrit=HCT,
):
    return predict_rbc_partitioning(
        drug_name=drug_name,
        logP=logP,
        pKa=pKa,
        drug_type=drug_type,
        fu_plasma=fu_plasma,
        hematocrit=hematocrit,
    )


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        result = make_result()
        assert isinstance(result, RBCPartitioningResult)

    def test_field_drug_name(self):
        result = make_result(drug_name="Amiodarone")
        assert result.drug_name == "Amiodarone"

    def test_field_logP(self):
        result = make_result(logP=3.0)
        assert result.logP == pytest.approx(3.0)

    def test_field_pKa(self):
        result = make_result(pKa=8.5)
        assert result.pKa == pytest.approx(8.5)

    def test_field_drug_type(self):
        result = make_result(drug_type="acid")
        assert result.drug_type == "acid"

    def test_field_fu_plasma(self):
        result = make_result(fu_plasma=0.3)
        assert result.fu_plasma == pytest.approx(0.3)

    def test_field_hematocrit(self):
        result = make_result(hematocrit=0.45)
        assert result.hematocrit == pytest.approx(0.45)

    def test_field_notes_is_str(self):
        result = make_result()
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Kp_rbc estimation
# ---------------------------------------------------------------------------


class TestKpRbc:
    def test_neutral_logP_zero_kp_rbc_is_1(self):
        # logP=0: excess = max(0, 0-1) = 0, Kp = 1 + 0.5*0 = 1.0
        result = make_result(logP=0.0, drug_type="neutral")
        assert result.kp_rbc == pytest.approx(1.0)

    def test_neutral_logP_1_kp_rbc_is_1(self):
        # logP=1: excess = 0, Kp = 1.0
        result = make_result(logP=1.0, drug_type="neutral")
        assert result.kp_rbc == pytest.approx(1.0)

    def test_neutral_logP_3(self):
        # logP=3: excess = 2, Kp = 1 + 0.5*2 = 2.0
        result = make_result(logP=3.0, drug_type="neutral")
        assert result.kp_rbc == pytest.approx(2.0)

    def test_acid_same_as_neutral(self):
        r_neutral = make_result(logP=3.0, drug_type="neutral")
        r_acid = make_result(logP=3.0, drug_type="acid")
        assert r_neutral.kp_rbc == pytest.approx(r_acid.kp_rbc)

    def test_base_higher_kp_rbc_than_neutral(self):
        # logP=4: neutral: 1+0.5*3=2.5; base: 1+1.2*3=4.6
        r_neutral = make_result(logP=4.0, drug_type="neutral", pKa=6.0)
        r_base = make_result(logP=4.0, drug_type="base", pKa=6.0)
        assert r_base.kp_rbc > r_neutral.kp_rbc

    def test_strong_base_extra_binding(self):
        # pKa > 8 → kp_rbc *= 1.5
        r_weak_base = make_result(logP=4.0, drug_type="base", pKa=7.0)
        r_strong_base = make_result(logP=4.0, drug_type="base", pKa=9.0)
        assert r_strong_base.kp_rbc == pytest.approx(r_weak_base.kp_rbc * 1.5, rel=1e-5)

    def test_kp_rbc_clamped_upper(self):
        # Very high logP should be clamped to 20.0
        result = make_result(logP=100.0, drug_type="base", pKa=10.0)
        assert result.kp_rbc <= 20.0

    def test_kp_rbc_clamped_lower(self):
        # Very low logP for any drug type: Kp >= 0.1
        result = make_result(logP=-50.0, drug_type="neutral")
        assert result.kp_rbc >= 0.1


# ---------------------------------------------------------------------------
# B:P ratio formula
# ---------------------------------------------------------------------------


class TestBPRatio:
    def test_bp_ratio_formula_neutral_logP0(self):
        # Kp_rbc=1 → B:P = 1 + 0.45*(1-1) = 1.0
        result = make_result(logP=0.0, drug_type="neutral")
        assert result.bp_ratio == pytest.approx(1.0)

    def test_bp_ratio_formula_explicit(self):
        # logP=3, neutral → Kp_rbc=2.0 → B:P = 1 + 0.45*(2-1) = 1.45
        result = make_result(logP=3.0, drug_type="neutral", hematocrit=0.45)
        expected_kp = 2.0
        expected_bp = 1.0 + 0.45 * (expected_kp - 1.0)
        assert result.bp_ratio == pytest.approx(expected_bp, rel=1e-5)

    def test_bp_ratio_clamped_upper(self):
        result = make_result(logP=100.0, drug_type="base", pKa=10.0)
        assert result.bp_ratio <= 10.0

    def test_bp_ratio_clamped_lower(self):
        result = make_result(logP=-100.0, drug_type="neutral")
        assert result.bp_ratio >= 0.1


# ---------------------------------------------------------------------------
# fu_blood
# ---------------------------------------------------------------------------


class TestFuBlood:
    def test_fu_blood_formula(self):
        result = make_result(logP=0.0, drug_type="neutral", fu_plasma=0.2)
        # Kp_rbc=1.0, B:P=1.0, fu_blood = 0.2 / 1.0 = 0.2
        assert result.fu_blood == pytest.approx(0.2, rel=1e-5)

    def test_fu_blood_less_when_bp_high(self):
        r_low_bp = make_result(logP=0.0, drug_type="neutral", fu_plasma=0.1)
        r_high_bp = make_result(logP=4.0, drug_type="base", pKa=9.0, fu_plasma=0.1)
        assert r_high_bp.fu_blood < r_low_bp.fu_blood


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


class TestFlags:
    def test_rbc_accumulation_true_when_kp_above_2(self):
        # logP=3, neutral → Kp_rbc=2.0 (not > 2.0, so False at exactly 2.0)
        result = make_result(logP=3.0, drug_type="neutral")
        # kp=2.0 is NOT > 2.0
        assert result.rbc_accumulation is False

    def test_rbc_accumulation_true_above_2(self):
        # logP=3, base, pKa<8 → Kp=1+1.2*2=3.4 > 2.0
        result = make_result(logP=3.0, drug_type="base", pKa=7.0)
        assert result.rbc_accumulation is True

    def test_plasma_depleting_true_when_bp_above_2(self):
        # logP=4, base, pKa=9 → Kp=1+1.2*3=4.6 *1.5=6.9, B:P=1+0.45*(6.9-1)=3.655
        result = make_result(logP=4.0, drug_type="base", pKa=9.0)
        assert result.plasma_depleting is True

    def test_plasma_depleting_false_when_bp_below_2(self):
        result = make_result(logP=0.0, drug_type="neutral")
        assert result.plasma_depleting is False


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            make_result(drug_type="zwitterion")

    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            make_result(fu_plasma=0.0)

    def test_fu_plasma_above_1_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            make_result(fu_plasma=1.1)

    def test_hematocrit_zero_raises(self):
        with pytest.raises(ValueError, match="hematocrit"):
            make_result(hematocrit=0.0)

    def test_hematocrit_one_raises(self):
        with pytest.raises(ValueError, match="hematocrit"):
            make_result(hematocrit=1.0)


# ---------------------------------------------------------------------------
# screen_rbc_partitioning
# ---------------------------------------------------------------------------


class TestScreenRBCPartitioning:
    def _make_compounds(self):
        return [
            {"drug_name": "LowBP", "logP": 0.0, "pKa": 7.0, "drug_type": "neutral"},
            {"drug_name": "HighBP", "logP": 5.0, "pKa": 9.0, "drug_type": "base"},
            {"drug_name": "MidBP", "logP": 3.0, "pKa": 6.0, "drug_type": "base"},
        ]

    def test_returns_list(self):
        results = screen_rbc_partitioning(self._make_compounds())
        assert isinstance(results, list)

    def test_sorted_by_bp_descending(self):
        results = screen_rbc_partitioning(self._make_compounds())
        bp_ratios = [r.bp_ratio for r in results]
        assert bp_ratios == sorted(bp_ratios, reverse=True)

    def test_correct_count(self):
        results = screen_rbc_partitioning(self._make_compounds())
        assert len(results) == 3

    def test_all_are_results(self):
        results = screen_rbc_partitioning(self._make_compounds())
        for r in results:
            assert isinstance(r, RBCPartitioningResult)

    def test_empty_list(self):
        assert screen_rbc_partitioning([]) == []
