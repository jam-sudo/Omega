"""Tests for Phase 529 — Urinary pH Manipulation and Drug Excretion."""

import pytest

from omega_pbpk.clinical.urinary_ph_manipulation import (
    UrinaryPHResult,
    compare_urine_ph,
    predict_urinary_excretion,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_acid(urine_pH=6.0, pKa=3.5, **kw):
    return predict_urinary_excretion("aspirin", pKa, "acid", urine_pH, **kw)


def make_base(urine_pH=6.0, pKa=9.5, **kw):
    return predict_urinary_excretion("amphetamine", pKa, "base", urine_pH, **kw)


def make_neutral(urine_pH=6.0, pKa=7.0, **kw):
    return predict_urinary_excretion("ethanol", pKa, "neutral", urine_pH, **kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_urinary_ph_result(self):
        result = make_acid()
        assert isinstance(result, UrinaryPHResult)

    def test_frozen_dataclass(self):
        result = make_acid()
        with pytest.raises((AttributeError, TypeError)):
            result.urine_pH = 7.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Field values
# ---------------------------------------------------------------------------


class TestFields:
    def test_drug_name(self):
        r = predict_urinary_excretion("warfarin", 5.0, "acid", 7.4)
        assert r.drug_name == "warfarin"

    def test_pKa_stored(self):
        r = make_acid(pKa=4.2)
        assert r.pKa == pytest.approx(4.2)

    def test_drug_type_stored(self):
        assert make_acid().drug_type == "acid"
        assert make_base().drug_type == "base"
        assert make_neutral().drug_type == "neutral"

    def test_urine_pH_stored(self):
        r = make_acid(urine_pH=7.5)
        assert r.urine_pH == pytest.approx(7.5)

    def test_ionized_fraction_between_0_and_1(self):
        for ph in [4.0, 6.0, 8.0]:
            r = make_acid(urine_pH=ph)
            assert 0.0 <= r.ionized_fraction <= 1.0

    def test_f_reabs_between_0_and_095(self):
        for ph in [4.0, 6.0, 8.0]:
            r = make_acid(urine_pH=ph)
            assert 0.0 <= r.f_reabs <= 0.95

    def test_fe_renal_between_0_and_1(self):
        r = make_acid()
        assert 0.0 <= r.fe_renal <= 1.0

    def test_t_half_positive(self):
        r = make_acid()
        assert r.t_half_h > 0


# ---------------------------------------------------------------------------
# Acid behaviour
# ---------------------------------------------------------------------------


class TestAcidBehaviour:
    def test_higher_ph_more_ionized_acid(self):
        """Acid at high pH has higher ionized fraction."""
        low = make_acid(urine_pH=5.0, pKa=4.0)
        high = make_acid(urine_pH=8.0, pKa=4.0)
        assert high.ionized_fraction > low.ionized_fraction

    def test_higher_ph_higher_cl_renal_acid(self):
        """Alkaline urine enhances acid excretion (more ionized, less reabsorbed)."""
        low = make_acid(urine_pH=5.0, pKa=4.0)
        high = make_acid(urine_pH=8.0, pKa=4.0)
        assert high.cl_renal_L_per_h > low.cl_renal_L_per_h

    def test_alkalization_benefit_true_for_acid(self):
        r = make_acid()
        assert r.alkalization_benefit is True

    def test_alkalization_benefit_false_for_base(self):
        r = make_base()
        assert r.alkalization_benefit is False

    def test_alkalization_benefit_false_for_neutral(self):
        r = make_neutral()
        assert r.alkalization_benefit is False


# ---------------------------------------------------------------------------
# Base behaviour
# ---------------------------------------------------------------------------


class TestBaseBehaviour:
    def test_lower_ph_more_ionized_base(self):
        """Base at low pH has higher ionized fraction."""
        low = make_base(urine_pH=5.0, pKa=9.5)
        high = make_base(urine_pH=8.0, pKa=9.5)
        assert low.ionized_fraction > high.ionized_fraction

    def test_lower_ph_higher_cl_renal_base(self):
        """Acidic urine enhances base excretion."""
        low = make_base(urine_pH=5.0, pKa=9.5)
        high = make_base(urine_pH=8.0, pKa=9.5)
        assert low.cl_renal_L_per_h > high.cl_renal_L_per_h


# ---------------------------------------------------------------------------
# Neutral behaviour
# ---------------------------------------------------------------------------


class TestNeutralBehaviour:
    def test_neutral_ionized_fraction_zero(self):
        r = make_neutral()
        assert r.ionized_fraction == pytest.approx(0.0)

    def test_neutral_f_reabs_is_095(self):
        r = make_neutral()
        assert r.f_reabs == pytest.approx(0.95)

    def test_neutral_cl_renal_same_at_all_ph(self):
        r1 = make_neutral(urine_pH=5.0)
        r2 = make_neutral(urine_pH=8.0)
        assert r1.cl_renal_L_per_h == pytest.approx(r2.cl_renal_L_per_h)


# ---------------------------------------------------------------------------
# t_half formula
# ---------------------------------------------------------------------------


class TestTHalf:
    def test_t_half_formula(self):
        fup = 0.15
        vd = 60.0
        cl_nonrenal = 4.0
        gfr = 7.2
        r = predict_urinary_excretion(
            "drug",
            4.0,
            "acid",
            7.0,
            fup=fup,
            vd_L=vd,
            cl_nonrenal_L_per_h=cl_nonrenal,
            gfr_L_per_h=gfr,
        )
        cl_total = r.cl_renal_L_per_h + cl_nonrenal
        expected_t_half = 0.693 * vd / cl_total
        assert r.t_half_h == pytest.approx(expected_t_half, rel=1e-6)


# ---------------------------------------------------------------------------
# CL_secretion
# ---------------------------------------------------------------------------


class TestSecretion:
    def test_secretion_increases_cl_renal(self):
        no_sec = predict_urinary_excretion("drug", 4.0, "acid", 6.0, cl_secretion_L_per_h=0.0)
        with_sec = predict_urinary_excretion("drug", 4.0, "acid", 6.0, cl_secretion_L_per_h=2.0)
        assert with_sec.cl_renal_L_per_h == pytest.approx(no_sec.cl_renal_L_per_h + 2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Clinical application classification
# ---------------------------------------------------------------------------


class TestClinicalApplication:
    def test_salicylate_overdose(self):
        r = predict_urinary_excretion("aspirin", 3.5, "acid", 6.0)
        assert r.clinical_application == "salicylate_overdose"

    def test_amphetamine_overdose(self):
        r = predict_urinary_excretion("amphetamine", 9.9, "base", 6.0)
        assert r.clinical_application == "amphetamine_overdose"

    def test_general_acid(self):
        # pKa=6.0 is outside [3,5] so should be "general"
        r = predict_urinary_excretion("warfarin", 6.0, "acid", 6.0)
        assert r.clinical_application == "general"

    def test_general_base(self):
        r = predict_urinary_excretion("metformin", 12.4, "base", 6.0)
        assert r.clinical_application == "general"

    def test_general_neutral(self):
        r = predict_urinary_excretion("caffeine", 7.0, "neutral", 6.0)
        assert r.clinical_application == "general"


# ---------------------------------------------------------------------------
# compare_urine_ph
# ---------------------------------------------------------------------------


class TestCompareUrinePH:
    def test_returns_list(self):
        result = compare_urine_ph("aspirin", 3.5, "acid", [5.0, 6.0, 8.0])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sorted_by_cl_renal_descending(self):
        results = compare_urine_ph("aspirin", 3.5, "acid", [5.0, 6.0, 8.0])
        cls = [r.cl_renal_L_per_h for r in results]
        assert cls == sorted(cls, reverse=True)

    def test_all_results_are_urinary_ph_result(self):
        results = compare_urine_ph("drug", 9.5, "base", [5.0, 7.0])
        for r in results:
            assert isinstance(r, UrinaryPHResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_fup_zero(self):
        with pytest.raises(ValueError, match="fup"):
            predict_urinary_excretion("drug", 4.0, "acid", 6.0, fup=0.0)

    def test_invalid_fup_negative(self):
        with pytest.raises(ValueError, match="fup"):
            predict_urinary_excretion("drug", 4.0, "acid", 6.0, fup=-0.1)

    def test_invalid_fup_above_1(self):
        with pytest.raises(ValueError, match="fup"):
            predict_urinary_excretion("drug", 4.0, "acid", 6.0, fup=1.5)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            predict_urinary_excretion("drug", 4.0, "acid", 6.0, vd_L=0.0)

    def test_invalid_cl_nonrenal_zero(self):
        with pytest.raises(ValueError, match="cl_nonrenal"):
            predict_urinary_excretion("drug", 4.0, "acid", 6.0, cl_nonrenal_L_per_h=0.0)

    def test_urine_ph_too_low(self):
        with pytest.raises(ValueError, match="urine_pH"):
            predict_urinary_excretion("drug", 4.0, "acid", 0.5)

    def test_urine_ph_too_high(self):
        with pytest.raises(ValueError, match="urine_pH"):
            predict_urinary_excretion("drug", 4.0, "acid", 9.5)

    def test_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_urinary_excretion("drug", 4.0, "zwitterion", 6.0)
