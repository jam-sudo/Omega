"""Tests for Phase 535 — Biliary Excretion Extended model."""

import pytest

from omega_pbpk.core.biliary_excretion_extended_p535 import (
    BiliaryExcretionExtendedResult,
    _compute_f_bile,
    compare_mw_biliary,
    simulate_biliary_excretion_extended,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_sim(mw_Da=600.0, dose_mg=100.0, f_deconj=0.7):
    return simulate_biliary_excretion_extended(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        cl_total_L_per_h=5.0,
        vd_L=50.0,
        f_deconj=f_deconj,
        t_end_h=48.0,
        dt_h=0.2,
    )


# ---------------------------------------------------------------------------
# Return type and field checks
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _default_sim()
        assert isinstance(result, BiliaryExcretionExtendedResult)

    def test_has_drug_name(self):
        result = _default_sim()
        assert result.drug_name == "TestDrug"

    def test_has_dose_mg(self):
        result = _default_sim(dose_mg=200.0)
        assert result.dose_mg == 200.0

    def test_has_mw_Da(self):
        result = _default_sim(mw_Da=700.0)
        assert result.mw_Da == 700.0

    def test_has_f_bile(self):
        result = _default_sim(mw_Da=700.0)
        assert 0.0 <= result.f_bile <= 0.8

    def test_has_f_deconj(self):
        result = _default_sim(f_deconj=0.5)
        assert result.f_deconj == 0.5

    def test_has_times_list(self):
        result = _default_sim()
        assert isinstance(result.times_h, list)
        assert len(result.times_h) > 0

    def test_has_c_plasma_list(self):
        result = _default_sim()
        assert isinstance(result.c_plasma_mg_L, list)
        assert len(result.c_plasma_mg_L) == len(result.times_h)

    def test_has_a_bile_list(self):
        result = _default_sim()
        assert isinstance(result.a_bile_mg, list)
        assert len(result.a_bile_mg) == len(result.times_h)

    def test_has_cmax(self):
        result = _default_sim()
        assert result.cmax_plasma > 0.0

    def test_has_tmax(self):
        result = _default_sim()
        assert result.tmax_plasma_h >= 0.0

    def test_has_auc(self):
        result = _default_sim()
        assert result.auc_plasma > 0.0

    def test_has_biliary_fraction_pct(self):
        result = _default_sim(mw_Da=900.0)
        assert result.biliary_fraction_pct > 0.0

    def test_has_recirculation_cycles(self):
        result = _default_sim()
        assert isinstance(result.recirculation_cycles, float)

    def test_has_effective_t_half(self):
        result = _default_sim()
        assert result.effective_t_half_h > 0.0

    def test_has_ehr_amplification(self):
        result = _default_sim()
        assert isinstance(result.ehr_amplification, bool)

    def test_has_notes(self):
        result = _default_sim()
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# MW threshold behaviour
# ---------------------------------------------------------------------------


class TestMWThreshold:
    def test_low_mw_no_bile(self):
        result = simulate_biliary_excretion_extended(
            drug_name="SmallDrug",
            dose_mg=100,
            mw_Da=200,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result.f_bile == 0.0

    def test_mw_exactly_400_no_bile(self):
        result = simulate_biliary_excretion_extended(
            drug_name="SmallDrug",
            dose_mg=100,
            mw_Da=400,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result.f_bile == 0.0

    def test_mw_above_400_has_bile(self):
        result = simulate_biliary_excretion_extended(
            drug_name="LargeDrug",
            dose_mg=100,
            mw_Da=700,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result.f_bile > 0.0

    def test_mw_900_bile_capped_at_0_8(self):
        result = simulate_biliary_excretion_extended(
            drug_name="HugeDrug",
            dose_mg=100,
            mw_Da=900,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result.f_bile == 0.8

    def test_compute_f_bile_helper_low(self):
        assert _compute_f_bile(300.0) == 0.0

    def test_compute_f_bile_helper_mid(self):
        val = _compute_f_bile(650.0)  # (650-400)/500 = 0.5
        assert abs(val - 0.5) < 1e-9

    def test_biliary_fraction_pct_equals_f_bile_times_100(self):
        result = _default_sim(mw_Da=650.0)
        assert abs(result.biliary_fraction_pct - result.f_bile * 100.0) < 1e-9


# ---------------------------------------------------------------------------
# ODE behaviour
# ---------------------------------------------------------------------------


class TestODEBehaviour:
    def test_a_bile_starts_at_zero(self):
        result = _default_sim()
        assert result.a_bile_mg[0] == 0.0

    def test_a_bile_rises_for_high_mw(self):
        result = _default_sim(mw_Da=800.0)
        assert max(result.a_bile_mg) > 0.0

    def test_a_bile_stays_zero_for_low_mw(self):
        result = simulate_biliary_excretion_extended(
            drug_name="SmallDrug",
            dose_mg=100,
            mw_Da=200,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
            t_end_h=48,
            dt_h=0.2,
        )
        # With no biliary clearance, a_bile should stay at zero
        assert max(result.a_bile_mg) == 0.0

    def test_c_plasma_starts_at_dose_over_vd(self):
        result = simulate_biliary_excretion_extended(
            drug_name="D",
            dose_mg=100,
            mw_Da=200,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
        )
        assert abs(result.c_plasma_mg_L[0] - 100.0 / 50.0) < 1e-9

    def test_c_plasma_non_negative(self):
        result = _default_sim()
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# EHR amplification
# ---------------------------------------------------------------------------


class TestEHRAmplification:
    def test_ehr_amplification_true_for_high_bile_and_deconj(self):
        result = simulate_biliary_excretion_extended(
            drug_name="EHRDrug",
            dose_mg=100,
            mw_Da=900,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
            f_deconj=0.8,
        )
        # f_bile=0.8 > 0.3 and f_deconj=0.8 > 0.5
        assert result.ehr_amplification is True

    def test_ehr_amplification_false_for_low_bile(self):
        result = simulate_biliary_excretion_extended(
            drug_name="NoEHR",
            dose_mg=100,
            mw_Da=450,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
            f_deconj=0.8,
        )
        # f_bile = (450-400)/500 = 0.1 < 0.3
        assert result.ehr_amplification is False

    def test_ehr_amplification_false_for_low_deconj(self):
        result = simulate_biliary_excretion_extended(
            drug_name="NoDeconj",
            dose_mg=100,
            mw_Da=900,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
            f_deconj=0.3,
        )
        # f_bile=0.8 > 0.3, but f_deconj=0.3 < 0.5
        assert result.ehr_amplification is False

    def test_recirculation_cycles_is_f_bile_times_f_deconj(self):
        result = _default_sim(mw_Da=700.0, f_deconj=0.6)
        expected = result.f_bile * result.f_deconj
        assert abs(result.recirculation_cycles - expected) < 1e-9


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_auc_doubles_with_double_dose(self):
        r1 = simulate_biliary_excretion_extended(
            drug_name="D",
            dose_mg=100,
            mw_Da=600,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
            t_end_h=48,
            dt_h=0.2,
        )
        r2 = simulate_biliary_excretion_extended(
            drug_name="D",
            dose_mg=200,
            mw_Da=600,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
            t_end_h=48,
            dt_h=0.2,
        )
        assert abs(r2.auc_plasma / r1.auc_plasma - 2.0) < 0.01

    def test_cmax_doubles_with_double_dose(self):
        r1 = simulate_biliary_excretion_extended(
            drug_name="D",
            dose_mg=100,
            mw_Da=200,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
        )
        r2 = simulate_biliary_excretion_extended(
            drug_name="D",
            dose_mg=200,
            mw_Da=200,
            cl_total_L_per_h=5.0,
            vd_L=50.0,
        )
        assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_raises_on_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_biliary_excretion_extended(
                drug_name="D",
                dose_mg=0,
                mw_Da=600,
                cl_total_L_per_h=5.0,
                vd_L=50.0,
            )

    def test_raises_on_negative_dose(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion_extended(
                drug_name="D",
                dose_mg=-10,
                mw_Da=600,
                cl_total_L_per_h=5.0,
                vd_L=50.0,
            )

    def test_raises_on_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_biliary_excretion_extended(
                drug_name="D",
                dose_mg=100,
                mw_Da=0,
                cl_total_L_per_h=5.0,
                vd_L=50.0,
            )

    def test_raises_on_zero_cl(self):
        with pytest.raises(ValueError, match="cl_total"):
            simulate_biliary_excretion_extended(
                drug_name="D",
                dose_mg=100,
                mw_Da=600,
                cl_total_L_per_h=0,
                vd_L=50.0,
            )

    def test_raises_on_zero_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_biliary_excretion_extended(
                drug_name="D",
                dose_mg=100,
                mw_Da=600,
                cl_total_L_per_h=5.0,
                vd_L=0,
            )

    def test_raises_on_f_deconj_out_of_range(self):
        with pytest.raises(ValueError, match="f_deconj"):
            simulate_biliary_excretion_extended(
                drug_name="D",
                dose_mg=100,
                mw_Da=600,
                cl_total_L_per_h=5.0,
                vd_L=50.0,
                f_deconj=1.5,
            )

    def test_raises_on_negative_f_deconj(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion_extended(
                drug_name="D",
                dose_mg=100,
                mw_Da=600,
                cl_total_L_per_h=5.0,
                vd_L=50.0,
                f_deconj=-0.1,
            )


# ---------------------------------------------------------------------------
# compare_mw_biliary
# ---------------------------------------------------------------------------


class TestCompareMWBiliary:
    def test_returns_list(self):
        result = compare_mw_biliary("Drug", 100.0, 5.0, 50.0, [300.0, 600.0, 900.0])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sorted_by_biliary_fraction_descending(self):
        result = compare_mw_biliary("Drug", 100.0, 5.0, 50.0, [300.0, 600.0, 900.0])
        fracs = [r.biliary_fraction_pct for r in result]
        assert fracs == sorted(fracs, reverse=True)

    def test_all_results_are_correct_type(self):
        result = compare_mw_biliary("Drug", 100.0, 5.0, 50.0, [400.0, 700.0])
        assert all(isinstance(r, BiliaryExcretionExtendedResult) for r in result)

    def test_low_mw_last(self):
        result = compare_mw_biliary("Drug", 100.0, 5.0, 50.0, [200.0, 900.0])
        assert result[-1].biliary_fraction_pct == 0.0  # 200 Da has no biliary excretion
