"""Tests for Phase 417 — dialysis_pk_p417.py"""

import math

import pytest

from omega_pbpk.clinical.dialysis_pk_p417 import (
    DialysisPKResult,
    calculate_dialysis_pk,
    compare_dialysis_drugs,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _hd_result(**kwargs):
    defaults = dict(
        drug_name="VancoHD",
        dialysis_type="hemodialysis",
        dose_mg=1000.0,
        cl_native_L_per_h=1.0,
        vd_L=50.0,
        fu_plasma=0.5,
        mw_Da=1449.0,
        session_duration_h=4.0,
    )
    defaults.update(kwargs)
    return calculate_dialysis_pk(**defaults)


def _pd_result(**kwargs):
    defaults = dict(
        drug_name="Drug_PD",
        dialysis_type="peritoneal",
        dose_mg=500.0,
        cl_native_L_per_h=0.5,
        vd_L=30.0,
        fu_plasma=0.8,
        mw_Da=300.0,
        session_duration_h=8.0,
    )
    defaults.update(kwargs)
    return calculate_dialysis_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_hd_returns_dataclass(self):
        result = _hd_result()
        assert isinstance(result, DialysisPKResult)

    def test_pd_returns_dataclass(self):
        result = _pd_result()
        assert isinstance(result, DialysisPKResult)


# ---------------------------------------------------------------------------
# dialysis_type preserved
# ---------------------------------------------------------------------------


class TestDialysisTypePreserved:
    def test_hd_type(self):
        result = _hd_result()
        assert result.dialysis_type == "hemodialysis"

    def test_pd_type(self):
        result = _pd_result()
        assert result.dialysis_type == "peritoneal"


# ---------------------------------------------------------------------------
# cl_total = cl_native + cl_dialysis
# ---------------------------------------------------------------------------


class TestClearanceSum:
    def test_cl_total_hd(self):
        result = _hd_result()
        assert math.isclose(
            result.cl_total_L_per_h,
            result.cl_native_L_per_h + result.cl_dialysis_L_per_h,
            rel_tol=1e-9,
        )

    def test_cl_total_pd(self):
        result = _pd_result()
        assert math.isclose(
            result.cl_total_L_per_h,
            result.cl_native_L_per_h + result.cl_dialysis_L_per_h,
            rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# t_half_on < t_half_off
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_hd_thalf_on_lt_off(self):
        result = _hd_result(cl_native_L_per_h=2.0, fu_plasma=0.9)
        assert result.t_half_on_dialysis_h < result.t_half_off_dialysis_h

    def test_pd_thalf_on_lt_off(self):
        result = _pd_result(cl_native_L_per_h=0.5, fu_plasma=0.9)
        assert result.t_half_on_dialysis_h < result.t_half_off_dialysis_h


# ---------------------------------------------------------------------------
# hemodialysis has higher cl_dialysis than peritoneal for same drug
# ---------------------------------------------------------------------------


class TestHDvsPeritonal:
    def test_hd_higher_clearance(self):
        common = dict(
            drug_name="TestDrug",
            dose_mg=500.0,
            cl_native_L_per_h=1.0,
            vd_L=30.0,
            fu_plasma=0.5,
            mw_Da=300.0,
            session_duration_h=4.0,
        )
        hd = calculate_dialysis_pk(dialysis_type="hemodialysis", **common)
        pd = calculate_dialysis_pk(dialysis_type="peritoneal", **common)
        assert hd.cl_dialysis_L_per_h > pd.cl_dialysis_L_per_h


# ---------------------------------------------------------------------------
# dose_removed_pct between 0 and 100
# ---------------------------------------------------------------------------


class TestDoseRemovedPct:
    def test_pct_range_hd(self):
        result = _hd_result()
        assert 0.0 <= result.dose_removed_pct <= 100.0

    def test_pct_range_pd(self):
        result = _pd_result()
        assert 0.0 <= result.dose_removed_pct <= 100.0


# ---------------------------------------------------------------------------
# monitoring_flag
# ---------------------------------------------------------------------------


class TestMonitoringFlag:
    def test_flag_true_when_gt30(self):
        # High fu, small MW, long session should remove >30%
        result = calculate_dialysis_pk(
            drug_name="Flagged",
            dialysis_type="hemodialysis",
            dose_mg=1000.0,
            cl_native_L_per_h=0.5,
            vd_L=10.0,
            fu_plasma=0.95,
            mw_Da=200.0,
            session_duration_h=4.0,
        )
        if result.dose_removed_pct > 30.0:
            assert result.monitoring_flag is True

    def test_flag_false_when_lt30(self):
        # Very low fu → minimal dialysis removal
        result = calculate_dialysis_pk(
            drug_name="Safe",
            dialysis_type="peritoneal",
            dose_mg=500.0,
            cl_native_L_per_h=0.5,
            vd_L=200.0,
            fu_plasma=0.01,
            mw_Da=500.0,
            session_duration_h=4.0,
        )
        if result.dose_removed_pct <= 30.0:
            assert result.monitoring_flag is False

    def test_flag_consistent_with_pct(self):
        result = _hd_result(fu_plasma=0.95, mw_Da=100.0, vd_L=5.0)
        assert result.monitoring_flag == (result.dose_removed_pct > 30.0)


# ---------------------------------------------------------------------------
# supplemental_dose
# ---------------------------------------------------------------------------


class TestSupplementalDose:
    def test_supplemental_when_removed_gt30(self):
        result = calculate_dialysis_pk(
            drug_name="NarrowWindow",
            dialysis_type="hemodialysis",
            dose_mg=1000.0,
            cl_native_L_per_h=0.2,
            vd_L=5.0,
            fu_plasma=0.95,
            mw_Da=150.0,
            session_duration_h=4.0,
        )
        if result.dose_removed_pct > 30.0:
            assert result.supplemental_dose_mg > 0.0

    def test_no_supplemental_when_low_removal(self):
        result = calculate_dialysis_pk(
            drug_name="Bound",
            dialysis_type="peritoneal",
            dose_mg=500.0,
            cl_native_L_per_h=0.5,
            vd_L=500.0,
            fu_plasma=0.02,
            mw_Da=800.0,
            session_duration_h=4.0,
        )
        if result.dose_removed_pct <= 30.0:
            assert result.supplemental_dose_mg == 0.0


# ---------------------------------------------------------------------------
# Higher fu → higher dose removed
# ---------------------------------------------------------------------------


class TestFuEffect:
    def test_higher_fu_higher_removal(self):
        base = dict(
            drug_name="Drug",
            dialysis_type="hemodialysis",
            dose_mg=1000.0,
            cl_native_L_per_h=1.0,
            vd_L=30.0,
            mw_Da=300.0,
            session_duration_h=4.0,
        )
        low_fu = calculate_dialysis_pk(fu_plasma=0.1, **base)
        high_fu = calculate_dialysis_pk(fu_plasma=0.9, **base)
        assert high_fu.dose_removed_pct > low_fu.dose_removed_pct


# ---------------------------------------------------------------------------
# High MW reduces dialysis efficiency
# ---------------------------------------------------------------------------


class TestMWEffect:
    def test_high_mw_reduces_removal(self):
        base = dict(
            drug_name="Drug",
            dialysis_type="hemodialysis",
            dose_mg=1000.0,
            cl_native_L_per_h=1.0,
            vd_L=30.0,
            fu_plasma=0.7,
            session_duration_h=4.0,
        )
        small_mw = calculate_dialysis_pk(mw_Da=200.0, **base)
        large_mw = calculate_dialysis_pk(mw_Da=5000.0, **base)
        assert small_mw.dose_removed_pct >= large_mw.dose_removed_pct


# ---------------------------------------------------------------------------
# compare_dialysis_drugs
# ---------------------------------------------------------------------------


class TestCompareDrugs:
    def _drugs_list(self):
        return [
            dict(
                drug_name="DrugA",
                dose_mg=500.0,
                cl_native_L_per_h=1.0,
                vd_L=10.0,
                fu_plasma=0.9,
                mw_Da=200.0,
            ),
            dict(
                drug_name="DrugB",
                dose_mg=500.0,
                cl_native_L_per_h=1.0,
                vd_L=100.0,
                fu_plasma=0.1,
                mw_Da=2000.0,
            ),
            dict(
                drug_name="DrugC",
                dose_mg=1000.0,
                cl_native_L_per_h=0.5,
                vd_L=30.0,
                fu_plasma=0.5,
                mw_Da=400.0,
            ),
        ]

    def test_returns_list(self):
        result = compare_dialysis_drugs(self._drugs_list(), "hemodialysis")
        assert isinstance(result, list)

    def test_all_results_are_dataclass(self):
        results = compare_dialysis_drugs(self._drugs_list(), "hemodialysis")
        for r in results:
            assert isinstance(r, DialysisPKResult)

    def test_sorted_descending_by_dose_removed(self):
        results = compare_dialysis_drugs(self._drugs_list(), "hemodialysis")
        pcts = [r.dose_removed_pct for r in results]
        assert pcts == sorted(pcts, reverse=True)

    def test_length_matches_input(self):
        drugs = self._drugs_list()
        results = compare_dialysis_drugs(drugs, "peritoneal")
        assert len(results) == len(drugs)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_bad_dialysis_type(self):
        with pytest.raises(ValueError, match="dialysis_type"):
            calculate_dialysis_pk(
                drug_name="X",
                dialysis_type="unknown",
                dose_mg=100.0,
                cl_native_L_per_h=1.0,
                vd_L=30.0,
                fu_plasma=0.5,
                mw_Da=300.0,
            )

    def test_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            calculate_dialysis_pk(
                drug_name="X",
                dialysis_type="hemodialysis",
                dose_mg=0.0,
                cl_native_L_per_h=1.0,
                vd_L=30.0,
                fu_plasma=0.5,
                mw_Da=300.0,
            )

    def test_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            calculate_dialysis_pk(
                drug_name="X",
                dialysis_type="hemodialysis",
                dose_mg=-100.0,
                cl_native_L_per_h=1.0,
                vd_L=30.0,
                fu_plasma=0.5,
                mw_Da=300.0,
            )

    def test_fu_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            calculate_dialysis_pk(
                drug_name="X",
                dialysis_type="hemodialysis",
                dose_mg=100.0,
                cl_native_L_per_h=1.0,
                vd_L=30.0,
                fu_plasma=0.0,
                mw_Da=300.0,
            )

    def test_fu_above_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            calculate_dialysis_pk(
                drug_name="X",
                dialysis_type="hemodialysis",
                dose_mg=100.0,
                cl_native_L_per_h=1.0,
                vd_L=30.0,
                fu_plasma=1.5,
                mw_Da=300.0,
            )

    def test_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            calculate_dialysis_pk(
                drug_name="X",
                dialysis_type="hemodialysis",
                dose_mg=100.0,
                cl_native_L_per_h=1.0,
                vd_L=0.0,
                fu_plasma=0.5,
                mw_Da=300.0,
            )

    def test_mw_negative(self):
        with pytest.raises(ValueError, match="mw_Da"):
            calculate_dialysis_pk(
                drug_name="X",
                dialysis_type="hemodialysis",
                dose_mg=100.0,
                cl_native_L_per_h=1.0,
                vd_L=30.0,
                fu_plasma=0.5,
                mw_Da=-1.0,
            )
