"""Tests for omega_pbpk.clinical.enantiomer_pk."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.clinical.enantiomer_pk import (
    EnantiomerPKResult,
    compare_enantiomers,
    simulate_enantiomer_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_r_L_per_h=5.0,
    vd_r_L=50.0,
    cl_s_L_per_h=3.0,
    vd_s_L=50.0,
)


def _run(**overrides):
    kw = {**_DEFAULT_KWARGS, **overrides}
    return simulate_enantiomer_pk(**kw)


# ---------------------------------------------------------------------------
# 1. Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_cl_r_zero_raises(self):
        with pytest.raises(ValueError, match="cl_r_L_per_h"):
            _run(cl_r_L_per_h=0.0)

    def test_vd_r_zero_raises(self):
        with pytest.raises(ValueError, match="vd_r_L"):
            _run(vd_r_L=0.0)

    def test_cl_s_zero_raises(self):
        with pytest.raises(ValueError, match="cl_s_L_per_h"):
            _run(cl_s_L_per_h=0.0)

    def test_vd_s_zero_raises(self):
        with pytest.raises(ValueError, match="vd_s_L"):
            _run(vd_s_L=0.0)

    def test_enantiomer_fraction_r_above_1_raises(self):
        with pytest.raises(ValueError, match="enantiomer_fraction_r"):
            _run(enantiomer_fraction_r=1.5)

    def test_enantiomer_fraction_r_negative_raises(self):
        with pytest.raises(ValueError, match="enantiomer_fraction_r"):
            _run(enantiomer_fraction_r=-0.1)

    def test_k_inv_rs_negative_raises(self):
        with pytest.raises(ValueError, match="k_inv_rs"):
            _run(k_inv_rs_per_h=-0.1)

    def test_k_inv_sr_negative_raises(self):
        with pytest.raises(ValueError, match="k_inv_sr"):
            _run(k_inv_sr_per_h=-0.1)


# ---------------------------------------------------------------------------
# 2. Return type and fields
# ---------------------------------------------------------------------------


class TestResultType:
    def test_returns_dataclass(self):
        result = _run()
        assert isinstance(result, EnantiomerPKResult)

    def test_has_all_expected_fields(self):
        result = _run()
        expected = {
            "drug_name",
            "dose_mg",
            "route",
            "racemate",
            "times_h",
            "c_r_mg_L",
            "c_s_mg_L",
            "cmax_r",
            "cmax_s",
            "tmax_r_h",
            "tmax_s_h",
            "auc_r",
            "auc_s",
            "auc_ratio_sr",
            "inversion_occurred",
            "notes",
        }
        for field in expected:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_times_h_is_ndarray(self):
        result = _run()
        assert isinstance(result.times_h, np.ndarray)

    def test_c_r_is_ndarray(self):
        result = _run()
        assert isinstance(result.c_r_mg_L, np.ndarray)

    def test_c_s_is_ndarray(self):
        result = _run()
        assert isinstance(result.c_s_mg_L, np.ndarray)

    def test_scalar_fields_are_float(self):
        result = _run()
        assert isinstance(result.cmax_r, float)
        assert isinstance(result.cmax_s, float)
        assert isinstance(result.auc_r, float)
        assert isinstance(result.auc_s, float)
        assert isinstance(result.auc_ratio_sr, float)


# ---------------------------------------------------------------------------
# 3. Racemate: both enantiomers non-zero
# ---------------------------------------------------------------------------


class TestRacemate:
    def test_racemate_both_concs_nonzero(self):
        result = _run(racemate=True)
        assert result.cmax_r > 0.0
        assert result.cmax_s > 0.0

    def test_racemate_true_both_auc_nonzero(self):
        result = _run(racemate=True)
        assert result.auc_r > 0.0
        assert result.auc_s > 0.0

    def test_racemate_default_50_50_split(self):
        result = _run(racemate=True, enantiomer_fraction_r=0.5)
        # With different CL_R vs CL_S, AUCs differ; but both are nonzero
        assert result.auc_r > 0.0
        assert result.auc_s > 0.0


# ---------------------------------------------------------------------------
# 4. No inversion: profiles from absorbed dose only
# ---------------------------------------------------------------------------


class TestNoInversion:
    def test_no_inversion_flag(self):
        result = _run(k_inv_rs_per_h=0.0, k_inv_sr_per_h=0.0)
        assert result.inversion_occurred is False

    def test_pure_r_no_inversion_s_approx_zero(self):
        result = _run(
            racemate=False,
            enantiomer_fraction_r=1.0,
            k_inv_rs_per_h=0.0,
            k_inv_sr_per_h=0.0,
        )
        assert result.cmax_s < 1e-9
        assert result.auc_s < 1e-9

    def test_pure_s_no_inversion_r_approx_zero(self):
        result = _run(
            racemate=False,
            enantiomer_fraction_r=0.0,
            k_inv_rs_per_h=0.0,
            k_inv_sr_per_h=0.0,
        )
        assert result.cmax_r < 1e-9
        assert result.auc_r < 1e-9


# ---------------------------------------------------------------------------
# 5. Inversion: R→S causes higher S exposure
# ---------------------------------------------------------------------------


class TestWithInversion:
    def test_inversion_occurred_flag_true(self):
        result = _run(k_inv_rs_per_h=0.1)
        assert result.inversion_occurred is True

    def test_rs_inversion_increases_s_auc(self):
        # Pure R dose with R→S inversion: S should get some exposure
        no_inv = _run(
            racemate=False,
            enantiomer_fraction_r=1.0,
            k_inv_rs_per_h=0.0,
        )
        with_inv = _run(
            racemate=False,
            enantiomer_fraction_r=1.0,
            k_inv_rs_per_h=0.2,
        )
        assert with_inv.auc_s > no_inv.auc_s

    def test_sr_inversion_flag(self):
        result = _run(k_inv_sr_per_h=0.1)
        assert result.inversion_occurred is True


# ---------------------------------------------------------------------------
# 6. Same CL/Vd for R and S → auc_ratio_sr ≈ 1
# ---------------------------------------------------------------------------


class TestSymmetricPK:
    def test_equal_pk_auc_ratio_near_1(self):
        result = simulate_enantiomer_pk(
            drug_name="Symmetric",
            dose_mg=100.0,
            cl_r_L_per_h=4.0,
            vd_r_L=40.0,
            cl_s_L_per_h=4.0,
            vd_s_L=40.0,
            racemate=True,
            enantiomer_fraction_r=0.5,
        )
        assert abs(result.auc_ratio_sr - 1.0) < 0.05


# ---------------------------------------------------------------------------
# 7. Higher CL_S → lower AUC_S → ratio < 1
# ---------------------------------------------------------------------------


class TestHigherCLS:
    def test_higher_cl_s_gives_lower_auc_s(self):
        result = simulate_enantiomer_pk(
            drug_name="Chiral",
            dose_mg=100.0,
            cl_r_L_per_h=2.0,
            vd_r_L=40.0,
            cl_s_L_per_h=10.0,  # much higher CL_S
            vd_s_L=40.0,
            racemate=True,
            enantiomer_fraction_r=0.5,
            k_inv_rs_per_h=0.0,
            k_inv_sr_per_h=0.0,
        )
        assert result.auc_ratio_sr < 1.0  # S/R < 1 because CL_S > CL_R


# ---------------------------------------------------------------------------
# 8. Dose linearity: 2× dose → 2× AUC
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_double_dose_doubles_auc_r(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert abs(r2.auc_r / r1.auc_r - 2.0) < 0.05

    def test_double_dose_doubles_auc_s(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert abs(r2.auc_s / r1.auc_s - 2.0) < 0.05


# ---------------------------------------------------------------------------
# 9. compare_enantiomers
# ---------------------------------------------------------------------------


class TestCompareEnantiomers:
    def test_returns_dict(self):
        result = compare_enantiomers(**_DEFAULT_KWARGS)
        assert isinstance(result, dict)

    def test_dict_has_expected_keys(self):
        result = compare_enantiomers(**_DEFAULT_KWARGS)
        for key in ["AUC_R", "AUC_S", "AUC_ratio", "Cmax_R", "Cmax_S", "notes"]:
            assert key in result, f"Missing key: {key}"

    def test_dict_inversion_occurred_key(self):
        result = compare_enantiomers(**_DEFAULT_KWARGS)
        assert "inversion_occurred" in result

    def test_dict_auc_ratio_consistent(self):
        result = compare_enantiomers(**_DEFAULT_KWARGS)
        expected_ratio = result["AUC_S"] / max(result["AUC_R"], 1e-12)
        assert abs(result["AUC_ratio"] - expected_ratio) < 0.01
