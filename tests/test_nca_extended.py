"""Tests for extended NCA analysis module (Phase 410)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.analysis.nca_extended import (
    ExtendedNCAResult,
    compare_nca_profiles,
    extended_nca,
)

# ---------------------------------------------------------------------------
# Synthetic PK profiles
# ---------------------------------------------------------------------------

# 1-compartment IV: C(t) = (Dose/Vd) * exp(-ke*t)
KE_IV = 0.15  # 1/h, elimination rate
VD_IV = 40.0  # L
DOSE_IV = 100.0  # mg
T_IV = list(np.linspace(0.01, 24.0, 50))
C_IV = [DOSE_IV / VD_IV * math.exp(-KE_IV * t) for t in T_IV]

# 1-compartment oral: C(t) = F*Dose*ka/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
KA_ORAL = 0.8  # 1/h
KE_ORAL = 0.1  # 1/h
VD_ORAL = 50.0  # L
F_ORAL = 1.0
DOSE_ORAL = 200.0  # mg
T_ORAL = list(np.linspace(0.05, 48.0, 100))
C_ORAL = [
    F_ORAL * DOSE_ORAL * KA_ORAL / (VD_ORAL * (KA_ORAL - KE_ORAL))
    * (math.exp(-KE_ORAL * t) - math.exp(-KA_ORAL * t))
    for t in T_ORAL
]


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


class TestExtendedNCAResult:
    def test_returns_correct_type(self):
        result = extended_nca(
            drug_name="test_iv",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert isinstance(result, ExtendedNCAResult)

    def test_required_fields_present(self):
        result = extended_nca(
            drug_name="test_iv",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        required = {
            "drug_name",
            "route",
            "dose_mg",
            "cmax",
            "tmax_h",
            "auc_last",
            "auc_inf",
            "lambda_z",
            "t_half_h",
            "aumc_inf",
            "mrt_h",
            "vss_L",
            "vz_L",
            "cl_L_per_h",
            "mat_h",
            "r_squared_terminal",
            "pct_extrapolated",
            "notes",
        }
        for f in required:
            assert hasattr(result, f), f"Missing field: {f}"

    def test_drug_name_stored(self):
        result = extended_nca(
            drug_name="warfarin",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert result.drug_name == "warfarin"

    def test_route_stored(self):
        result = extended_nca(
            drug_name="drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert result.route == "iv"


# ---------------------------------------------------------------------------
# IV route tests
# ---------------------------------------------------------------------------


class TestIVRoute:
    def test_cmax_at_first_point(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        # IV: Cmax should be at t=0 (or earliest time)
        assert result.cmax == pytest.approx(max(C_IV), rel=1e-6)

    def test_lambda_z_positive(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert result.lambda_z > 0

    def test_lambda_z_close_to_ke(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
            n_terminal=10,
        )
        assert result.lambda_z == pytest.approx(KE_IV, rel=0.05)

    def test_t_half_formula(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        expected_t_half = math.log(2.0) / result.lambda_z
        assert result.t_half_h == pytest.approx(expected_t_half, rel=1e-6)

    def test_auc_last_positive(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert result.auc_last > 0

    def test_auc_inf_greater_than_auc_last(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert result.auc_inf > result.auc_last

    def test_vss_finite_for_iv(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert math.isfinite(result.vss_L)
        assert result.vss_L > 0

    def test_cl_approx_dose_over_auc(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        expected_cl = DOSE_IV / result.auc_inf
        assert result.cl_L_per_h == pytest.approx(expected_cl, rel=1e-6)

    def test_mat_is_none_for_iv(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert result.mat_h is None

    def test_r_squared_high_for_monoexponential(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
            n_terminal=10,
        )
        assert result.r_squared_terminal > 0.98

    def test_mrt_positive(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert result.mrt_h > 0

    def test_aumc_inf_positive(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )
        assert result.aumc_inf > 0

    def test_vz_approx_vd(self):
        result = extended_nca(
            drug_name="iv_drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
            n_terminal=15,
        )
        # Vz = Dose / (AUC_inf * lambda_z) ≈ Vd for 1-compartment
        assert result.vz_L == pytest.approx(VD_IV, rel=0.1)


# ---------------------------------------------------------------------------
# Oral route tests
# ---------------------------------------------------------------------------


class TestOralRoute:
    def test_oral_cmax_positive(self):
        result = extended_nca(
            drug_name="oral_drug",
            times_h=T_ORAL,
            c_plasma=C_ORAL,
            dose_mg=DOSE_ORAL,
            route="oral",
        )
        assert result.cmax > 0

    def test_oral_tmax_positive(self):
        result = extended_nca(
            drug_name="oral_drug",
            times_h=T_ORAL,
            c_plasma=C_ORAL,
            dose_mg=DOSE_ORAL,
            route="oral",
        )
        assert result.tmax_h > 0

    def test_mat_not_none_for_oral(self):
        result = extended_nca(
            drug_name="oral_drug",
            times_h=T_ORAL,
            c_plasma=C_ORAL,
            dose_mg=DOSE_ORAL,
            route="oral",
            n_terminal=10,
        )
        assert result.mat_h is not None

    def test_vss_nan_for_oral(self):
        result = extended_nca(
            drug_name="oral_drug",
            times_h=T_ORAL,
            c_plasma=C_ORAL,
            dose_mg=DOSE_ORAL,
            route="oral",
        )
        assert math.isnan(result.vss_L)

    def test_cl_positive_for_oral(self):
        result = extended_nca(
            drug_name="oral_drug",
            times_h=T_ORAL,
            c_plasma=C_ORAL,
            dose_mg=DOSE_ORAL,
            route="oral",
        )
        assert result.cl_L_per_h > 0

    def test_lambda_z_close_to_ke_oral(self):
        result = extended_nca(
            drug_name="oral_drug",
            times_h=T_ORAL,
            c_plasma=C_ORAL,
            dose_mg=DOSE_ORAL,
            route="oral",
            n_terminal=15,
        )
        assert result.lambda_z == pytest.approx(KE_ORAL, rel=0.05)


# ---------------------------------------------------------------------------
# Swing and peak-trough tests
# ---------------------------------------------------------------------------


class TestSwingPeakTrough:
    def test_cmin_produces_swing_note(self):
        result = extended_nca(
            drug_name="drug",
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
            cmin_mg_L=0.1,
        )
        assert any("Swing" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_too_few_points(self):
        with pytest.raises(ValueError, match="3 time points"):
            extended_nca(
                drug_name="drug",
                times_h=[0.0, 1.0],
                c_plasma=[1.0, 0.8],
                dose_mg=100.0,
                route="iv",
            )

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            extended_nca(
                drug_name="drug",
                times_h=[0.0, 1.0, 2.0],
                c_plasma=[1.0, 0.8],
                dose_mg=100.0,
                route="iv",
            )

    def test_non_increasing_times(self):
        with pytest.raises(ValueError, match="monotonically"):
            extended_nca(
                drug_name="drug",
                times_h=[0.0, 2.0, 1.0],
                c_plasma=[1.0, 0.8, 0.6],
                dose_mg=100.0,
                route="iv",
            )

    def test_negative_concentration(self):
        with pytest.raises(ValueError, match=">="):
            extended_nca(
                drug_name="drug",
                times_h=[0.0, 1.0, 2.0],
                c_plasma=[1.0, -0.1, 0.5],
                dose_mg=100.0,
                route="iv",
            )

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            extended_nca(
                drug_name="drug",
                times_h=T_IV,
                c_plasma=C_IV,
                dose_mg=0.0,
                route="iv",
            )

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            extended_nca(
                drug_name="drug",
                times_h=T_IV,
                c_plasma=C_IV,
                dose_mg=100.0,
                route="im",
            )

    def test_n_terminal_too_small(self):
        with pytest.raises(ValueError, match="n_terminal"):
            extended_nca(
                drug_name="drug",
                times_h=T_IV,
                c_plasma=C_IV,
                dose_mg=100.0,
                route="iv",
                n_terminal=2,
            )

    def test_invalid_f_oral(self):
        with pytest.raises(ValueError, match="f_oral"):
            extended_nca(
                drug_name="drug",
                times_h=T_ORAL,
                c_plasma=C_ORAL,
                dose_mg=100.0,
                route="oral",
                f_oral=0.0,
            )

    def test_negative_cmin(self):
        with pytest.raises(ValueError, match="cmin_mg_L"):
            extended_nca(
                drug_name="drug",
                times_h=T_IV,
                c_plasma=C_IV,
                dose_mg=100.0,
                route="iv",
                cmin_mg_L=-1.0,
            )


# ---------------------------------------------------------------------------
# compare_nca_profiles tests
# ---------------------------------------------------------------------------


class TestCompareNCAProfiles:
    def _make_result(self, name: str) -> ExtendedNCAResult:
        return extended_nca(
            drug_name=name,
            times_h=T_IV,
            c_plasma=C_IV,
            dose_mg=DOSE_IV,
            route="iv",
        )

    def test_returns_dict(self):
        r1 = self._make_result("drug_a")
        r2 = self._make_result("drug_b")
        table = compare_nca_profiles([r1, r2])
        assert isinstance(table, dict)

    def test_drug_name_key(self):
        r1 = self._make_result("drug_a")
        table = compare_nca_profiles([r1])
        assert "drug_name" in table
        assert table["drug_name"] == ["drug_a"]

    def test_all_expected_keys_present(self):
        r1 = self._make_result("drug_a")
        table = compare_nca_profiles([r1])
        expected_keys = {
            "drug_name",
            "route",
            "dose_mg",
            "cmax",
            "tmax_h",
            "auc_last",
            "auc_inf",
            "lambda_z",
            "t_half_h",
            "aumc_inf",
            "mrt_h",
            "vss_L",
            "vz_L",
            "cl_L_per_h",
            "mat_h",
            "r_squared_terminal",
            "pct_extrapolated",
        }
        assert expected_keys.issubset(set(table.keys()))

    def test_list_lengths_match_profiles(self):
        profiles = [self._make_result(f"drug_{i}") for i in range(3)]
        table = compare_nca_profiles(profiles)
        for key, values in table.items():
            assert len(values) == 3, f"Key '{key}' has wrong length."

    def test_empty_profiles_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compare_nca_profiles([])
