"""Tests for peritoneal PK simulation module (Phase 409)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.peritoneal_pk import (
    PeritonealPKResult,
    compare_ip_drugs,
    simulate_peritoneal_pk,
)

# ---------------------------------------------------------------------------
# Default parameter set
# ---------------------------------------------------------------------------

BASE = dict(
    drug_name="cisplatin",
    dose_mg=100.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    peritoneal_volume_L=2.0,
    k_peritoneal_abs_per_h=0.1,
    dialysis_flow_L_per_h=0.0,
    t_end_h=24.0,
    dt_h=0.05,
    application="ip_chemo",
)


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestPeritonealPKResult:
    def test_returns_correct_type(self):
        result = simulate_peritoneal_pk(**BASE)
        assert isinstance(result, PeritonealPKResult)

    def test_dataclass_fields_present(self):
        result = simulate_peritoneal_pk(**BASE)
        required = {
            "drug_name",
            "dose_mg",
            "peritoneal_volume_L",
            "application",
            "times_h",
            "c_peri_mg_L",
            "c_sys_mg_L",
            "cmax_sys",
            "auc_sys",
            "auc_peri",
            "auc_ratio_peri_sys",
            "notes",
        }
        for f in required:
            assert hasattr(result, f), f"Missing field: {f}"

    def test_drug_name_stored(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.drug_name == "cisplatin"

    def test_application_stored(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.application == "ip_chemo"

    def test_peritoneal_volume_stored(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.peritoneal_volume_L == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Initial condition tests
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_initial_peritoneal_conc(self):
        result = simulate_peritoneal_pk(**BASE)
        expected_c0 = BASE["dose_mg"] / BASE["peritoneal_volume_L"]
        assert result.c_peri_mg_L[0] == pytest.approx(expected_c0, rel=1e-6)

    def test_initial_systemic_conc_zero(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.c_sys_mg_L[0] == pytest.approx(0.0, abs=1e-10)

    def test_time_starts_at_zero(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ip_chemo application tests
# ---------------------------------------------------------------------------


class TestIPChemo:
    def test_peritoneal_conc_declines(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.c_peri_mg_L[-1] < result.c_peri_mg_L[0]

    def test_systemic_conc_rises_then_falls(self):
        result = simulate_peritoneal_pk(**BASE)
        c_sys = result.c_sys_mg_L
        cmax_idx = c_sys.index(max(c_sys))
        assert cmax_idx > 0
        assert c_sys[-1] < c_sys[cmax_idx]

    def test_cmax_sys_matches_array(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.cmax_sys == pytest.approx(max(result.c_sys_mg_L), rel=1e-6)

    def test_auc_peri_greater_than_auc_sys(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.auc_peri > result.auc_sys

    def test_auc_ratio_positive(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.auc_ratio_peri_sys > 1.0

    def test_ip_chemo_note_present(self):
        result = simulate_peritoneal_pk(**BASE)
        assert any("IP chemotherapy" in n for n in result.notes)

    def test_zero_dialysis_flow_for_ip_chemo(self):
        # With dialysis_flow provided but ip_chemo specified, dialysis is ignored
        params = {**BASE, "dialysis_flow_L_per_h": 5.0}
        result_with_flow = simulate_peritoneal_pk(**params)
        result_without = simulate_peritoneal_pk(**BASE)
        # ip_chemo ignores dialysis_flow, so results should be identical
        assert result_with_flow.auc_peri == pytest.approx(result_without.auc_peri, rel=1e-6)


# ---------------------------------------------------------------------------
# pd_dialysis application tests
# ---------------------------------------------------------------------------


class TestPDDialysis:
    def test_dialysis_reduces_peritoneal_conc(self):
        params_no_dialysis = {**BASE, "application": "pd_dialysis", "dialysis_flow_L_per_h": 0.0}
        params_dialysis = {
            **BASE,
            "application": "pd_dialysis",
            "dialysis_flow_L_per_h": 1.0,
        }
        r_no = simulate_peritoneal_pk(**params_no_dialysis)
        r_yes = simulate_peritoneal_pk(**params_dialysis)
        assert r_yes.auc_peri < r_no.auc_peri

    def test_dialysis_reduces_auc_ratio(self):
        r_low = simulate_peritoneal_pk(
            **{**BASE, "application": "pd_dialysis", "dialysis_flow_L_per_h": 0.1}
        )
        r_high = simulate_peritoneal_pk(
            **{**BASE, "application": "pd_dialysis", "dialysis_flow_L_per_h": 2.0}
        )
        assert r_high.auc_ratio_peri_sys < r_low.auc_ratio_peri_sys

    def test_pd_dialysis_note_present(self):
        params = {**BASE, "application": "pd_dialysis", "dialysis_flow_L_per_h": 1.0}
        result = simulate_peritoneal_pk(**params)
        assert any("Peritoneal dialysis" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


class TestNumerics:
    def test_time_array_length(self):
        result = simulate_peritoneal_pk(**BASE)
        assert len(result.times_h) == len(result.c_peri_mg_L)
        assert len(result.times_h) == len(result.c_sys_mg_L)

    def test_all_concentrations_non_negative(self):
        result = simulate_peritoneal_pk(**BASE)
        assert all(c >= 0 for c in result.c_peri_mg_L)
        assert all(c >= 0 for c in result.c_sys_mg_L)

    def test_auc_values_positive(self):
        result = simulate_peritoneal_pk(**BASE)
        assert result.auc_sys > 0
        assert result.auc_peri > 0

    def test_high_k_abs_faster_absorption(self):
        r_slow = simulate_peritoneal_pk(**{**BASE, "k_peritoneal_abs_per_h": 0.05})
        r_fast = simulate_peritoneal_pk(**{**BASE, "k_peritoneal_abs_per_h": 0.5})
        # Faster absorption -> higher systemic Cmax earlier
        assert r_fast.cmax_sys > r_slow.cmax_sys

    def test_larger_dose_higher_conc(self):
        r_low = simulate_peritoneal_pk(**{**BASE, "dose_mg": 50.0})
        r_high = simulate_peritoneal_pk(**{**BASE, "dose_mg": 200.0})
        assert r_high.cmax_sys > r_low.cmax_sys
        assert r_high.auc_sys > r_low.auc_sys


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_peritoneal_pk(**{**BASE, "dose_mg": -10.0})

    def test_invalid_cl_sys(self):
        with pytest.raises(ValueError, match="cl_sys"):
            simulate_peritoneal_pk(**{**BASE, "cl_sys_L_per_h": 0.0})

    def test_invalid_vd_sys(self):
        with pytest.raises(ValueError, match="vd_sys"):
            simulate_peritoneal_pk(**{**BASE, "vd_sys_L": -5.0})

    def test_invalid_peritoneal_volume(self):
        with pytest.raises(ValueError, match="peritoneal_volume"):
            simulate_peritoneal_pk(**{**BASE, "peritoneal_volume_L": 0.0})

    def test_invalid_k_abs(self):
        with pytest.raises(ValueError, match="k_peritoneal_abs"):
            simulate_peritoneal_pk(**{**BASE, "k_peritoneal_abs_per_h": -0.1})

    def test_invalid_dialysis_flow(self):
        with pytest.raises(ValueError, match="dialysis_flow"):
            simulate_peritoneal_pk(**{**BASE, "dialysis_flow_L_per_h": -1.0})

    def test_invalid_application(self):
        with pytest.raises(ValueError, match="application"):
            simulate_peritoneal_pk(**{**BASE, "application": "invalid"})

    def test_invalid_t_end(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_peritoneal_pk(**{**BASE, "t_end_h": 0.0})

    def test_invalid_dt(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_peritoneal_pk(**{**BASE, "dt_h": 0.0})

    def test_dt_larger_than_t_end(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_peritoneal_pk(**{**BASE, "dt_h": 50.0})


# ---------------------------------------------------------------------------
# compare_ip_drugs tests
# ---------------------------------------------------------------------------


class TestCompareIPDrugs:
    def test_returns_list(self):
        result = compare_ip_drugs(
            dose_mg=100.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
            k_abs_values=[
                {"drug": "cisplatin", "k_abs": 0.05},
                {"drug": "paclitaxel", "k_abs": 0.2},
            ],
        )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_result_keys(self):
        result = compare_ip_drugs(
            dose_mg=100.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
            k_abs_values=[{"drug": "drug_a", "k_abs": 0.1}],
        )
        expected_keys = {"drug", "k_abs", "auc_ratio", "auc_sys", "auc_peri"}
        assert set(result[0].keys()) == expected_keys

    def test_sorted_by_auc_ratio_descending(self):
        result = compare_ip_drugs(
            dose_mg=100.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
            k_abs_values=[
                {"drug": "fast", "k_abs": 0.5},
                {"drug": "slow", "k_abs": 0.05},
            ],
        )
        ratios = [r["auc_ratio"] for r in result if math.isfinite(r["auc_ratio"])]
        assert ratios == sorted(ratios, reverse=True)

    def test_lower_k_abs_higher_ratio(self):
        result = compare_ip_drugs(
            dose_mg=100.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
            k_abs_values=[
                {"drug": "slow", "k_abs": 0.02},
                {"drug": "fast", "k_abs": 0.5},
            ],
        )
        # Slow absorption -> more time in peritoneal -> higher ratio
        slow = next(r for r in result if r["drug"] == "slow")
        fast = next(r for r in result if r["drug"] == "fast")
        assert slow["auc_ratio"] > fast["auc_ratio"]

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            compare_ip_drugs(
                dose_mg=-1.0,
                cl_sys_L_per_h=5.0,
                vd_sys_L=50.0,
                k_abs_values=[{"drug": "a", "k_abs": 0.1}],
            )

    def test_empty_k_abs_raises(self):
        with pytest.raises(ValueError, match="k_abs_values"):
            compare_ip_drugs(
                dose_mg=100.0,
                cl_sys_L_per_h=5.0,
                vd_sys_L=50.0,
                k_abs_values=[],
            )

    def test_negative_k_abs_raises(self):
        with pytest.raises(ValueError, match="k_abs"):
            compare_ip_drugs(
                dose_mg=100.0,
                cl_sys_L_per_h=5.0,
                vd_sys_L=50.0,
                k_abs_values=[{"drug": "a", "k_abs": -0.1}],
            )
