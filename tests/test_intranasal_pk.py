"""Tests for intranasal drug delivery PK module."""

from __future__ import annotations

import pytest

from omega_pbpk.core.intranasal_pk import (
    IntranasalPKResult,
    compare_delivery_routes,
    simulate_intranasal_pk,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _default_result(**kwargs) -> IntranasalPKResult:
    params = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        ka_systemic_per_h=2.0,
        ka_cns_per_h=0.5,
        k_mcc_per_h=1.0,
        cl_systemic_L_per_h=10.0,
        vd_systemic_L=50.0,
        v_cns_L=1.5,
        cl_cns_per_h=0.1,
        t_end_h=12.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_intranasal_pk(**params)


# ── Validation errors ────────────────────────────────────────────────────────


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=-5.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_systemic"):
            _default_result(cl_systemic_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_systemic"):
            _default_result(vd_systemic_L=0.0)

    def test_ka_systemic_zero_raises(self):
        with pytest.raises(ValueError, match="ka_systemic"):
            _default_result(ka_systemic_per_h=0.0)

    def test_ka_cns_negative_raises(self):
        with pytest.raises(ValueError, match="ka_cns"):
            _default_result(ka_cns_per_h=-0.1)

    def test_k_mcc_zero_raises(self):
        with pytest.raises(ValueError, match="k_mcc"):
            _default_result(k_mcc_per_h=0.0)


# ── Absorption behaviour ─────────────────────────────────────────────────────


class TestAbsorptionBehaviour:
    def test_cmax_systemic_positive(self):
        r = _default_result()
        assert r.cmax_systemic > 0.0

    def test_f_absorbed_systemic_in_range(self):
        r = _default_result()
        assert 0.0 <= r.f_absorbed_systemic <= 1.0

    def test_f_absorbed_cns_in_range(self):
        r = _default_result()
        assert 0.0 <= r.f_absorbed_cns <= 1.0

    def test_f_absorbed_sum_leq_one(self):
        r = _default_result()
        assert r.f_absorbed_systemic + r.f_absorbed_cns <= 1.0 + 1e-9

    def test_high_mcc_reduces_f_absorbed_systemic(self):
        low_mcc = _default_result(k_mcc_per_h=0.1)
        high_mcc = _default_result(k_mcc_per_h=20.0)
        assert high_mcc.f_absorbed_systemic < low_mcc.f_absorbed_systemic

    def test_lower_mcc_gives_higher_f_absorbed_systemic(self):
        r_low = _default_result(k_mcc_per_h=0.2)
        r_high = _default_result(k_mcc_per_h=5.0)
        assert r_low.f_absorbed_systemic > r_high.f_absorbed_systemic

    def test_ka_cns_zero_cns_stays_zero(self):
        r = _default_result(ka_cns_per_h=0.0)
        assert max(r.c_cns_mg_L) == pytest.approx(0.0, abs=1e-10)

    def test_ka_cns_zero_auc_cns_zero(self):
        r = _default_result(ka_cns_per_h=0.0)
        assert r.auc_cns == pytest.approx(0.0, abs=1e-10)

    def test_nose_to_brain_ratio_positive_when_ka_cns_positive(self):
        r = _default_result(ka_cns_per_h=0.5)
        assert r.nose_to_brain_ratio > 0.0

    def test_nose_to_brain_ratio_zero_when_ka_cns_zero(self):
        r = _default_result(ka_cns_per_h=0.0)
        assert r.nose_to_brain_ratio == pytest.approx(0.0, abs=1e-10)

    def test_faster_mcc_lowers_systemic_auc(self):
        r_slow = _default_result(k_mcc_per_h=0.5)
        r_fast = _default_result(k_mcc_per_h=10.0)
        assert r_fast.auc_systemic < r_slow.auc_systemic


# ── Result field types ────────────────────────────────────────────────────────


class TestResultFieldTypes:
    def test_result_is_dataclass(self):
        r = _default_result()
        assert isinstance(r, IntranasalPKResult)

    def test_times_is_list(self):
        r = _default_result()
        assert isinstance(r.times_h, list)

    def test_m_nasal_is_list(self):
        r = _default_result()
        assert isinstance(r.m_nasal_mg, list)

    def test_c_systemic_is_list(self):
        r = _default_result()
        assert isinstance(r.c_systemic_mg_L, list)

    def test_c_cns_is_list(self):
        r = _default_result()
        assert isinstance(r.c_cns_mg_L, list)

    def test_cmax_is_float(self):
        r = _default_result()
        assert isinstance(r.cmax_systemic, float)

    def test_tmax_is_float(self):
        r = _default_result()
        assert isinstance(r.tmax_systemic_h, float)

    def test_notes_is_list(self):
        r = _default_result()
        assert isinstance(r.notes, list)

    def test_nasal_mass_starts_at_dose(self):
        r = _default_result(dose_mg=10.0)
        assert r.m_nasal_mg[0] == pytest.approx(10.0, rel=1e-6)

    def test_nasal_mass_declines_over_time(self):
        r = _default_result()
        assert r.m_nasal_mg[-1] < r.m_nasal_mg[0]


# ── compare_delivery_routes ───────────────────────────────────────────────────


class TestCompareDeliveryRoutes:
    def test_returns_dict_with_expected_keys(self):
        result = compare_delivery_routes(
            drug_name="Drug",
            dose_mg=10.0,
            cl_systemic_L_per_h=10.0,
            vd_systemic_L=50.0,
        )
        assert "intranasal" in result
        assert "reference_label" in result
        assert "reference_auc" in result
        assert "reference_cmax" in result
        assert "bioavailability_ratio" in result

    def test_iv_reference_label(self):
        result = compare_delivery_routes(
            drug_name="Drug",
            dose_mg=10.0,
            cl_systemic_L_per_h=10.0,
            vd_systemic_L=50.0,
        )
        assert result["reference_label"] == "iv"

    def test_oral_reference_label(self):
        result = compare_delivery_routes(
            drug_name="Drug",
            dose_mg=10.0,
            cl_systemic_L_per_h=10.0,
            vd_systemic_L=50.0,
            ka_iv_equivalent=1.0,
        )
        assert result["reference_label"] == "oral"

    def test_bioavailability_ratio_nonnegative(self):
        result = compare_delivery_routes(
            drug_name="Drug",
            dose_mg=10.0,
            cl_systemic_L_per_h=10.0,
            vd_systemic_L=50.0,
        )
        assert result["bioavailability_ratio"] >= 0.0

    def test_intranasal_result_type(self):
        result = compare_delivery_routes(
            drug_name="Drug",
            dose_mg=10.0,
            cl_systemic_L_per_h=10.0,
            vd_systemic_L=50.0,
        )
        assert isinstance(result["intranasal"], IntranasalPKResult)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            compare_delivery_routes(
                drug_name="Drug",
                dose_mg=0.0,
                cl_systemic_L_per_h=10.0,
                vd_systemic_L=50.0,
            )
