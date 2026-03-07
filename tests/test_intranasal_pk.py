"""Tests for intranasal drug delivery PK module."""

from __future__ import annotations

import pytest

from omega_pbpk.core.intranasal_pk import (
    IntranasalPKResult,
    compare_delivery_routes,
    compare_nasal_formulations,
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


# ── compare_nasal_formulations ────────────────────────────────────────────────


_FORMULATIONS = [
    {"name": "solution", "f_nasal": 0.9, "k_nasal": 2.0, "mcc": 0.5},
    {"name": "gel", "f_nasal": 0.7, "k_nasal": 1.0, "mcc": 0.3},
    {"name": "powder", "f_nasal": 0.5, "k_nasal": 1.5, "mcc": 1.0},
]


def _compare(**kwargs) -> list[dict]:
    params = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        cl_systemic_L_per_h=10.0,
        vd_systemic_L=50.0,
        formulations=_FORMULATIONS,
    )
    params.update(kwargs)
    return compare_nasal_formulations(**params)


class TestCompareNasalFormulations:
    def test_returns_list(self):
        result = _compare()
        assert isinstance(result, list)

    def test_length_matches_formulations(self):
        result = _compare()
        assert len(result) == len(_FORMULATIONS)

    def test_sorted_by_auc_descending(self):
        result = _compare()
        aucs = [d["auc"] for d in result]
        assert aucs == sorted(aucs, reverse=True)

    def test_each_entry_has_name(self):
        result = _compare()
        for d in result:
            assert "name" in d

    def test_each_entry_has_auc(self):
        result = _compare()
        for d in result:
            assert "auc" in d

    def test_each_entry_has_cmax(self):
        result = _compare()
        for d in result:
            assert "cmax" in d

    def test_each_entry_has_tmax_h(self):
        result = _compare()
        for d in result:
            assert "tmax_h" in d

    def test_each_entry_has_f_absorbed(self):
        result = _compare()
        for d in result:
            assert "f_absorbed" in d

    def test_each_entry_has_result_object(self):
        result = _compare()
        for d in result:
            assert isinstance(d["result"], IntranasalPKResult)

    def test_auc_positive(self):
        result = _compare()
        for d in result:
            assert d["auc"] >= 0.0

    def test_cmax_positive(self):
        result = _compare()
        for d in result:
            assert d["cmax"] >= 0.0

    def test_f_absorbed_in_range(self):
        result = _compare()
        for d in result:
            assert 0.0 <= d["f_absorbed"] <= 1.0 + 1e-9

    def test_high_f_nasal_gives_higher_auc(self):
        high_f = [{"name": "A", "f_nasal": 0.95, "k_nasal": 2.0, "mcc": 0.5}]
        low_f = [{"name": "B", "f_nasal": 0.30, "k_nasal": 2.0, "mcc": 0.5}]
        r_high = compare_nasal_formulations(
            "D", 10.0, 10.0, 50.0, formulations=high_f
        )
        r_low = compare_nasal_formulations(
            "D", 10.0, 10.0, 50.0, formulations=low_f
        )
        assert r_high[0]["auc"] > r_low[0]["auc"]

    def test_single_formulation(self):
        forms = [{"name": "only", "f_nasal": 0.8, "k_nasal": 1.5, "mcc": 0.5}]
        result = compare_nasal_formulations("D", 10.0, 10.0, 50.0, formulations=forms)
        assert len(result) == 1
        assert result[0]["name"] == "only"

    def test_all_names_preserved(self):
        result = _compare()
        names_out = {d["name"] for d in result}
        names_in = {f["name"] for f in _FORMULATIONS}
        assert names_out == names_in

    def test_f_nasal_preserved_in_result(self):
        result = _compare()
        for d in result:
            assert "f_nasal" in d

    def test_k_nasal_preserved_in_result(self):
        result = _compare()
        for d in result:
            assert "k_nasal" in d

    def test_mcc_preserved_in_result(self):
        result = _compare()
        for d in result:
            assert "mcc" in d

    def test_empty_formulations_raises(self):
        with pytest.raises(ValueError):
            compare_nasal_formulations("D", 10.0, 10.0, 50.0, formulations=[])

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            compare_nasal_formulations("D", 0.0, 10.0, 50.0, formulations=_FORMULATIONS)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_systemic"):
            compare_nasal_formulations("D", 10.0, 0.0, 50.0, formulations=_FORMULATIONS)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_systemic"):
            compare_nasal_formulations("D", 10.0, 10.0, 0.0, formulations=_FORMULATIONS)

    def test_missing_key_raises(self):
        bad = [{"name": "X", "f_nasal": 0.8, "k_nasal": 1.0}]  # missing 'mcc'
        with pytest.raises(ValueError):
            compare_nasal_formulations("D", 10.0, 10.0, 50.0, formulations=bad)

    def test_invalid_f_nasal_raises(self):
        bad = [{"name": "X", "f_nasal": 1.5, "k_nasal": 1.0, "mcc": 0.5}]
        with pytest.raises(ValueError):
            compare_nasal_formulations("D", 10.0, 10.0, 50.0, formulations=bad)

    def test_negative_mcc_raises(self):
        bad = [{"name": "X", "f_nasal": 0.8, "k_nasal": 1.0, "mcc": -0.1}]
        with pytest.raises(ValueError):
            compare_nasal_formulations("D", 10.0, 10.0, 50.0, formulations=bad)
