"""Tests for TMDD QSS model (Phase 111)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.target_mediated import TMDDResult, simulate_tmdd_qss


def _default(**kw):
    defaults = dict(
        drug_name="mAb",
        dose_mg=10.0,
        vd_L=5.0,
        cl_L_per_h=0.01,
        ksyn_nmol_L_h=0.1,
        kdeg_per_h=0.02,
        t_end_h=168.0,
        dt_h=1.0,
    )
    defaults.update(kw)
    return simulate_tmdd_qss(**defaults)


class TestTMDDResult:
    def test_returns_result_type(self):
        assert isinstance(_default(), TMDDResult)

    def test_free_drug_positive_at_t0(self):
        r = _default()
        assert r.free_drug_mg_L[0] > 0.0

    def test_free_drug_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.free_drug_mg_L)

    def test_auc_free_positive(self):
        r = _default()
        assert r.auc_free > 0.0

    def test_cmax_positive(self):
        r = _default()
        assert r.cmax_free > 0.0

    def test_ro_max_between_0_and_1(self):
        r = _default()
        assert 0.0 <= r.receptor_occupancy_max <= 1.0

    def test_kd_equals_koff_over_kon(self):
        r = _default(kon_per_nmol_per_h=0.1, koff_per_h=0.01)
        assert abs(r.kd_nmol_L - 0.1) < 1e-6

    def test_times_and_conc_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.free_drug_mg_L) == len(r.free_target_nmol_L)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            _default(vd_L=0.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            _default(cl_L_per_h=0.0)

    def test_total_target_positive(self):
        r = _default()
        assert r.total_target_nmol_L > 0.0

    def test_oral_route_tmax_after_t0(self):
        r = simulate_tmdd_qss(
            "Drug", 10.0, 5.0, 0.01, 0.1, 0.02, route="oral", ka_per_h=0.5, t_end_h=168.0, dt_h=1.0
        )
        assert r.tmax_h > 0.0
