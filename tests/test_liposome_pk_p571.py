"""Tests for Phase 571 — Liposome Drug Encapsulation PK."""

import pytest

from omega_pbpk.core.liposome_pk_p571 import LiposomePKResult, simulate_liposome_pk

# Common parameters
BASE_PARAMS = dict(
    drug_name="Doxorubicin",
    dose_mg=50.0,
    pegylated=False,
    cl_liposome_L_per_h=0.5,
    cl_free_L_per_h=2.0,
    vd_liposome_L=3.0,
    vd_free_L=30.0,
    k_release_per_h=0.05,
    t_end_h=168.0,
    dt_h=0.5,
)


def _run(**overrides):
    params = {**BASE_PARAMS, **overrides}
    return simulate_liposome_pk(**params)


class TestReturnType:
    def test_returns_liposome_pk_result(self):
        r = _run()
        assert isinstance(r, LiposomePKResult)

    def test_list_lengths_consistent(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_encapsulated_mg_L) == n
        assert len(r.c_free_mg_L) == n
        assert len(r.c_tissue_mg_L) == n

    def test_expected_number_of_points(self):
        r = _run(t_end_h=10.0, dt_h=0.5)
        assert len(r.times_h) == 21  # 0..10 step 0.5

    def test_drug_name_stored(self):
        r = _run()
        assert r.drug_name == "Doxorubicin"

    def test_dose_stored(self):
        r = _run()
        assert r.dose_mg == 50.0


class TestPEGylation:
    def test_pegylated_longer_half_life(self):
        r_peg = _run(pegylated=True)
        r_nopeg = _run(pegylated=False)
        assert r_peg.t_half_liposome_h > r_nopeg.t_half_liposome_h

    def test_pegylated_flag_stored(self):
        assert _run(pegylated=True).pegylated is True
        assert _run(pegylated=False).pegylated is False

    def test_pegylated_4x_half_life(self):
        r_peg = _run(pegylated=True)
        r_nopeg = _run(pegylated=False)
        assert abs(r_peg.t_half_liposome_h / r_nopeg.t_half_liposome_h - 4.0) < 0.01


class TestConcentrationProfiles:
    def test_c_free_starts_near_zero(self):
        r = _run()
        assert r.c_free_mg_L[0] == 0.0

    def test_c_free_rises_then_falls(self):
        r = _run()
        # Should rise from 0 to cmax, then decline
        assert r.cmax_free_mg_L > 0
        assert r.tmax_free_h > 0
        # After tmax, concentration should eventually be lower
        assert r.c_free_mg_L[-1] < r.cmax_free_mg_L

    def test_encapsulated_decreases(self):
        r = _run()
        assert r.c_encapsulated_mg_L[0] > r.c_encapsulated_mg_L[-1]

    def test_tissue_starts_at_zero(self):
        r = _run()
        assert r.c_tissue_mg_L[0] == 0.0


class TestAUC:
    def test_auc_encapsulated_positive(self):
        r = _run()
        assert r.auc_encapsulated_mg_h_per_L > 0

    def test_auc_free_positive(self):
        r = _run()
        assert r.auc_free_mg_h_per_L > 0


class TestDrugRelease:
    def test_drug_release_time_positive(self):
        r = _run()
        assert r.drug_release_time_h > 0

    def test_higher_k_release_shorter_release_time(self):
        # Use low MPS clearance so release dominates
        r_slow = _run(k_release_per_h=0.02, cl_liposome_L_per_h=0.01, t_end_h=300.0)
        r_fast = _run(k_release_per_h=0.1, cl_liposome_L_per_h=0.01, t_end_h=300.0)
        assert r_fast.drug_release_time_h < r_slow.drug_release_time_h


class TestDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=50.0)
        r2 = _run(dose_mg=100.0)
        ratio = r2.cmax_free_mg_L / r1.cmax_free_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=50.0)
        r2 = _run(dose_mg=100.0)
        ratio = r2.auc_free_mg_h_per_L / r1.auc_free_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


class TestValidation:
    def test_dose_positive(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_cl_liposome_positive(self):
        with pytest.raises(ValueError):
            _run(cl_liposome_L_per_h=-1)

    def test_cl_free_positive(self):
        with pytest.raises(ValueError):
            _run(cl_free_L_per_h=0)

    def test_vd_liposome_positive(self):
        with pytest.raises(ValueError):
            _run(vd_liposome_L=-5)

    def test_vd_free_positive(self):
        with pytest.raises(ValueError):
            _run(vd_free_L=0)

    def test_k_release_positive(self):
        with pytest.raises(ValueError):
            _run(k_release_per_h=-0.01)
