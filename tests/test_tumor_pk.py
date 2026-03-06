"""Tests for tumor PK/PD model."""

import pytest

from omega_pbpk.clinical.tumor_pk import (
    TumorPKResult,
    dose_response_tumor,
    simulate_tumor_pk,
)

# ---- Shared helper kwargs ----
_BASE = dict(drug_name="Doxorubicin", dose_mg=50.0, cl_L_per_h=30.0, vd_L=500.0)


# ---- Input validation ----


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_pk(**{**_BASE, "dose_mg": -1.0})


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_pk(**{**_BASE, "dose_mg": 0.0})


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_tumor_pk(**{**_BASE, "cl_L_per_h": 0.0})


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_tumor_pk(**{**_BASE, "vd_L": 0.0})


def test_invalid_pd_model_raises():
    with pytest.raises(ValueError, match="pd_model"):
        simulate_tumor_pk(**_BASE, pd_model="invalid")


# ---- Basic simulation ----


def test_basic_simulation_returns_result():
    r = simulate_tumor_pk(**_BASE, t_end_h=24.0)
    assert isinstance(r, TumorPKResult)
    assert r.cmax_sys > 0
    assert r.auc_sys > 0


def test_times_array_length():
    r = simulate_tumor_pk(**_BASE, t_end_h=10.0, dt_h=0.1)
    n_steps = max(int(10.0 / 0.1), 1)
    assert len(r.times_h) == n_steps + 1


# ---- Drug reaches tumor ----


def test_drug_reaches_tumor_vascular():
    r = simulate_tumor_pk(**_BASE, t_end_h=48.0)
    assert r.cmax_tumor_vasc > 0


def test_avascular_lower_than_vascular():
    r = simulate_tumor_pk(**_BASE, t_end_h=48.0, drug_permeability=0.5)
    assert r.cmax_tumor_avsc < r.cmax_tumor_vasc


def test_high_permeability_avascular_approaches_vascular():
    r = simulate_tumor_pk(
        **_BASE,
        t_end_h=168.0,
        drug_permeability=1.0,
        k_penetration_per_h=10.0,
        tumor_blood_flow_L_per_h=0.1,
    )
    # With very high penetration, avascular should approach vascular
    assert r.cmax_tumor_avsc > 0.3 * r.cmax_tumor_vasc


def test_no_drug_penetration():
    r = simulate_tumor_pk(**_BASE, t_end_h=48.0, drug_permeability=0.0)
    assert r.cmax_tumor_avsc < 1e-10


# ---- PD: cell kill ----


def test_tumor_cell_kill_occurs():
    r = simulate_tumor_pk(**_BASE, t_end_h=168.0)
    assert r.tumor_cells_norm[-1] < 1.0


def test_cell_kill_scales_with_dose():
    r1 = simulate_tumor_pk(**{**_BASE, "dose_mg": 50.0}, t_end_h=168.0)
    r2 = simulate_tumor_pk(**{**_BASE, "dose_mg": 100.0}, t_end_h=168.0)
    assert r2.max_log_kill > r1.max_log_kill


def test_emax_saturates():
    r_mid = simulate_tumor_pk(**{**_BASE, "dose_mg": 500.0}, pd_model="emax", t_end_h=48.0)
    r_high = simulate_tumor_pk(**{**_BASE, "dose_mg": 5000.0}, pd_model="emax", t_end_h=48.0)
    # Emax should saturate — 10x dose should NOT give 10x more kill
    ratio = r_high.max_log_kill / r_mid.max_log_kill if r_mid.max_log_kill > 0 else 999
    assert ratio < 5.0


def test_log_kill_model():
    r = simulate_tumor_pk(**_BASE, pd_model="log_kill", t_end_h=168.0)
    assert r.max_log_kill > 0
    assert r.pd_model == "log_kill"


def test_direct_model():
    r = simulate_tumor_pk(**_BASE, pd_model="direct", t_end_h=168.0)
    assert r.max_log_kill > 0
    assert r.pd_model == "direct"


def test_subeffective_dose_cells_grow():
    r = simulate_tumor_pk(
        **{**_BASE, "dose_mg": 0.001},
        t_end_h=168.0,
        emax=0.001,
        ec50_mg_L=100.0,
    )
    # Trace dose, high EC50, low emax → growth dominates
    assert r.tumor_cells_norm[-1] > 1.0


def test_tumor_eradication():
    r = simulate_tumor_pk(
        **{**_BASE, "dose_mg": 5000.0},
        t_end_h=168.0,
        emax=5.0,
        ec50_mg_L=0.1,
    )
    assert r.tumor_eradication is True


def test_max_log_kill_positive():
    r = simulate_tumor_pk(**_BASE, t_end_h=48.0)
    assert r.max_log_kill > 0


# ---- dose_response_tumor ----


def test_dose_response_returns_correct_length():
    doses = [10.0, 50.0, 100.0]
    results = dose_response_tumor("Drug", doses, cl_L_per_h=30.0, vd_L=500.0, t_end_h=24.0)
    assert len(results) == len(doses)


def test_dose_response_higher_dose_more_kill():
    doses = [10.0, 50.0, 200.0]
    results = dose_response_tumor("Drug", doses, cl_L_per_h=30.0, vd_L=500.0, t_end_h=168.0)
    kills = [r.max_log_kill for r in results]
    assert kills == sorted(kills)


# ---- Oral route ----


def test_oral_delayed_peak():
    r_iv = simulate_tumor_pk(**_BASE, route="iv", t_end_h=24.0, dt_h=0.05)
    r_oral = simulate_tumor_pk(**_BASE, route="oral", ka_per_h=1.0, t_end_h=24.0, dt_h=0.05)
    tmax_iv = r_iv.times_h[r_iv.c_sys_mg_L.index(max(r_iv.c_sys_mg_L))]
    tmax_oral = r_oral.times_h[r_oral.c_sys_mg_L.index(max(r_oral.c_sys_mg_L))]
    assert tmax_oral > tmax_iv


def test_oral_f_zero_no_exposure():
    r = simulate_tumor_pk(**_BASE, route="oral", f_oral=0.0, t_end_h=24.0)
    assert r.cmax_sys < 1e-10


# ---- Notes ----


def test_notes_contain_drug_name():
    r = simulate_tumor_pk(**_BASE, t_end_h=24.0)
    assert _BASE["drug_name"] in r.notes
