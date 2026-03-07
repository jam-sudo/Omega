"""Tests for Phase 812: bispecific antibody PK model."""

import pytest

from omega_pbpk.prediction.bispecific_antibody_pk import (
    BispecificAntibodyPKResult,
    simulate_bispecific_antibody_pk,
)


def default_sim(**kwargs):
    defaults = dict(
        drug_name="BsAb001",
        dose_mg_per_kg=1.0,
    )
    defaults.update(kwargs)
    return simulate_bispecific_antibody_pk(**defaults)


# --- Return type ---


def test_returns_bispecific_result():
    r = default_sim()
    assert isinstance(r, BispecificAntibodyPKResult)


def test_list_fields_are_lists():
    r = default_sim()
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_antibody_mg_L, list)
    assert isinstance(r.c_target1_nM, list)
    assert isinstance(r.c_target2_nM, list)
    assert isinstance(r.c_complex1_nM, list)
    assert isinstance(r.c_complex2_nM, list)


def test_list_lengths_match():
    r = default_sim()
    n = len(r.times_h)
    assert len(r.c_antibody_mg_L) == n
    assert len(r.c_target1_nM) == n
    assert len(r.c_target2_nM) == n
    assert len(r.c_complex1_nM) == n
    assert len(r.c_complex2_nM) == n


# --- cmax > 0, auc > 0 ---


def test_cmax_positive():
    r = default_sim()
    assert r.cmax_antibody > 0.0


def test_auc_positive():
    r = default_sim()
    assert r.auc_antibody > 0.0


def test_dose_mg_equals_dose_per_kg_times_weight():
    r = simulate_bispecific_antibody_pk("Ab", dose_mg_per_kg=2.0, body_weight_kg=70.0)
    assert abs(r.dose_mg - 140.0) < 1e-6


def test_drug_name_preserved():
    r = simulate_bispecific_antibody_pk("MyBsAb", dose_mg_per_kg=1.0)
    assert r.drug_name == "MyBsAb"


# --- Both targets partially engaged at Cmax ---


def test_target1_engaged_at_cmax():
    r = simulate_bispecific_antibody_pk(
        "BsAb",
        dose_mg_per_kg=5.0,
        target1_baseline_nM=10.0,
        kon1_per_nM_per_day=0.5,
        koff1_per_day=0.01,
        t_end_days=7.0,
    )
    assert r.target1_engagement_pct > 0.0


def test_target2_engaged_at_cmax():
    r = simulate_bispecific_antibody_pk(
        "BsAb",
        dose_mg_per_kg=5.0,
        target2_baseline_nM=5.0,
        kon2_per_nM_per_day=0.5,
        koff2_per_day=0.01,
        t_end_days=7.0,
    )
    assert r.target2_engagement_pct > 0.0


def test_engagement_between_0_and_100():
    r = default_sim(dose_mg_per_kg=3.0)
    assert 0.0 <= r.target1_engagement_pct <= 100.0
    assert 0.0 <= r.target2_engagement_pct <= 100.0


# --- SC has later Tmax than IV ---


def test_sc_tmax_later_than_iv():
    iv_result = simulate_bispecific_antibody_pk(
        "BsAb", dose_mg_per_kg=1.0, route="iv", t_end_days=14.0, dt_days=0.05
    )
    sc_result = simulate_bispecific_antibody_pk(
        "BsAb", dose_mg_per_kg=1.0, route="sc", t_end_days=14.0, dt_days=0.05
    )
    assert sc_result.tmax_antibody_h > iv_result.tmax_antibody_h


# --- Higher dose → higher engagement ---


def test_higher_dose_higher_target1_engagement():
    r_low = simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=0.1, t_end_days=7.0)
    r_high = simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=10.0, t_end_days=7.0)
    assert r_high.target1_engagement_pct >= r_low.target1_engagement_pct


def test_higher_dose_higher_cmax():
    r_low = simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=1.0)
    r_high = simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=5.0)
    assert r_high.cmax_antibody > r_low.cmax_antibody


def test_higher_dose_higher_auc():
    r_low = simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=1.0)
    r_high = simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=5.0)
    assert r_high.auc_antibody > r_low.auc_antibody


# --- Validation ---


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=-1.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError):
        simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=1.0, route="oral")


def test_zero_body_weight_raises():
    with pytest.raises(ValueError):
        simulate_bispecific_antibody_pk("BsAb", dose_mg_per_kg=1.0, body_weight_kg=0.0)


# --- Additional checks ---


def test_times_start_at_zero():
    r = default_sim()
    assert r.times_h[0] == 0.0


def test_antibody_conc_nonnegative():
    r = default_sim()
    assert all(c >= 0.0 for c in r.c_antibody_mg_L)


def test_target_conc_nonnegative():
    r = default_sim()
    assert all(t >= 0.0 for t in r.c_target1_nM)
    assert all(t >= 0.0 for t in r.c_target2_nM)


def test_notes_is_string():
    r = default_sim()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0
