"""Tests for subcutaneous depot PK model (Phase 776)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.subcutaneous_depot_pk import (
    SCDepotPKResult,
    estimate_repeat_dose_accumulation,
    simulate_sc_depot_pk,
)

# ---------------------------------------------------------------------------
# Return type and field types
# ---------------------------------------------------------------------------


def test_returns_sc_depot_pk_result():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert isinstance(r, SCDepotPKResult)


def test_drug_name_preserved():
    r = simulate_sc_depot_pk("mymab", dose_mg=200.0)
    assert r.drug_name == "mymab"


def test_dose_preserved():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=150.0)
    assert r.dose_mg == 150.0


def test_f_sc_preserved():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0, f_sc=0.80)
    assert r.f_sc == pytest.approx(0.80)


def test_times_h_is_list():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert isinstance(r.times_h, list)


def test_a_depot_is_list():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert isinstance(r.a_depot_mg, list)


def test_c_central_is_list():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert isinstance(r.c_central_mg_L, list)


def test_c_peripheral_is_list():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert isinstance(r.c_peripheral_mg_L, list)


# ---------------------------------------------------------------------------
# PK metric validity
# ---------------------------------------------------------------------------


def test_cmax_central_positive():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert r.cmax_central > 0.0


def test_auc_central_positive():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert r.auc_central > 0.0


def test_t_half_terminal_positive():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert r.t_half_terminal_h > 0.0


def test_accumulation_ratio_gte_one():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert r.accumulation_ratio >= 1.0


def test_notes_nonempty():
    r = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    r1 = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    r2 = simulate_sc_depot_pk("biologic_a", dose_mg=200.0)
    assert r2.cmax_central == pytest.approx(2.0 * r1.cmax_central, rel=1e-4)


def test_double_dose_doubles_auc():
    r1 = simulate_sc_depot_pk("biologic_a", dose_mg=100.0)
    r2 = simulate_sc_depot_pk("biologic_a", dose_mg=200.0)
    assert r2.auc_central == pytest.approx(2.0 * r1.auc_central, rel=1e-4)


# ---------------------------------------------------------------------------
# Bioavailability effect
# ---------------------------------------------------------------------------


def test_higher_f_sc_higher_auc():
    r_low = simulate_sc_depot_pk("biologic_a", dose_mg=100.0, f_sc=0.50)
    r_high = simulate_sc_depot_pk("biologic_a", dose_mg=100.0, f_sc=0.90)
    assert r_high.auc_central > r_low.auc_central


def test_higher_f_sc_higher_cmax():
    r_low = simulate_sc_depot_pk("biologic_a", dose_mg=100.0, f_sc=0.40)
    r_high = simulate_sc_depot_pk("biologic_a", dose_mg=100.0, f_sc=0.90)
    assert r_high.cmax_central > r_low.cmax_central


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sc_depot_pk("biologic_a", dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sc_depot_pk("biologic_a", dose_mg=-10.0)


def test_ka_zero_raises():
    with pytest.raises(ValueError, match="ka_sc"):
        simulate_sc_depot_pk("biologic_a", dose_mg=100.0, ka_sc_per_h=0.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_sc_depot_pk("biologic_a", dose_mg=100.0, cl_L_per_h=0.0)


def test_vd_central_zero_raises():
    with pytest.raises(ValueError, match="vd_central_L"):
        simulate_sc_depot_pk("biologic_a", dose_mg=100.0, vd_central_L=0.0)


def test_vd_peripheral_zero_raises():
    with pytest.raises(ValueError, match="vd_peripheral_L"):
        simulate_sc_depot_pk("biologic_a", dose_mg=100.0, vd_peripheral_L=0.0)


def test_f_sc_zero_raises():
    with pytest.raises(ValueError, match="f_sc"):
        simulate_sc_depot_pk("biologic_a", dose_mg=100.0, f_sc=0.0)


def test_f_sc_gt_one_raises():
    with pytest.raises(ValueError, match="f_sc"):
        simulate_sc_depot_pk("biologic_a", dose_mg=100.0, f_sc=1.5)


# ---------------------------------------------------------------------------
# estimate_repeat_dose_accumulation
# ---------------------------------------------------------------------------


def test_repeat_dose_returns_dict():
    out = estimate_repeat_dose_accumulation(
        "biologic_a", dose_mg=100.0, tau_h=168.0, t_half_h=120.0
    )
    assert isinstance(out, dict)


def test_repeat_dose_expected_keys():
    out = estimate_repeat_dose_accumulation(
        "biologic_a", dose_mg=100.0, tau_h=168.0, t_half_h=120.0
    )
    for key in (
        "drug_name",
        "dose_mg",
        "tau_h",
        "t_half_h",
        "accumulation_ratio",
        "css_max_estimate_mg_L",
        "notes",
    ):
        assert key in out, f"Missing key: {key}"


def test_repeat_dose_accumulation_ratio_gte_one():
    out = estimate_repeat_dose_accumulation(
        "biologic_a", dose_mg=100.0, tau_h=168.0, t_half_h=120.0
    )
    assert out["accumulation_ratio"] >= 1.0


def test_repeat_dose_accumulation_ratio_lte_ten():
    # Very short tau relative to half-life -> ratio clamped at 10
    out = estimate_repeat_dose_accumulation("biologic_a", dose_mg=100.0, tau_h=1.0, t_half_h=1000.0)
    assert out["accumulation_ratio"] <= 10.0


def test_repeat_dose_higher_tau_lower_accumulation():
    out_short = estimate_repeat_dose_accumulation(
        "biologic_a", dose_mg=100.0, tau_h=24.0, t_half_h=100.0
    )
    out_long = estimate_repeat_dose_accumulation(
        "biologic_a", dose_mg=100.0, tau_h=200.0, t_half_h=100.0
    )
    assert out_short["accumulation_ratio"] >= out_long["accumulation_ratio"]


def test_repeat_dose_tau_zero_raises():
    with pytest.raises(ValueError, match="tau_h"):
        estimate_repeat_dose_accumulation("biologic_a", dose_mg=100.0, tau_h=0.0, t_half_h=100.0)


def test_repeat_dose_t_half_zero_raises():
    with pytest.raises(ValueError, match="t_half_h"):
        estimate_repeat_dose_accumulation("biologic_a", dose_mg=100.0, tau_h=168.0, t_half_h=0.0)
