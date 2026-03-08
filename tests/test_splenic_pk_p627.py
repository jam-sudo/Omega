"""Tests for Phase 627 — Drug Splenic Distribution."""

import math

import pytest

from omega_pbpk.core.splenic_pk_p627 import SplenicPKResult, simulate_splenic_pk

# ── helpers ──────────────────────────────────────────────────────────────────


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.0,
        mw_Da=400.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    params.update(kwargs)
    return simulate_splenic_pk(**params)


# ── return type and structure ─────────────────────────────────────────────────


def test_returns_splenic_pk_result():
    result = default_result()
    assert isinstance(result, SplenicPKResult)


def test_result_is_plain_dataclass_not_frozen():
    result = default_result()
    # plain @dataclass allows mutation
    result.drug_name = "Modified"
    assert result.drug_name == "Modified"


def test_list_fields_are_lists():
    result = default_result()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.c_spleen_mg_g, list)


def test_list_lengths_consistent():
    result = default_result(t_end_h=24.0, dt_h=0.1)
    n = len(result.times_h)
    assert n == len(result.c_plasma_mg_L)
    assert n == len(result.c_spleen_mg_g)


def test_list_length_matches_steps():
    result = default_result(t_end_h=10.0, dt_h=0.1)
    # n_steps = ceil(10/0.1) = 100 → 101 points (0..100 inclusive)
    assert len(result.times_h) == 101


def test_times_start_at_zero():
    result = default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    result = default_result(t_end_h=48.0, dt_h=0.1)
    assert result.times_h[-1] == pytest.approx(48.0, abs=0.11)


def test_concentrations_nonnegative():
    result = default_result()
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)
    assert all(c >= 0.0 for c in result.c_spleen_mg_g)


# ── nanoparticle vs small molecule Kp ────────────────────────────────────────


def test_nanoparticle_kp_is_20():
    result = simulate_splenic_pk(
        "Nano",
        100.0,
        logP=0.0,
        mw_Da=1e6,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        nanoparticle=True,
    )
    assert result.kp_spleen == pytest.approx(20.0)


def test_small_molecule_kp_below_nanoparticle():
    nano = simulate_splenic_pk(
        "Nano",
        100.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        nanoparticle=True,
    )
    small = simulate_splenic_pk(
        "Small",
        100.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        nanoparticle=False,
    )
    assert nano.kp_spleen > small.kp_spleen


def test_kp_clamp_lower():
    # logP = -5 → raw = (-5 + 1.5)*0.8 / 1.0 = -2.8 → clamp to 0.2
    result = simulate_splenic_pk(
        "LowLogP",
        100.0,
        logP=-5.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    assert result.kp_spleen == pytest.approx(0.2)


def test_kp_clamp_upper():
    # logP = 15, mw_Da=100 → raw = (15+1.5)*0.8 / max(1, 100/400) = 13.2 → clamp to 10.0
    result = simulate_splenic_pk(
        "HighLogP",
        100.0,
        logP=15.0,
        mw_Da=100.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    assert result.kp_spleen == pytest.approx(10.0)


def test_higher_logP_gives_higher_kp():
    low = simulate_splenic_pk("L", 100.0, logP=1.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0)
    high = simulate_splenic_pk("H", 100.0, logP=3.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0)
    assert high.kp_spleen > low.kp_spleen


# ── macrophage uptake ─────────────────────────────────────────────────────────


def test_small_molecule_no_mac_uptake():
    result = simulate_splenic_pk(
        "Small",
        100.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        nanoparticle=False,
    )
    assert result.macrophage_uptake_fraction == pytest.approx(0.0)


def test_nanoparticle_has_mac_uptake():
    result = simulate_splenic_pk(
        "Nano",
        100.0,
        logP=2.0,
        mw_Da=1e6,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        nanoparticle=True,
        t_end_h=72.0,
    )
    assert result.macrophage_uptake_fraction > 0.0


def test_large_mw_molecule_has_mac_uptake():
    result = simulate_splenic_pk(
        "Biologic",
        100.0,
        logP=0.0,
        mw_Da=15000.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=30.0,
        nanoparticle=False,
        t_end_h=72.0,
    )
    assert result.macrophage_uptake_fraction > 0.0


def test_mac_uptake_fraction_between_0_and_1():
    result = default_result(nanoparticle=True, t_end_h=72.0)
    assert 0.0 <= result.macrophage_uptake_fraction <= 1.0


# ── AUC and AUC ratio ─────────────────────────────────────────────────────────


def test_auc_plasma_positive():
    result = default_result()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_spleen_positive():
    result = default_result()
    assert result.auc_spleen_mg_h_per_g > 0.0


def test_spleen_to_plasma_auc_ratio_positive():
    result = default_result()
    assert result.spleen_to_plasma_auc_ratio > 0.0


def test_nanoparticle_higher_spleen_auc_ratio():
    nano = simulate_splenic_pk(
        "Nano",
        100.0,
        logP=2.0,
        mw_Da=1e6,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        nanoparticle=True,
    )
    small = simulate_splenic_pk(
        "Small",
        100.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        nanoparticle=False,
    )
    assert nano.spleen_to_plasma_auc_ratio > small.spleen_to_plasma_auc_ratio


# ── t_half_spleen ─────────────────────────────────────────────────────────────


def test_t_half_spleen_positive():
    result = default_result()
    assert result.t_half_spleen_h > 0.0


def test_t_half_spleen_formula_small_molecule():
    # For small molecule: k_mac = 0, t_half = ln(2) / k_out
    # k_out = Qs / Vd_spleen_L = 3.6 / (150 * kp / 1000)
    result = simulate_splenic_pk(
        "SM",
        100.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        spleen_weight_g=150.0,
        nanoparticle=False,
    )
    Qs = 3.6
    Vd_spleen_L = 150.0 * result.kp_spleen / 1000.0
    k_out = Qs / Vd_spleen_L
    expected_t_half = math.log(2.0) / k_out
    assert result.t_half_spleen_h == pytest.approx(expected_t_half, rel=1e-6)


def test_nanoparticle_shorter_t_half_spleen():
    # Nanoparticle has extra k_mac=0.05, so t_half should be shorter
    nano = simulate_splenic_pk(
        "Nano",
        100.0,
        logP=2.0,
        mw_Da=1e6,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        nanoparticle=True,
        spleen_weight_g=150.0,
    )
    # Same kp=20 but with k_mac
    Qs = 3.6
    Vd_spleen_L = 150.0 * 20.0 / 1000.0
    k_out = Qs / Vd_spleen_L
    k_mac = 0.05
    expected = math.log(2.0) / (k_out + k_mac)
    assert nano.t_half_spleen_h == pytest.approx(expected, rel=1e-6)


# ── cmax values ──────────────────────────────────────────────────────────────


def test_cmax_plasma_positive():
    result = default_result()
    assert result.cmax_plasma_mg_L > 0.0


def test_cmax_spleen_positive():
    result = default_result()
    assert result.cmax_spleen_mg_g > 0.0


def test_cmax_equals_max_of_list():
    result = default_result()
    assert result.cmax_plasma_mg_L == pytest.approx(max(result.c_plasma_mg_L))
    assert result.cmax_spleen_mg_g == pytest.approx(max(result.c_spleen_mg_g))


# ── validation ────────────────────────────────────────────────────────────────


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_splenic_pk("X", 0.0, 1.0, 300.0, 5.0, 50.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_splenic_pk("X", -10.0, 1.0, 300.0, 5.0, 50.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_splenic_pk("X", 100.0, 1.0, 300.0, 0.0, 50.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys"):
        simulate_splenic_pk("X", 100.0, 1.0, 300.0, 5.0, 0.0)


def test_validation_spleen_weight_zero():
    with pytest.raises(ValueError, match="spleen_weight"):
        simulate_splenic_pk("X", 100.0, 1.0, 300.0, 5.0, 50.0, spleen_weight_g=0.0)


# ── notes field ───────────────────────────────────────────────────────────────


def test_notes_contains_drug_name():
    result = simulate_splenic_pk("Aspirin", 100.0, 1.0, 180.0, 5.0, 50.0)
    assert "Aspirin" in result.notes


def test_notes_contains_kp():
    result = default_result()
    assert "kp_spleen" in result.notes
