"""Tests for Phase 631 — Drug Cellular Uptake Kinetics."""

import pytest

from omega_pbpk.core.cellular_uptake_p631 import (
    CellularUptakeResult,
    simulate_cellular_uptake,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def run_base(
    drug_name="TestDrug",
    cell_type="hepatocyte",
    c_extracell_mg_L=10.0,
    logP=2.0,
    mw_Da=300.0,
    vmax_mg_L_per_h=0.0,
    km_mg_L=1.0,
    k_binding=0.5,
    bmax_mg_L=5.0,
    t_end_h=6.0,
    dt_h=0.02,
):
    return simulate_cellular_uptake(
        drug_name=drug_name,
        cell_type=cell_type,
        c_extracell_mg_L=c_extracell_mg_L,
        logP=logP,
        mw_Da=mw_Da,
        vmax_mg_L_per_h=vmax_mg_L_per_h,
        km_mg_L=km_mg_L,
        k_binding=k_binding,
        bmax_mg_L=bmax_mg_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )


# ── Return type and structure ─────────────────────────────────────────────────


def test_returns_cellular_uptake_result():
    r = run_base()
    assert isinstance(r, CellularUptakeResult)


def test_drug_name_preserved():
    r = run_base(drug_name="Ibuprofen")
    assert r.drug_name == "Ibuprofen"


def test_cell_type_preserved():
    r = run_base(cell_type="cardiomyocyte")
    assert r.cell_type == "cardiomyocyte"


def test_dose_conc_preserved():
    r = run_base(c_extracell_mg_L=5.0)
    assert r.dose_conc_mg_L == 5.0


def test_list_lengths_match():
    r = run_base(t_end_h=4.0, dt_h=0.1)
    n = len(r.times_h)
    assert len(r.c_cytoplasm_mg_L) == n
    assert len(r.c_bound_mg_L) == n
    assert len(r.c_total_intracell_mg_L) == n
    assert len(r.cellular_accumulation_ratio) == n


def test_times_start_at_zero():
    r = run_base()
    assert r.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    r = run_base(t_end_h=6.0, dt_h=0.02)
    assert r.times_h[-1] == pytest.approx(6.0, abs=0.05)


def test_initial_concentrations_zero():
    r = run_base()
    assert r.c_cytoplasm_mg_L[0] == pytest.approx(0.0)
    assert r.c_bound_mg_L[0] == pytest.approx(0.0)
    assert r.c_total_intracell_mg_L[0] == pytest.approx(0.0)


# ── Accumulation dynamics ─────────────────────────────────────────────────────


def test_cytoplasm_increases_over_time():
    r = run_base()
    # Concentration should rise from 0
    assert r.c_cytoplasm_mg_L[-1] > r.c_cytoplasm_mg_L[0]


def test_bound_increases_over_time():
    r = run_base()
    assert r.c_bound_mg_L[-1] > r.c_bound_mg_L[0]


def test_total_intracell_is_sum():
    r = run_base()
    for f, b, tot in zip(r.c_cytoplasm_mg_L, r.c_bound_mg_L, r.c_total_intracell_mg_L):
        assert tot == pytest.approx(f + b, abs=1e-9)


def test_accumulation_ratio_nonneg():
    r = run_base()
    assert all(v >= 0.0 for v in r.cellular_accumulation_ratio)


def test_steady_state_accumulation_matches_last():
    r = run_base()
    assert r.steady_state_accumulation == pytest.approx(r.cellular_accumulation_ratio[-1], rel=1e-6)


def test_accumulation_ratio_calculation():
    r = run_base(c_extracell_mg_L=10.0)
    for ratio, tot in zip(r.cellular_accumulation_ratio, r.c_total_intracell_mg_L):
        assert ratio == pytest.approx(tot / 10.0, abs=1e-9)


# ── Active transport ──────────────────────────────────────────────────────────


def test_no_active_transport_contribution_zero():
    r = run_base(vmax_mg_L_per_h=0.0)
    assert r.active_transport_contribution == pytest.approx(0.0, abs=1e-9)


def test_active_transport_increases_accumulation():
    r_passive = run_base(vmax_mg_L_per_h=0.0)
    r_active = run_base(vmax_mg_L_per_h=5.0, km_mg_L=1.0)
    assert r_active.steady_state_accumulation > r_passive.steady_state_accumulation


def test_active_transport_contribution_in_range():
    r = run_base(vmax_mg_L_per_h=3.0)
    assert 0.0 <= r.active_transport_contribution <= 1.0


def test_high_vmax_high_contribution():
    r = run_base(vmax_mg_L_per_h=20.0, c_extracell_mg_L=1.0, km_mg_L=0.1)
    assert r.active_transport_contribution > 0.5


# ── Binding effects ───────────────────────────────────────────────────────────


def test_bound_does_not_exceed_bmax():
    r = run_base(bmax_mg_L=5.0)
    assert all(b <= 5.0 + 1e-9 for b in r.c_bound_mg_L)


def test_higher_bmax_increases_total_uptake():
    r_low = run_base(bmax_mg_L=2.0)
    r_high = run_base(bmax_mg_L=20.0)
    assert r_high.steady_state_accumulation > r_low.steady_state_accumulation


def test_higher_k_binding_speeds_uptake():
    r_slow = run_base(k_binding=0.1, t_end_h=10.0)
    r_fast = run_base(k_binding=2.0, t_end_h=10.0)
    # Faster binding → binds more rapidly → higher bound at early time points
    idx = len(r_fast.c_bound_mg_L) // 4
    assert r_fast.c_bound_mg_L[idx] >= r_slow.c_bound_mg_L[idx]


# ── logP effect ───────────────────────────────────────────────────────────────


def test_higher_logP_increases_passive_uptake():
    r_low = run_base(logP=0.5, vmax_mg_L_per_h=0.0, bmax_mg_L=0.01)
    r_high = run_base(logP=4.0, vmax_mg_L_per_h=0.0, bmax_mg_L=0.01)
    assert r_high.steady_state_accumulation > r_low.steady_state_accumulation


def test_kp_cell_positive():
    r = run_base()
    assert r.kp_cell >= 0.0


def test_kp_cell_is_max_free_over_extracell():
    r = run_base(c_extracell_mg_L=10.0)
    expected = max(r.c_cytoplasm_mg_L) / 10.0
    assert r.kp_cell == pytest.approx(expected, rel=1e-6)


# ── t_half_uptake ─────────────────────────────────────────────────────────────


def test_t_half_uptake_within_simulation():
    r = run_base(t_end_h=6.0)
    assert 0.0 <= r.t_half_uptake_h <= 6.0


def test_t_half_shorter_with_higher_passive():
    r_slow = run_base(logP=0.1, t_end_h=20.0, dt_h=0.1)
    r_fast = run_base(logP=3.0, t_end_h=20.0, dt_h=0.1)
    assert r_fast.t_half_uptake_h <= r_slow.t_half_uptake_h


# ── Validation errors ─────────────────────────────────────────────────────────


def test_invalid_c_extracell_zero():
    with pytest.raises(ValueError):
        simulate_cellular_uptake("Drug", "cell", 0.0, 2.0, 300.0)


def test_invalid_c_extracell_negative():
    with pytest.raises(ValueError):
        simulate_cellular_uptake("Drug", "cell", -1.0, 2.0, 300.0)


def test_invalid_vmax_negative():
    with pytest.raises(ValueError):
        simulate_cellular_uptake("Drug", "cell", 10.0, 2.0, 300.0, vmax_mg_L_per_h=-1.0)


def test_invalid_km_zero():
    with pytest.raises(ValueError):
        simulate_cellular_uptake("Drug", "cell", 10.0, 2.0, 300.0, km_mg_L=0.0)


def test_invalid_k_binding_zero():
    with pytest.raises(ValueError):
        simulate_cellular_uptake("Drug", "cell", 10.0, 2.0, 300.0, k_binding=0.0)


def test_invalid_bmax_zero():
    with pytest.raises(ValueError):
        simulate_cellular_uptake("Drug", "cell", 10.0, 2.0, 300.0, bmax_mg_L=0.0)


# ── Notes field ───────────────────────────────────────────────────────────────


def test_notes_is_string():
    r = run_base()
    assert isinstance(r.notes, str)


def test_notes_contains_k_passive():
    r = run_base()
    assert "k_passive" in r.notes
