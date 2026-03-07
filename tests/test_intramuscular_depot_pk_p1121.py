"""Tests for IM depot PK simulation — Phase 1121."""

import math
import pytest

from omega_pbpk.core.intramuscular_depot_pk_p1121 import (
    IMDepotPKResult,
    _DEPOT_PARAMS,
    compare_depot_types,
    simulate_im_depot_pk,
)


# ---------------------------------------------------------------------------
# Basic return type / field checks
# ---------------------------------------------------------------------------

def test_returns_correct_type():
    result = simulate_im_depot_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, IMDepotPKResult)


def test_drug_name_preserved():
    result = simulate_im_depot_pk("Haloperidol", dose_mg=50.0)
    assert result.drug_name == "Haloperidol"


def test_dose_mg_preserved():
    result = simulate_im_depot_pk("DrugA", dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_depot_type_preserved():
    for depot in _DEPOT_PARAMS:
        r = simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type=depot)
        assert r.depot_type == depot


def test_times_starts_at_zero():
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0)
    assert result.times_h[0] == 0.0


def test_plasma_initial_zero():
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0)
    assert math.isclose(result.c_plasma_mg_L[0], 0.0, abs_tol=1e-12)


def test_depot_initial_equals_dose_times_fbio():
    for depot, params in _DEPOT_PARAMS.items():
        r = simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type=depot)
        expected = 100.0 * params["f_bioavailable"]
        assert math.isclose(r.a_depot_mg[0], expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Pharmacokinetic correctness
# ---------------------------------------------------------------------------

def test_cmax_positive():
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_tmax_positive():
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0)
    assert result.tmax_h > 0.0


def test_auc_positive():
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_aqueous_highest_cmax():
    """Aqueous (fastest release) should give highest Cmax among all types."""
    results = {
        depot: simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type=depot, t_end_h=720.0)
        for depot in _DEPOT_PARAMS
    }
    assert results["aqueous"].cmax_mg_L == max(r.cmax_mg_L for r in results.values())


def test_oil_longest_tmax():
    """Oil depot (slowest release) should give latest Tmax."""
    results = {
        depot: simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type=depot, t_end_h=720.0)
        for depot in _DEPOT_PARAMS
    }
    assert results["oil"].tmax_h == max(r.tmax_h for r in results.values())


def test_aqueous_fastest_tmax():
    """Aqueous (fastest release) should have smallest Tmax."""
    results = {
        depot: simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type=depot, t_end_h=720.0)
        for depot in _DEPOT_PARAMS
    }
    assert results["aqueous"].tmax_h == min(r.tmax_h for r in results.values())


def test_t_half_apparent_from_k_rel():
    """t_half_apparent should equal ln(2)/k_rel for each depot type."""
    for depot, params in _DEPOT_PARAMS.items():
        r = simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type=depot)
        expected = math.log(2.0) / params["k_rel_per_h"]
        assert math.isclose(r.t_half_apparent_h, expected, rel_tol=1e-9), (
            f"Depot '{depot}': expected t½={expected:.2f}h, got {r.t_half_apparent_h:.2f}h"
        )


def test_depot_depletes_over_time():
    """Depot amount should decrease monotonically."""
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type="aqueous", t_end_h=48.0)
    assert result.a_depot_mg[-1] < result.a_depot_mg[0]


def test_depot_depleted_pct_in_range():
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0)
    assert 0.0 <= result.depot_depleted_pct <= 100.0


def test_aqueous_depot_mostly_depleted_short_run():
    """Aqueous has k_rel=0.5/h; after 720h essentially all released."""
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type="aqueous", t_end_h=720.0)
    assert result.depot_depleted_pct > 99.0


def test_oil_depot_not_fully_depleted_at_720h():
    """Oil k_rel=0.005/h, t½≈138h; at 720h ~96% depleted but <100%."""
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type="oil", t_end_h=720.0)
    # at 720h, remaining = exp(-0.005*720) ≈ 0.027 → ~97% depleted
    assert result.depot_depleted_pct < 100.0
    assert result.depot_depleted_pct > 90.0


# ---------------------------------------------------------------------------
# Flip-flop kinetics
# ---------------------------------------------------------------------------

def test_flip_flop_oil_microsphere():
    """Oil and microsphere have slow k_rel → flip-flop when k_rel < CL/Vd."""
    # Default: CL=5, Vd=50 → ke=0.1; oil k_rel=0.005 < 0.1 → flip-flop
    oil = simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type="oil")
    assert oil.flip_flop is True

    micro = simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type="microsphere")
    assert micro.flip_flop is True


def test_flip_flop_false_for_aqueous_slow_elimination():
    """Aqueous k_rel=0.5/h; with very slow ke, no flip-flop."""
    # ke = 1/1000 = 0.001 < 0.5 → no flip-flop
    result = simulate_im_depot_pk(
        "DrugA", dose_mg=100.0, depot_type="aqueous", cl_L_per_h=0.1, vd_L=100.0
    )
    assert result.flip_flop is False


def test_flip_flop_suspension_default_params():
    """Suspension k_rel=0.1/h; ke=0.1 by default → borderline. Test with faster ke."""
    # ke = 10/50 = 0.2 > 0.1 → flip-flop
    result = simulate_im_depot_pk(
        "DrugA", dose_mg=100.0, depot_type="suspension", cl_L_per_h=10.0, vd_L=50.0
    )
    assert result.flip_flop is True


# ---------------------------------------------------------------------------
# compare_depot_types
# ---------------------------------------------------------------------------

def test_compare_depot_types_returns_four():
    results = compare_depot_types("DrugA", dose_mg=100.0)
    assert len(results) == 4


def test_compare_depot_types_sorted_by_auc_descending():
    results = compare_depot_types("DrugA", dose_mg=100.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_depot_types_all_different_types():
    results = compare_depot_types("DrugA", dose_mg=100.0)
    depot_types = {r.depot_type for r in results}
    assert depot_types == set(_DEPOT_PARAMS.keys())


def test_compare_depot_types_passes_cl_vd():
    results_a = compare_depot_types("DrugA", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0)
    results_b = compare_depot_types("DrugA", dose_mg=100.0, cl_L_per_h=20.0, vd_L=10.0)
    # Higher CL → lower AUC
    assert max(r.auc_mg_h_per_L for r in results_a) > max(r.auc_mg_h_per_L for r in results_b)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_im_depot_pk("DrugA", dose_mg=0.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_im_depot_pk("DrugA", dose_mg=-50.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_im_depot_pk("DrugA", dose_mg=100.0, cl_L_per_h=0.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_im_depot_pk("DrugA", dose_mg=100.0, vd_L=-1.0)


def test_invalid_depot_type_raises():
    with pytest.raises(ValueError, match="depot_type"):
        simulate_im_depot_pk("DrugA", dose_mg=100.0, depot_type="gel")


def test_notes_is_string():
    result = simulate_im_depot_pk("DrugA", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
