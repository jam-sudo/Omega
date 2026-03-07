"""Tests for Phase 216: vaginal_pk."""

import pytest

from omega_pbpk.core.vaginal_pk import (
    VaginalPKResult,
    compare_formulations,
    simulate_vaginal_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    formulation_type="gel",
    logP=2.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
)


def gel_result() -> VaginalPKResult:
    return simulate_vaginal_pk(**BASE_KWARGS)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = gel_result()
    assert isinstance(result, VaginalPKResult)


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = gel_result()
    assert result.drug_name == "TestDrug"


def test_dose_preserved():
    result = gel_result()
    assert result.dose_mg == 100.0


def test_formulation_type_preserved():
    result = gel_result()
    assert result.formulation_type == "gel"


def test_logP_preserved():
    result = gel_result()
    assert result.logP == 2.0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_systemic_starts_at_zero():
    result = gel_result()
    assert result.c_systemic_mg_L[0] == 0.0


def test_local_starts_at_zero():
    result = gel_result()
    assert result.c_local_mg_mL[0] == 0.0


# ---------------------------------------------------------------------------
# PK behaviour
# ---------------------------------------------------------------------------


def test_cmax_systemic_positive():
    result = gel_result()
    assert result.cmax_systemic_mg_L > 0.0


def test_cmax_local_positive():
    result = gel_result()
    assert result.cmax_local_mg_mL > 0.0


def test_auc_systemic_positive():
    result = gel_result()
    assert result.auc_systemic_mg_h_per_L > 0.0


def test_f_systemic_absorbed_in_range():
    result = gel_result()
    assert 0.0 <= result.f_systemic_absorbed <= 1.0


def test_local_concentration_rises_then_falls():
    result = gel_result()
    c_local = result.c_local_mg_mL
    # Find peak
    peak_idx = c_local.index(max(c_local))
    # Peak must not be at the start
    assert peak_idx > 0
    # Last value should be lower than peak
    assert c_local[-1] < max(c_local)


def test_tmax_systemic_positive():
    result = gel_result()
    assert result.tmax_systemic_h > 0.0


def test_times_list_length():
    result = gel_result()
    assert len(result.times_h) == len(result.c_systemic_mg_L) == len(result.c_local_mg_mL)


# ---------------------------------------------------------------------------
# Ring vs gel comparison
# ---------------------------------------------------------------------------


def test_ring_lower_cmax_systemic_than_gel():
    """Ring releases slower than gel → lower systemic peak."""
    gel = simulate_vaginal_pk(
        drug_name="Drug",
        dose_mg=100.0,
        formulation_type="gel",
        logP=2.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=96.0,
    )
    ring = simulate_vaginal_pk(
        drug_name="Drug",
        dose_mg=100.0,
        formulation_type="ring",
        logP=2.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=96.0,
    )
    assert ring.cmax_systemic_mg_L < gel.cmax_systemic_mg_L


def test_tablet_higher_release_rate_than_ring():
    """Tablet (k_release=0.5) faster than ring (k_release=0.02)."""
    tablet = simulate_vaginal_pk(
        drug_name="Drug",
        dose_mg=100.0,
        formulation_type="tablet",
        logP=2.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    ring = simulate_vaginal_pk(
        drug_name="Drug",
        dose_mg=100.0,
        formulation_type="ring",
        logP=2.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert tablet.cmax_systemic_mg_L > ring.cmax_systemic_mg_L


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_returns_three():
    results = compare_formulations(
        drug_name="Drug", dose_mg=100.0, logP=2.0, cl_L_per_h=5.0, vd_L=50.0
    )
    assert len(results) == 3


def test_compare_formulations_sorted_by_auc():
    results = compare_formulations(
        drug_name="Drug", dose_mg=100.0, logP=2.0, cl_L_per_h=5.0, vd_L=50.0
    )
    aucs = [r.auc_systemic_mg_h_per_L for r in results]
    assert aucs == sorted(aucs)


def test_compare_formulations_all_formulations_present():
    results = compare_formulations(
        drug_name="Drug", dose_mg=100.0, logP=2.0, cl_L_per_h=5.0, vd_L=50.0
    )
    types = {r.formulation_type for r in results}
    assert types == {"gel", "ring", "tablet"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_vaginal_pk(
            drug_name="X",
            dose_mg=0.0,
            formulation_type="gel",
            logP=2.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


def test_validation_dose_negative():
    with pytest.raises(ValueError):
        simulate_vaginal_pk(
            drug_name="X",
            dose_mg=-10.0,
            formulation_type="gel",
            logP=2.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


def test_validation_invalid_formulation():
    with pytest.raises(ValueError):
        simulate_vaginal_pk(
            drug_name="X",
            dose_mg=100.0,
            formulation_type="patch",
            logP=2.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


def test_validation_cl_zero():
    with pytest.raises(ValueError):
        simulate_vaginal_pk(
            drug_name="X",
            dose_mg=100.0,
            formulation_type="gel",
            logP=2.0,
            cl_L_per_h=0.0,
            vd_L=50.0,
        )


def test_validation_cl_negative():
    with pytest.raises(ValueError):
        simulate_vaginal_pk(
            drug_name="X",
            dose_mg=100.0,
            formulation_type="gel",
            logP=2.0,
            cl_L_per_h=-1.0,
            vd_L=50.0,
        )
