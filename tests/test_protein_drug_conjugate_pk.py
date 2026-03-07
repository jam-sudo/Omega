"""Tests for Phase 315: Protein Drug Conjugate PK Model."""

import pytest

from omega_pbpk.core.protein_drug_conjugate_pk import (
    PDCPKResult,
    compare_release_rates,
    simulate_pdc_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _default_iv():
    return simulate_pdc_pk(
        drug_name="TestDrug",
        dose_mg=10.0,
        route="iv_bolus",
        cl_protein_L_per_h=0.02,
        vd_protein_L=5.0,
        k_release_per_h=0.01,
        cl_free_drug_L_per_h=2.0,
        vd_free_drug_L=20.0,
        t_end_h=168.0,
        dt_h=0.5,
    )


def _default_sc():
    return simulate_pdc_pk(
        drug_name="TestDrugSC",
        dose_mg=10.0,
        route="sc",
        cl_protein_L_per_h=0.02,
        vd_protein_L=5.0,
        k_release_per_h=0.01,
        cl_free_drug_L_per_h=2.0,
        vd_free_drug_L=20.0,
        t_end_h=168.0,
        dt_h=0.5,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type_iv():
    result = _default_iv()
    assert isinstance(result, PDCPKResult)


def test_return_type_sc():
    result = _default_sc()
    assert isinstance(result, PDCPKResult)


# ---------------------------------------------------------------------------
# Drug name and route preserved
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = _default_iv()
    assert result.drug_name == "TestDrug"


def test_route_preserved_iv():
    result = _default_iv()
    assert result.route == "iv_bolus"


def test_route_preserved_sc():
    result = _default_sc()
    assert result.route == "sc"


# ---------------------------------------------------------------------------
# Time array consistency
# ---------------------------------------------------------------------------


def test_time_array_matches_concentration_arrays():
    result = _default_iv()
    assert len(result.times_h) == len(result.c_pdc_mg_L)
    assert len(result.times_h) == len(result.c_free_mg_L)


def test_time_array_starts_at_zero():
    result = _default_iv()
    assert result.times_h[0] == pytest.approx(0.0)


def test_time_array_ends_near_t_end():
    result = _default_iv()
    assert result.times_h[-1] == pytest.approx(168.0, abs=1.0)


# ---------------------------------------------------------------------------
# PDC concentrations
# ---------------------------------------------------------------------------


def test_cmax_pdc_positive_iv():
    result = _default_iv()
    assert result.cmax_pdc_mg_L > 0.0


def test_cmax_pdc_positive_sc():
    result = _default_sc()
    assert result.cmax_pdc_mg_L > 0.0


def test_auc_pdc_positive_iv():
    result = _default_iv()
    assert result.auc_pdc_mg_h_per_L > 0.0


def test_auc_pdc_positive_sc():
    result = _default_sc()
    assert result.auc_pdc_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Free drug concentrations
# ---------------------------------------------------------------------------


def test_cmax_free_positive():
    result = _default_iv()
    assert result.cmax_free_mg_L > 0.0


def test_auc_free_positive():
    result = _default_iv()
    assert result.auc_free_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# tmax_free is delayed
# ---------------------------------------------------------------------------


def test_tmax_free_positive():
    result = _default_iv()
    assert result.tmax_free_h > 0.0


def test_tmax_free_delayed_vs_iv_pdc():
    """For IV bolus, PDC peaks at t=0; free drug peaks later."""
    result = _default_iv()
    # PDC starts at max at t=0 for IV bolus, tmax_free should be > 0
    assert result.tmax_free_h > 0.0


# ---------------------------------------------------------------------------
# sustained_release_index
# ---------------------------------------------------------------------------


def test_sustained_release_index_positive():
    result = _default_iv()
    assert result.sustained_release_index > 0.0


def test_sustained_release_index_formula():
    """sustained_release_index = auc_free / auc_pdc"""
    result = _default_iv()
    expected = result.auc_free_mg_h_per_L / result.auc_pdc_mg_h_per_L
    assert result.sustained_release_index == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Effect of k_release
# ---------------------------------------------------------------------------


def test_higher_k_release_higher_auc_free():
    """Higher release rate should yield higher free drug AUC (within short sim)."""
    low = simulate_pdc_pk(
        drug_name="Drug",
        dose_mg=10.0,
        route="iv_bolus",
        cl_protein_L_per_h=0.02,
        vd_protein_L=5.0,
        k_release_per_h=0.005,
        cl_free_drug_L_per_h=2.0,
        vd_free_drug_L=20.0,
        t_end_h=48.0,
        dt_h=0.5,
    )
    high = simulate_pdc_pk(
        drug_name="Drug",
        dose_mg=10.0,
        route="iv_bolus",
        cl_protein_L_per_h=0.02,
        vd_protein_L=5.0,
        k_release_per_h=0.1,
        cl_free_drug_L_per_h=2.0,
        vd_free_drug_L=20.0,
        t_end_h=48.0,
        dt_h=0.5,
    )
    assert high.auc_free_mg_h_per_L > low.auc_free_mg_h_per_L


def test_k_release_stored_in_result():
    result = simulate_pdc_pk(
        drug_name="Drug",
        dose_mg=10.0,
        route="iv_bolus",
        cl_protein_L_per_h=0.02,
        vd_protein_L=5.0,
        k_release_per_h=0.03,
        cl_free_drug_L_per_h=2.0,
        vd_free_drug_L=20.0,
        t_end_h=24.0,
        dt_h=0.5,
    )
    assert result.k_release_per_h == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# compare_release_rates
# ---------------------------------------------------------------------------


def test_compare_release_rates_returns_list():
    results = compare_release_rates(
        drug_name="Drug",
        dose_mg=10.0,
        k_release_list=[0.005, 0.05, 0.1],
        cl_protein_L_per_h=0.02,
        vd_protein_L=5.0,
        cl_free_drug_L_per_h=2.0,
        vd_free_drug_L=20.0,
        t_end_h=48.0,
        dt_h=0.5,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_release_rates_sorted_descending():
    results = compare_release_rates(
        drug_name="Drug",
        dose_mg=10.0,
        k_release_list=[0.005, 0.05, 0.1],
        cl_protein_L_per_h=0.02,
        vd_protein_L=5.0,
        cl_free_drug_L_per_h=2.0,
        vd_free_drug_L=20.0,
        t_end_h=48.0,
        dt_h=0.5,
    )
    aucs = [r.auc_free_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_release_rates_all_pdc_results():
    results = compare_release_rates(
        drug_name="Drug",
        dose_mg=10.0,
        k_release_list=[0.01, 0.1],
        cl_protein_L_per_h=0.02,
        vd_protein_L=5.0,
        cl_free_drug_L_per_h=2.0,
        vd_free_drug_L=20.0,
        t_end_h=48.0,
        dt_h=0.5,
    )
    for r in results:
        assert isinstance(r, PDCPKResult)


# ---------------------------------------------------------------------------
# notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = _default_iv()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_pdc_pk(
            drug_name="Drug",
            dose_mg=10.0,
            route="oral",
            cl_protein_L_per_h=0.02,
            vd_protein_L=5.0,
            k_release_per_h=0.01,
            cl_free_drug_L_per_h=2.0,
            vd_free_drug_L=20.0,
        )


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pdc_pk(
            drug_name="Drug",
            dose_mg=0.0,
            route="iv_bolus",
            cl_protein_L_per_h=0.02,
            vd_protein_L=5.0,
            k_release_per_h=0.01,
            cl_free_drug_L_per_h=2.0,
            vd_free_drug_L=20.0,
        )


def test_invalid_cl_protein_negative():
    with pytest.raises(ValueError, match="cl_protein"):
        simulate_pdc_pk(
            drug_name="Drug",
            dose_mg=10.0,
            route="iv_bolus",
            cl_protein_L_per_h=-0.1,
            vd_protein_L=5.0,
            k_release_per_h=0.01,
            cl_free_drug_L_per_h=2.0,
            vd_free_drug_L=20.0,
        )


def test_invalid_k_release_zero():
    with pytest.raises(ValueError, match="k_release"):
        simulate_pdc_pk(
            drug_name="Drug",
            dose_mg=10.0,
            route="iv_bolus",
            cl_protein_L_per_h=0.02,
            vd_protein_L=5.0,
            k_release_per_h=0.0,
            cl_free_drug_L_per_h=2.0,
            vd_free_drug_L=20.0,
        )


def test_invalid_cl_free_drug_negative():
    with pytest.raises(ValueError, match="cl_free"):
        simulate_pdc_pk(
            drug_name="Drug",
            dose_mg=10.0,
            route="iv_bolus",
            cl_protein_L_per_h=0.02,
            vd_protein_L=5.0,
            k_release_per_h=0.01,
            cl_free_drug_L_per_h=-1.0,
            vd_free_drug_L=20.0,
        )


def test_invalid_vd_free_zero():
    with pytest.raises(ValueError, match="vd_free"):
        simulate_pdc_pk(
            drug_name="Drug",
            dose_mg=10.0,
            route="iv_bolus",
            cl_protein_L_per_h=0.02,
            vd_protein_L=5.0,
            k_release_per_h=0.01,
            cl_free_drug_L_per_h=2.0,
            vd_free_drug_L=0.0,
        )


def test_invalid_t_end_zero():
    with pytest.raises(ValueError, match="t_end"):
        simulate_pdc_pk(
            drug_name="Drug",
            dose_mg=10.0,
            route="iv_bolus",
            cl_protein_L_per_h=0.02,
            vd_protein_L=5.0,
            k_release_per_h=0.01,
            cl_free_drug_L_per_h=2.0,
            vd_free_drug_L=20.0,
            t_end_h=0.0,
        )
