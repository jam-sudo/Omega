"""Tests for core/inhalation_pbpk.py — Phase 877."""

import pytest

from omega_pbpk.core.inhalation_pbpk import (
    InhalationPKResult,
    compare_inhalation_routes,
    simulate_inhalation_pk,
)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_returns_inhalation_pk_result():
    result = simulate_inhalation_pk("salbutamol", dose_mg=1.0)
    assert isinstance(result, InhalationPKResult)


def test_result_fields_correct_types():
    result = simulate_inhalation_pk("drug_a", dose_mg=1.0)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.dose_mg, float)
    assert isinstance(result.route, str)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.a_central_airway_mg, list)
    assert isinstance(result.a_alveolar_mg, list)
    assert isinstance(result.cmax_plasma_mg_L, float)
    assert isinstance(result.tmax_plasma_h, float)
    assert isinstance(result.auc_plasma_mg_h_per_L, float)
    assert isinstance(result.f_pulmonary, float)
    assert isinstance(result.lung_deposition_central_frac, float)
    assert isinstance(result.lung_deposition_alveolar_frac, float)
    assert isinstance(result.notes, str)


def test_cmax_positive():
    result = simulate_inhalation_pk("drug_a", dose_mg=1.0)
    assert result.cmax_plasma_mg_L > 0


def test_auc_positive():
    result = simulate_inhalation_pk("drug_a", dose_mg=1.0)
    assert result.auc_plasma_mg_h_per_L > 0


def test_times_length_matches_plasma():
    result = simulate_inhalation_pk("drug_a", dose_mg=1.0, t_end_h=4.0, dt_h=0.1)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.a_central_airway_mg)
    assert len(result.times_h) == len(result.a_alveolar_mg)


# ---------------------------------------------------------------------------
# Route-specific tests
# ---------------------------------------------------------------------------


def test_dpi_route_stored():
    result = simulate_inhalation_pk("drug", dose_mg=1.0, route="DPI")
    assert result.route == "DPI"


def test_nebulizer_route_stored():
    result = simulate_inhalation_pk("drug", dose_mg=1.0, route="nebulizer")
    assert result.route == "nebulizer"


def test_mdi_route_stored():
    result = simulate_inhalation_pk("drug", dose_mg=1.0, route="MDI")
    assert result.route == "MDI"


def test_dpi_higher_auc_than_nebulizer():
    """DPI has higher alveolar fraction → more absorption than nebulizer."""
    dpi = simulate_inhalation_pk("drug", dose_mg=1.0, route="DPI")
    neb = simulate_inhalation_pk("drug", dose_mg=1.0, route="nebulizer")
    assert dpi.auc_plasma_mg_h_per_L > neb.auc_plasma_mg_h_per_L


def test_dpi_higher_auc_than_mdi():
    """DPI has higher alveolar fraction than MDI."""
    dpi = simulate_inhalation_pk("drug", dose_mg=1.0, route="DPI")
    mdi = simulate_inhalation_pk("drug", dose_mg=1.0, route="MDI")
    assert dpi.auc_plasma_mg_h_per_L > mdi.auc_plasma_mg_h_per_L


def test_deposition_fractions_sum_to_one():
    result = simulate_inhalation_pk("drug", dose_mg=1.0)
    total = result.lung_deposition_central_frac + result.lung_deposition_alveolar_frac
    assert abs(total - 1.0) < 1e-9


def test_nebulizer_alveolar_frac_is_lower():
    dpi = simulate_inhalation_pk("drug", dose_mg=1.0, route="DPI")
    neb = simulate_inhalation_pk("drug", dose_mg=1.0, route="nebulizer")
    assert neb.lung_deposition_alveolar_frac < dpi.lung_deposition_alveolar_frac


# ---------------------------------------------------------------------------
# Pharmacological sensitivity tests
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    res1 = simulate_inhalation_pk("drug", dose_mg=1.0)
    res2 = simulate_inhalation_pk("drug", dose_mg=2.0)
    ratio = res2.cmax_plasma_mg_L / res1.cmax_plasma_mg_L
    assert abs(ratio - 2.0) < 0.05, f"Expected ~2x Cmax, got ratio={ratio:.3f}"


def test_double_dose_doubles_auc():
    res1 = simulate_inhalation_pk("drug", dose_mg=1.0)
    res2 = simulate_inhalation_pk("drug", dose_mg=2.0)
    ratio = res2.auc_plasma_mg_h_per_L / res1.auc_plasma_mg_h_per_L
    assert abs(ratio - 2.0) < 0.05, f"Expected ~2x AUC, got ratio={ratio:.3f}"


def test_f_pulmonary_in_range():
    result = simulate_inhalation_pk("drug", dose_mg=1.0)
    assert 0.0 <= result.f_pulmonary <= 1.5  # can exceed 1 only with very slow CL


def test_notes_present():
    result = simulate_inhalation_pk("drug", dose_mg=1.0)
    assert len(result.notes) > 0


def test_notes_good_bioavailability_high_dose():
    """Large f_lung should give good bioavailability."""
    result = simulate_inhalation_pk(
        "drug",
        dose_mg=5.0,
        f_lung=0.9,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert "Good" in result.notes or "Moderate" in result.notes


# ---------------------------------------------------------------------------
# compare_inhalation_routes tests
# ---------------------------------------------------------------------------


def test_compare_returns_three_results():
    results = compare_inhalation_routes("drug", dose_mg=1.0)
    assert len(results) == 3


def test_compare_sorted_descending_auc():
    results = compare_inhalation_routes("drug", dose_mg=1.0)
    aucs = [r.auc_plasma_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_contains_all_routes():
    results = compare_inhalation_routes("drug", dose_mg=1.0)
    routes = {r.route for r in results}
    assert routes == {"DPI", "nebulizer", "MDI"}


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_inhalation_pk("drug", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_inhalation_pk("drug", dose_mg=-1.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_inhalation_pk("drug", dose_mg=1.0, cl_L_per_h=0.0)


def test_validation_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_inhalation_pk("drug", dose_mg=1.0, vd_L=-1.0)


def test_validation_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_inhalation_pk("drug", dose_mg=1.0, route="spacer")
