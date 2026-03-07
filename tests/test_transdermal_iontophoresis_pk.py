"""Tests for transdermal iontophoresis PK model — Phase 925."""

import pytest

from omega_pbpk.core.transdermal_iontophoresis_pk import (
    TransdermalIontophoresisPKResult,
    compare_modes,
    simulate_iontophoresis,
)

# ── Basic return-type and structure ──────────────────────────────────────────


def test_return_type_is_correct():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0)
    assert isinstance(result, TransdermalIontophoresisPKResult)


def test_c_plasma_starts_at_zero():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_cmax_plasma_positive():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0)
    assert result.cmax_plasma_mg_L > 0.0


def test_auc_plasma_positive():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0)
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_tmax_positive():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0)
    assert result.tmax_plasma_h > 0.0


def test_bioavailability_pct_in_range():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0)
    assert 0.0 <= result.bioavailability_pct <= 100.0


def test_flux_peak_positive():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0)
    assert result.flux_peak_mg_h > 0.0


def test_notes_non_empty_string():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ── Enhancement factor behaviour ─────────────────────────────────────────────


def test_iontophoresis_enhancement_factor_greater_than_one():
    result = simulate_iontophoresis(
        "TestDrug", dose_mg=10.0, route="iontophoresis", current_mA=0.5, charge_type="cationic"
    )
    assert result.enhancement_factor > 1.0


def test_passive_enhancement_factor_equals_one():
    result = simulate_iontophoresis("TestDrug", dose_mg=10.0, route="passive", current_mA=0.0)
    assert result.enhancement_factor == pytest.approx(1.0)


def test_iontophoresis_gives_higher_auc_than_passive():
    ionto = simulate_iontophoresis(
        "Drug", dose_mg=10.0, route="iontophoresis", current_mA=1.0, charge_type="cationic"
    )
    passive = simulate_iontophoresis("Drug", dose_mg=10.0, route="passive", current_mA=0.0)
    assert ionto.auc_plasma_mg_h_per_L > passive.auc_plasma_mg_h_per_L


def test_cationic_enhancement_greater_than_neutral():
    cat = simulate_iontophoresis(
        "Drug", dose_mg=10.0, route="iontophoresis", current_mA=1.0, charge_type="cationic"
    )
    neu = simulate_iontophoresis(
        "Drug", dose_mg=10.0, route="iontophoresis", current_mA=1.0, charge_type="neutral"
    )
    assert cat.enhancement_factor > neu.enhancement_factor


def test_higher_current_higher_enhancement_factor():
    low = simulate_iontophoresis(
        "Drug", dose_mg=10.0, route="iontophoresis", current_mA=0.5, charge_type="cationic"
    )
    high = simulate_iontophoresis(
        "Drug", dose_mg=10.0, route="iontophoresis", current_mA=2.0, charge_type="cationic"
    )
    assert high.enhancement_factor > low.enhancement_factor


# ── Amount compartment behaviour ─────────────────────────────────────────────


def test_a_skin_decreases_monotonically():
    result = simulate_iontophoresis("Drug", dose_mg=10.0, t_end_h=12.0, dt_h=0.5)
    skin = result.a_skin_mg
    for i in range(1, len(skin)):
        assert skin[i] <= skin[i - 1] + 1e-9, (
            f"a_skin not monotonically decreasing at step {i}: {skin[i - 1]} -> {skin[i]}"
        )


# ── compare_modes ────────────────────────────────────────────────────────────


def test_compare_modes_returns_two_results():
    results = compare_modes("Drug", dose_mg=10.0, current_mA=1.0, charge_type="cationic")
    assert len(results) == 2


def test_compare_modes_sorted_by_auc_descending():
    results = compare_modes("Drug", dose_mg=10.0, current_mA=1.0, charge_type="cationic")
    assert results[0].auc_plasma_mg_h_per_L >= results[1].auc_plasma_mg_h_per_L


def test_compare_modes_iontophoresis_first():
    results = compare_modes("Drug", dose_mg=10.0, current_mA=1.0, charge_type="cationic")
    assert results[0].route == "iontophoresis"


def test_compare_modes_passive_second():
    results = compare_modes("Drug", dose_mg=10.0, current_mA=1.0, charge_type="cationic")
    assert results[1].route == "passive"


def test_compare_modes_both_correct_route_fields():
    results = compare_modes("Drug", dose_mg=10.0, current_mA=0.5, charge_type="anionic")
    routes = {r.route for r in results}
    assert routes == {"iontophoresis", "passive"}


# ── Linearity / dose proportionality ─────────────────────────────────────────


def test_double_dose_approximately_doubles_cmax():
    r1 = simulate_iontophoresis("Drug", dose_mg=10.0, route="passive")
    r2 = simulate_iontophoresis("Drug", dose_mg=20.0, route="passive")
    ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
    assert ratio == pytest.approx(2.0, rel=0.05)


# ── Validation errors ─────────────────────────────────────────────────────────


def test_validation_dose_mg_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_iontophoresis("Drug", dose_mg=0.0)


def test_validation_dose_mg_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_iontophoresis("Drug", dose_mg=-5.0)


def test_validation_current_mA_negative_raises():
    with pytest.raises(ValueError, match="current_mA"):
        simulate_iontophoresis("Drug", dose_mg=10.0, current_mA=-1.0)


def test_validation_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_iontophoresis("Drug", dose_mg=10.0, route="injection")


def test_validation_invalid_charge_type_raises():
    with pytest.raises(ValueError, match="charge_type"):
        simulate_iontophoresis("Drug", dose_mg=10.0, charge_type="zwitterion")


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_iontophoresis("Drug", dose_mg=10.0, cl_L_per_h=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_iontophoresis("Drug", dose_mg=10.0, vd_L=0.0)
