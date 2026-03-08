"""Tests for Phase 858: Enzyme Saturation PK Simulation."""

import pytest

from omega_pbpk.core.enzyme_saturation_pk import (
    EnzymeSaturationPKResult,
    compare_doses_nonlinear,
    simulate_enzyme_saturation_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and field preservation
# ---------------------------------------------------------------------------


def test_returns_correct_type_iv():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0, route="iv"
    )
    assert isinstance(result, EnzymeSaturationPKResult)


def test_returns_correct_type_oral():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0, route="oral"
    )
    assert isinstance(result, EnzymeSaturationPKResult)


def test_field_preservation_drug_name():
    result = simulate_enzyme_saturation_pk(
        "Ibuprofen", dose_mg=200.0, vmax_mg_per_h=20.0, km_mg_per_L=10.0
    )
    assert result.drug_name == "Ibuprofen"


def test_field_preservation_dose_mg():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=500.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0
    )
    assert result.dose_mg == 500.0


def test_field_preservation_route_iv():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0, route="iv"
    )
    assert result.route == "iv"


def test_field_preservation_route_oral():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0, route="oral"
    )
    assert result.route == "oral"


def test_field_preservation_vmax():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=15.0, km_mg_per_L=5.0
    )
    assert result.vmax_mg_per_h == 15.0


def test_field_preservation_km():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=7.5
    )
    assert result.km_mg_per_L == 7.5


def test_notes_is_string():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0
    )
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Output plausibility
# ---------------------------------------------------------------------------


def test_cmax_positive_iv():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0, route="iv"
    )
    assert result.cmax_mg_L > 0


def test_auc_positive_iv():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0, route="iv"
    )
    assert result.auc_mg_h_per_L > 0


def test_cmax_positive_oral():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0, route="oral"
    )
    assert result.cmax_mg_L > 0


def test_auc_positive_oral():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0, route="oral"
    )
    assert result.auc_mg_h_per_L > 0


def test_times_h_is_list():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0
    )
    assert isinstance(result.times_h, list)
    assert len(result.times_h) > 0


def test_c_plasma_mg_L_is_list():
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0
    )
    assert isinstance(result.c_plasma_mg_L, list)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_t_half_terminal_positive():
    result = simulate_enzyme_saturation_pk(
        "TestDrug",
        dose_mg=100.0,
        vmax_mg_per_h=10.0,
        km_mg_per_L=5.0,
        t_end_h=48.0,
    )
    assert result.t_half_terminal_h > 0


# ---------------------------------------------------------------------------
# Enzyme saturation behaviour
# ---------------------------------------------------------------------------


def test_high_dose_saturation_close_to_1():
    """High dose >> Km*Vd: enzymes should be nearly fully saturated at Cmax."""
    # Vd=10 L, Km=1 mg/L -> Km*Vd = 10 mg. Dose = 1000 mg >> 10 mg.
    result = simulate_enzyme_saturation_pk(
        "TestDrug",
        dose_mg=1000.0,
        vmax_mg_per_h=5.0,
        km_mg_per_L=1.0,
        route="iv",
        vd_L=10.0,
    )
    assert result.saturation_fraction_at_cmax > 0.9


def test_low_dose_saturation_low():
    """Low dose << Km*Vd: minimal enzyme saturation at Cmax."""
    # Vd=10 L, Km=100 mg/L -> Km*Vd = 1000 mg. Dose = 1 mg << 1000 mg.
    result = simulate_enzyme_saturation_pk(
        "TestDrug",
        dose_mg=1.0,
        vmax_mg_per_h=5.0,
        km_mg_per_L=100.0,
        route="iv",
        vd_L=10.0,
    )
    assert result.saturation_fraction_at_cmax < 0.1


def test_auc_disproportional_with_dose():
    """2x dose under saturation -> AUC ratio > 2 (dose-disproportional)."""
    r_low = simulate_enzyme_saturation_pk(
        "TestDrug",
        dose_mg=50.0,
        vmax_mg_per_h=5.0,
        km_mg_per_L=2.0,
        route="iv",
        vd_L=10.0,
        t_end_h=48.0,
    )
    r_high = simulate_enzyme_saturation_pk(
        "TestDrug",
        dose_mg=100.0,
        vmax_mg_per_h=5.0,
        km_mg_per_L=2.0,
        route="iv",
        vd_L=10.0,
        t_end_h=48.0,
    )
    auc_ratio = r_high.auc_mg_h_per_L / r_low.auc_mg_h_per_L
    assert auc_ratio > 2.0, f"Expected AUC ratio > 2.0, got {auc_ratio:.3f}"


def test_dose_proportional_always_false():
    """dose_proportional is always False for Michaelis-Menten PK."""
    result = simulate_enzyme_saturation_pk(
        "TestDrug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_per_L=5.0
    )
    assert result.dose_proportional is False


# ---------------------------------------------------------------------------
# Low dose near first-order behaviour
# ---------------------------------------------------------------------------


def test_low_dose_auc_approximates_first_order():
    """Low dose: AUC ≈ dose * Km / Vmax (first-order limit).

    For IV 1-cpt with pure MM elimination at low C:
      AUC ~ (dose/Vd) / (Vmax/Km / Vd) = dose * Km / Vmax

    Use high Vmax to ensure fast elimination (short t1/2) so simulation
    captures most of the AUC within a feasible t_end.
    """
    dose = 0.5  # mg
    vmax = 50.0  # mg/h  (high Vmax -> fast elimination, short t1/2)
    km = 50.0  # mg/L
    vd = 10.0  # L
    # ke_approx = Vmax/(Km*Vd) = 50/(50*10) = 0.1/h -> t1/2 ~ 7h
    result = simulate_enzyme_saturation_pk(
        "TestDrug",
        dose_mg=dose,
        vmax_mg_per_h=vmax,
        km_mg_per_L=km,
        route="iv",
        vd_L=vd,
        t_end_h=72.0,
    )
    expected_auc = dose * km / vmax
    ratio = result.auc_mg_h_per_L / expected_auc
    assert 0.8 <= ratio <= 1.2, f"Low-dose AUC ratio vs first-order: {ratio:.3f}"


# ---------------------------------------------------------------------------
# IV vs oral route
# ---------------------------------------------------------------------------


def test_iv_cmax_at_early_time():
    """IV bolus: Cmax should occur at or near t=0."""
    result = simulate_enzyme_saturation_pk(
        "TestDrug",
        dose_mg=100.0,
        vmax_mg_per_h=5.0,
        km_mg_per_L=50.0,
        route="iv",
        vd_L=10.0,
        dt_h=0.05,
    )
    assert result.tmax_h < 0.5  # within first time step


def test_oral_tmax_positive():
    """Oral route: tmax > 0 due to absorption phase."""
    result = simulate_enzyme_saturation_pk(
        "TestDrug",
        dose_mg=100.0,
        vmax_mg_per_h=5.0,
        km_mg_per_L=50.0,
        route="oral",
        vd_L=10.0,
        ka_per_h=1.0,
    )
    assert result.tmax_h > 0


# ---------------------------------------------------------------------------
# compare_doses_nonlinear
# ---------------------------------------------------------------------------


def test_compare_doses_returns_list():
    results = compare_doses_nonlinear(
        "TestDrug",
        doses_mg=[50.0, 100.0, 200.0],
        vmax_mg_per_h=10.0,
        km_mg_per_L=5.0,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_doses_sorted_ascending():
    """Results are sorted by dose_mg ascending."""
    results = compare_doses_nonlinear(
        "TestDrug",
        doses_mg=[200.0, 50.0, 100.0],
        vmax_mg_per_h=10.0,
        km_mg_per_L=5.0,
    )
    doses = [r.dose_mg for r in results]
    assert doses == sorted(doses)


def test_compare_doses_all_results_valid():
    results = compare_doses_nonlinear(
        "TestDrug",
        doses_mg=[10.0, 100.0],
        vmax_mg_per_h=5.0,
        km_mg_per_L=5.0,
    )
    for r in results:
        assert r.cmax_mg_L > 0
        assert r.auc_mg_h_per_L > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_enzyme_saturation_pk("TestDrug", 0.0, 10.0, 5.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_enzyme_saturation_pk("TestDrug", -50.0, 10.0, 5.0)


def test_validation_vmax_zero():
    with pytest.raises(ValueError, match="vmax_mg_per_h must be > 0"):
        simulate_enzyme_saturation_pk("TestDrug", 100.0, 0.0, 5.0)


def test_validation_km_zero():
    with pytest.raises(ValueError, match="km_mg_per_L must be > 0"):
        simulate_enzyme_saturation_pk("TestDrug", 100.0, 10.0, 0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        simulate_enzyme_saturation_pk("TestDrug", 100.0, 10.0, 5.0, vd_L=0.0)


def test_validation_route_invalid():
    with pytest.raises(ValueError, match="route must be 'iv' or 'oral'"):
        simulate_enzyme_saturation_pk("TestDrug", 100.0, 10.0, 5.0, route="subcutaneous")
