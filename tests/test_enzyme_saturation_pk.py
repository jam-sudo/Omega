"""Tests for Phase 640 — clinical/enzyme_saturation_pk.py."""

import pytest

from omega_pbpk.clinical.enzyme_saturation_pk import (
    EnzymeSaturationResult,
    dose_escalation_mm,
    simulate_mm_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iv_low_dose():
    """Low dose: Cmax << Km → linear elimination."""
    return simulate_mm_pk(
        drug_name="TestDrug",
        dose_mg=10.0,  # Cmax = 10/50 = 0.2 mg/L << Km=10
        vmax_mg_per_h=20.0,
        km_mg_L=10.0,
        vd_L=50.0,
        route="iv",
        t_end_h=48.0,
        dt_h=0.05,
    )


@pytest.fixture
def iv_high_dose():
    """High dose: Cmax >> Km → saturable elimination."""
    return simulate_mm_pk(
        drug_name="TestDrug",
        dose_mg=5000.0,  # Cmax = 5000/50 = 100 mg/L >> Km=10
        vmax_mg_per_h=20.0,
        km_mg_L=10.0,
        vd_L=50.0,
        route="iv",
        t_end_h=240.0,
        dt_h=0.05,
    )


@pytest.fixture
def oral_result():
    """Oral route simulation."""
    return simulate_mm_pk(
        drug_name="OralDrug",
        dose_mg=200.0,
        vmax_mg_per_h=15.0,
        km_mg_L=5.0,
        vd_L=30.0,
        route="oral",
        ka_per_h=1.5,
        f_oral=0.9,
        t_end_h=48.0,
        dt_h=0.05,
    )


# ---------------------------------------------------------------------------
# Return type and field types
# ---------------------------------------------------------------------------


def test_returns_enzyme_saturation_result(iv_low_dose):
    assert isinstance(iv_low_dose, EnzymeSaturationResult)


def test_field_types(iv_low_dose):
    r = iv_low_dose
    assert isinstance(r.drug_name, str)
    assert isinstance(r.dose_mg, float)
    assert isinstance(r.vmax_mg_per_h, float)
    assert isinstance(r.km_mg_L, float)
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_plasma_mg_L, list)
    assert isinstance(r.cmax_mg_L, float)
    assert isinstance(r.tmax_h, float)
    assert isinstance(r.auc_mg_h_per_L, float)
    assert isinstance(r.t_half_effective_h, float)
    assert isinstance(r.dose_proportionality_index, float)
    assert isinstance(r.saturation_fraction, float)
    assert isinstance(r.elimination_mode, str)
    assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_cmax_positive(iv_low_dose):
    assert iv_low_dose.cmax_mg_L > 0.0


def test_auc_positive(iv_low_dose):
    assert iv_low_dose.auc_mg_h_per_L > 0.0


def test_saturation_fraction_in_0_1(iv_low_dose, iv_high_dose):
    assert 0.0 <= iv_low_dose.saturation_fraction <= 1.0
    assert 0.0 <= iv_high_dose.saturation_fraction <= 1.0


def test_elimination_mode_valid_values(iv_low_dose, iv_high_dose):
    valid = {"linear", "mixed", "saturable"}
    assert iv_low_dose.elimination_mode in valid
    assert iv_high_dose.elimination_mode in valid


def test_t_half_effective_positive(iv_low_dose):
    assert iv_low_dose.t_half_effective_h > 0.0


def test_notes_nonempty(iv_low_dose):
    assert len(iv_low_dose.notes) > 0


# ---------------------------------------------------------------------------
# Kinetic mode classification
# ---------------------------------------------------------------------------


def test_low_dose_elimination_mode_linear(iv_low_dose):
    """Cmax=0.2 mg/L << Km=10 → saturation_fraction < 0.2 → linear."""
    assert iv_low_dose.elimination_mode == "linear"


def test_high_dose_elimination_mode_saturable(iv_high_dose):
    """Cmax=100 mg/L >> Km=10 → saturation_fraction > 0.8 → saturable."""
    assert iv_high_dose.elimination_mode == "saturable"


# ---------------------------------------------------------------------------
# Nonlinearity: 2× dose at saturation → >2× AUC
# ---------------------------------------------------------------------------


def test_nonlinearity_supraproportional_auc():
    """At saturation, 2× dose should give >2× AUC."""
    r1 = simulate_mm_pk(
        "Drug",
        dose_mg=1000.0,
        vmax_mg_per_h=20.0,
        km_mg_L=10.0,
        vd_L=50.0,
        t_end_h=200.0,
        dt_h=0.05,
    )
    r2 = simulate_mm_pk(
        "Drug",
        dose_mg=2000.0,
        vmax_mg_per_h=20.0,
        km_mg_L=10.0,
        vd_L=50.0,
        t_end_h=200.0,
        dt_h=0.05,
    )
    assert r2.auc_mg_h_per_L > 2.0 * r1.auc_mg_h_per_L


def test_dose_proportionality_index_gt_1_at_saturation():
    """At saturating doses, DPI > 1.0 relative to low reference dose."""
    result = simulate_mm_pk(
        "Drug",
        dose_mg=2000.0,
        vmax_mg_per_h=20.0,
        km_mg_L=10.0,
        vd_L=50.0,
        t_end_h=200.0,
        dt_h=0.05,
        reference_dose_mg=10.0,
    )
    assert result.dose_proportionality_index > 1.0


def test_saturation_fraction_gt_08_very_high_dose(iv_high_dose):
    assert iv_high_dose.saturation_fraction > 0.8


# ---------------------------------------------------------------------------
# dose_escalation_mm
# ---------------------------------------------------------------------------


def test_dose_escalation_returns_correct_length():
    doses = [100.0, 200.0, 500.0]
    results = dose_escalation_mm("Drug", doses, vmax_mg_per_h=20.0, km_mg_L=5.0, vd_L=40.0)
    assert len(results) == len(doses)


def test_dose_escalation_sorted_ascending():
    doses = [500.0, 100.0, 200.0]
    results = dose_escalation_mm("Drug", doses, vmax_mg_per_h=20.0, km_mg_L=5.0, vd_L=40.0)
    dose_sequence = [r.dose_mg for r in results]
    assert dose_sequence == sorted(dose_sequence)


def test_dose_escalation_higher_dose_higher_cmax():
    doses = [100.0, 500.0, 2000.0]
    results = dose_escalation_mm("Drug", doses, vmax_mg_per_h=20.0, km_mg_L=5.0, vd_L=40.0)
    cmaxes = [r.cmax_mg_L for r in results]
    assert cmaxes[0] < cmaxes[1] < cmaxes[2]


def test_dose_escalation_higher_dose_higher_saturation_fraction():
    doses = [10.0, 500.0, 5000.0]
    results = dose_escalation_mm("Drug", doses, vmax_mg_per_h=20.0, km_mg_L=10.0, vd_L=50.0)
    sats = [r.saturation_fraction for r in results]
    assert sats[0] < sats[1] < sats[2]


# ---------------------------------------------------------------------------
# Route-specific tests
# ---------------------------------------------------------------------------


def test_oral_tmax_positive(oral_result):
    """Oral route: peak occurs after t=0."""
    assert oral_result.tmax_h > 0.0


def test_iv_cmax_equals_dose_over_vd():
    """IV route: initial concentration = dose / Vd."""
    dose = 500.0
    vd = 50.0
    result = simulate_mm_pk(
        "Drug",
        dose_mg=dose,
        vmax_mg_per_h=20.0,
        km_mg_L=5.0,
        vd_L=vd,
        route="iv",
        t_end_h=0.0,
        dt_h=0.05,
    )
    expected_c0 = dose / vd
    assert abs(result.cmax_mg_L - expected_c0) < 0.01 * expected_c0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_vmax_zero():
    with pytest.raises(ValueError):
        simulate_mm_pk("Drug", dose_mg=100.0, vmax_mg_per_h=0.0, km_mg_L=5.0)


def test_validation_km_zero():
    with pytest.raises(ValueError):
        simulate_mm_pk("Drug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_L=0.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError):
        simulate_mm_pk("Drug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_L=5.0, route="im")


def test_validation_f_oral_too_high():
    with pytest.raises(ValueError):
        simulate_mm_pk(
            "Drug", dose_mg=100.0, vmax_mg_per_h=10.0, km_mg_L=5.0, route="oral", f_oral=1.5
        )
