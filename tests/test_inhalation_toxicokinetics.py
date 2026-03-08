"""Tests for inhalation_toxicokinetics — Phase 1025."""

import pytest

from omega_pbpk.core.inhalation_toxicokinetics import (
    InhalationToxResult,
    simulate_inhalation_toxicokinetics,
)

# ---------------------------------------------------------------------------
# Basic return type and structural tests
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    assert isinstance(result, InhalationToxResult)


def test_initial_blood_concentration_zero():
    result = simulate_inhalation_toxicokinetics("benzene", exposure_ppm=50.0, mw_Da=78.0)
    assert result.c_blood_mg_L[0] == 0.0


def test_initial_tissue_concentration_zero():
    result = simulate_inhalation_toxicokinetics("xylene", exposure_ppm=50.0, mw_Da=106.0)
    assert result.c_tissue_mg_L[0] == 0.0


def test_blood_increases_during_exposure():
    result = simulate_inhalation_toxicokinetics(
        "styrene", exposure_ppm=200.0, mw_Da=104.0, exposure_duration_h=4.0, t_end_h=10.0
    )
    # Blood should rise during first few steps
    assert result.c_blood_mg_L[5] > result.c_blood_mg_L[0]


def test_blood_eventually_decreases_after_exposure():
    result = simulate_inhalation_toxicokinetics(
        "acetone", exposure_ppm=500.0, mw_Da=58.0, exposure_duration_h=4.0, t_end_h=20.0
    )
    # Find value just after exposure end and at end
    end_val = result.c_blood_mg_L[-1]
    peak_val = result.cmax_blood
    assert end_val < peak_val


# ---------------------------------------------------------------------------
# AUC, Cmax, Tmax
# ---------------------------------------------------------------------------


def test_auc_blood_positive():
    result = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    assert result.auc_blood > 0.0


def test_auc_tissue_positive():
    result = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    assert result.auc_tissue > 0.0


def test_cmax_blood_positive():
    result = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    assert result.cmax_blood > 0.0


def test_tmax_blood_nonnegative():
    result = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    assert result.tmax_blood_h >= 0.0


def test_clearance_t50_nonnegative():
    result = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    assert result.clearance_t50_h >= 0.0


# ---------------------------------------------------------------------------
# Dose-response consistency
# ---------------------------------------------------------------------------


def test_higher_ppm_gives_higher_cmax():
    r_low = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=50.0, mw_Da=92.0)
    r_high = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=500.0, mw_Da=92.0)
    assert r_high.cmax_blood > r_low.cmax_blood


def test_higher_ppm_proportional_cmax():
    """Doubling ppm should approximately double cmax_blood (linear system)."""
    r_low = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    r_high = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=200.0, mw_Da=92.0)
    ratio = r_high.cmax_blood / r_low.cmax_blood
    assert 1.5 < ratio < 2.5


def test_higher_blood_air_partition_increases_blood_conc():
    """Very low vs very high BAP: higher partition → blood acts as bigger sink → higher Cmax."""
    r_low = simulate_inhalation_toxicokinetics(
        "compound_x",
        exposure_ppm=100.0,
        mw_Da=100.0,
        blood_air_partition=0.5,
        cardiac_output_L_h=10.0,
        cl_metabolic_L_per_h=0.1,
        vd_tissue_L=5.0,
    )
    r_high = simulate_inhalation_toxicokinetics(
        "compound_x",
        exposure_ppm=100.0,
        mw_Da=100.0,
        blood_air_partition=100.0,
        cardiac_output_L_h=10.0,
        cl_metabolic_L_per_h=0.1,
        vd_tissue_L=5.0,
    )
    assert r_high.cmax_blood > r_low.cmax_blood


def test_longer_exposure_increases_auc():
    r_short = simulate_inhalation_toxicokinetics(
        "toluene", exposure_ppm=100.0, mw_Da=92.0, exposure_duration_h=2.0, t_end_h=16.0
    )
    r_long = simulate_inhalation_toxicokinetics(
        "toluene", exposure_ppm=100.0, mw_Da=92.0, exposure_duration_h=8.0, t_end_h=16.0
    )
    assert r_long.auc_blood > r_short.auc_blood


# ---------------------------------------------------------------------------
# Tissue compartment
# ---------------------------------------------------------------------------


def test_tissue_peak_positive():
    result = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    assert max(result.c_tissue_mg_L) > 0.0


# ---------------------------------------------------------------------------
# Alveolar concentration
# ---------------------------------------------------------------------------


def test_alveolar_conc_during_exposure():
    result = simulate_inhalation_toxicokinetics(
        "toluene", exposure_ppm=100.0, mw_Da=92.0, exposure_duration_h=4.0, t_end_h=12.0
    )
    # During exposure window the alveolar concentration should be non-zero
    exposure_steps = [c for t, c in zip(result.times_h, result.c_alveolar_mg_L) if t <= 4.0]
    assert all(c > 0.0 for c in exposure_steps)


def test_alveolar_conc_zero_after_exposure():
    result = simulate_inhalation_toxicokinetics(
        "toluene", exposure_ppm=100.0, mw_Da=92.0, exposure_duration_h=4.0, t_end_h=12.0
    )
    post_steps = [c for t, c in zip(result.times_h, result.c_alveolar_mg_L) if t > 4.0]
    assert all(c == 0.0 for c in post_steps)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_inhalation_toxicokinetics("toluene", exposure_ppm=100.0, mw_Da=92.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_ppm_zero():
    with pytest.raises(ValueError, match="exposure_ppm"):
        simulate_inhalation_toxicokinetics("gas", exposure_ppm=0.0, mw_Da=50.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        simulate_inhalation_toxicokinetics("gas", exposure_ppm=100.0, mw_Da=0.0)


def test_validation_t_end_too_small():
    with pytest.raises(ValueError, match="t_end_h"):
        simulate_inhalation_toxicokinetics(
            "gas",
            exposure_ppm=100.0,
            mw_Da=100.0,
            exposure_duration_h=8.0,
            t_end_h=8.0,  # must be > exposure_duration_h
        )
