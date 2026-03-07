"""Tests for enterohepatic circulation PK model — Phase 256."""

from __future__ import annotations

import pytest

from omega_pbpk.core.enterohepatic_circulation import (
    EHCResult,
    simulate_ehc,
)

# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ehc("Drug", dose_mg=0.0, cl_L_per_h=5.0, vd_L=50.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ehc("Drug", dose_mg=-100.0, cl_L_per_h=5.0, vd_L=50.0)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=0.0, vd_L=50.0)


def test_invalid_cl_negative():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=-1.0, vd_L=50.0)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=0.0)


def test_invalid_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=-10.0)


def test_invalid_f_biliary_negative():
    with pytest.raises(ValueError, match="f_biliary"):
        simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_biliary=-0.1)


def test_invalid_f_biliary_one():
    """f_biliary must be strictly < 1."""
    with pytest.raises(ValueError, match="f_biliary"):
        simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_biliary=1.0)


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="sc")


# ---------------------------------------------------------------------------
# IV route tests
# ---------------------------------------------------------------------------


def test_iv_initial_concentration():
    """IV bolus: c_plasma[0] should equal dose / vd."""
    dose, vd = 100.0, 50.0
    result = simulate_ehc("Drug", dose_mg=dose, cl_L_per_h=5.0, vd_L=vd, route="iv")
    assert result.c_plasma_mg_L[0] == pytest.approx(dose / vd, rel=1e-6)


def test_iv_cmax_positive():
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv")
    assert result.cmax > 0.0


def test_iv_auc_positive():
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv")
    assert result.auc > 0.0


# ---------------------------------------------------------------------------
# Oral route tests
# ---------------------------------------------------------------------------


def test_oral_initial_plasma_zero():
    """Oral dose: plasma starts at 0 (drug in gut depot)."""
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="oral")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_oral_cmax_positive():
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="oral")
    assert result.cmax > 0.0


def test_oral_auc_positive():
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="oral")
    assert result.auc > 0.0


# ---------------------------------------------------------------------------
# Array length tests
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    """times_h, c_plasma_mg_L, a_bile_mg, a_gi_ehc_mg must all be the same length."""
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.a_bile_mg) == n
    assert len(result.a_gi_ehc_mg) == n


# ---------------------------------------------------------------------------
# EHC behaviour tests
# ---------------------------------------------------------------------------


def test_no_ehc_secondary_peaks_count():
    """With f_biliary=0, no bile -> typically 0 secondary peaks."""
    result = simulate_ehc(
        "Drug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_biliary=0.0,
        route="iv",
    )
    assert result.n_secondary_peaks == 0


def test_ehc_secondary_peak_nonnegative():
    """secondary_peak field must be >= 0."""
    result = simulate_ehc(
        "Drug",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=50.0,
        f_biliary=0.5,
        ehc_cycle_delay_h=4.0,
        route="iv",
    )
    assert result.secondary_peak >= 0.0


# ---------------------------------------------------------------------------
# Metric sanity tests
# ---------------------------------------------------------------------------


def test_tmax_nonnegative():
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv")
    assert result.tmax_h >= 0.0


def test_t_half_apparent_positive():
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv")
    assert result.t_half_apparent_h > 0.0


def test_drug_name_stored():
    result = simulate_ehc("Morphine", dose_mg=10.0, cl_L_per_h=2.0, vd_L=20.0)
    assert result.drug_name == "Morphine"


def test_route_stored():
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="oral")
    assert result.route == "oral"


# ---------------------------------------------------------------------------
# Linearity test
# ---------------------------------------------------------------------------


def test_dose_linearity_iv():
    """Linear model: doubling dose should approximately double Cmax."""
    r1 = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv")
    r2 = simulate_ehc("Drug", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0, route="iv")
    assert r2.cmax == pytest.approx(r1.cmax * 2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Result type test
# ---------------------------------------------------------------------------


def test_result_is_frozen_dataclass():
    result = simulate_ehc("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert isinstance(result, EHCResult)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_concentrations_non_negative():
    result = simulate_ehc(
        "Drug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_biliary=0.3,
        route="iv",
    )
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)
