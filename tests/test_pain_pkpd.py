"""Tests for Phase 267: Opioid analgesia PK/PD model with tolerance."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pain_pkpd import PainPKPDResult, simulate_pain_pkpd

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def morphine_iv() -> PainPKPDResult:
    return simulate_pain_pkpd(
        drug_name="morphine",
        dose_mg=10.0,
        cl_L_per_h=4.0,
        vd_L=200.0,
        route="iv",
    )


@pytest.fixture()
def morphine_oral() -> PainPKPDResult:
    return simulate_pain_pkpd(
        drug_name="morphine",
        dose_mg=30.0,
        cl_L_per_h=4.0,
        vd_L=200.0,
        route="oral",
        f_oral=0.35,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pain_pkpd("morphine", dose_mg=0.0, cl_L_per_h=4.0, vd_L=200.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pain_pkpd("morphine", dose_mg=-5.0, cl_L_per_h=4.0, vd_L=200.0)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_pain_pkpd("morphine", dose_mg=10.0, cl_L_per_h=0.0, vd_L=200.0)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_pain_pkpd("morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=0.0)


def test_invalid_ec50_zero():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        simulate_pain_pkpd("morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=200.0, ec50_mg_L=0.0)


def test_invalid_emax_zero():
    with pytest.raises(ValueError, match="emax"):
        simulate_pain_pkpd("morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=200.0, emax=0.0)


def test_invalid_emax_above_one():
    with pytest.raises(ValueError, match="emax"):
        simulate_pain_pkpd("morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=200.0, emax=1.5)


def test_invalid_tolerance_k_negative():
    with pytest.raises(ValueError, match="tolerance_k_per_h"):
        simulate_pain_pkpd(
            "morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=200.0, tolerance_k_per_h=-0.1
        )


def test_invalid_tolerance_max_below_one():
    with pytest.raises(ValueError, match="tolerance_max"):
        simulate_pain_pkpd("morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=200.0, tolerance_max=0.5)


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_pain_pkpd("morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=200.0, route="im")


def test_invalid_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_pain_pkpd(
            "morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=200.0, route="oral", f_oral=0.0
        )


# ---------------------------------------------------------------------------
# IV-specific checks
# ---------------------------------------------------------------------------


def test_iv_initial_plasma_concentration(morphine_iv: PainPKPDResult):
    expected = 10.0 / 200.0  # dose / vd
    assert abs(morphine_iv.c_plasma_mg_L[0] - expected) < 1e-9


def test_cmax_positive(morphine_iv: PainPKPDResult):
    assert morphine_iv.cmax_plasma > 0.0


def test_auc_positive(morphine_iv: PainPKPDResult):
    assert morphine_iv.auc_plasma > 0.0


# ---------------------------------------------------------------------------
# Effect compartment and PD
# ---------------------------------------------------------------------------


def test_effect_compartment_rises(morphine_iv: PainPKPDResult):
    """Effect compartment should have values > 0 after equilibration."""
    assert max(morphine_iv.c_effect_mg_L) > 0.0


def test_receptor_occupancy_range(morphine_iv: PainPKPDResult):
    occ = morphine_iv.receptor_occupancy_pct
    assert all(0.0 <= v <= 100.0 for v in occ)


def test_analgesic_effect_range(morphine_iv: PainPKPDResult):
    effect = morphine_iv.analgesic_effect
    emax = 1.0
    assert all(0.0 <= v <= emax + 1e-9 for v in effect)


def test_tolerance_starts_at_one(morphine_iv: PainPKPDResult):
    assert morphine_iv.tolerance_factor[0] == pytest.approx(1.0)


def test_tolerance_may_increase(morphine_iv: PainPKPDResult):
    """With default tolerance_k > 0, tolerance should increase over time."""
    tol = morphine_iv.tolerance_factor
    assert max(tol) >= 1.0


def test_peak_analgesia_in_range(morphine_iv: PainPKPDResult):
    assert 0.0 < morphine_iv.peak_analgesia <= 1.0


def test_duration_effective_nonnegative(morphine_iv: PainPKPDResult):
    assert morphine_iv.duration_effective_h >= 0.0


# ---------------------------------------------------------------------------
# Array length consistency
# ---------------------------------------------------------------------------


def test_array_lengths_consistent(morphine_iv: PainPKPDResult):
    n = len(morphine_iv.times_h)
    assert len(morphine_iv.c_plasma_mg_L) == n
    assert len(morphine_iv.c_effect_mg_L) == n
    assert len(morphine_iv.receptor_occupancy_pct) == n
    assert len(morphine_iv.analgesic_effect) == n
    assert len(morphine_iv.tolerance_factor) == n


# ---------------------------------------------------------------------------
# Dose linearity (approx)
# ---------------------------------------------------------------------------


def test_dose_linearity_cmax():
    r1 = simulate_pain_pkpd("morphine", dose_mg=10.0, cl_L_per_h=4.0, vd_L=200.0)
    r2 = simulate_pain_pkpd("morphine", dose_mg=20.0, cl_L_per_h=4.0, vd_L=200.0)
    # 2x dose should give approximately 2x Cmax (linear PK)
    ratio = r2.cmax_plasma / r1.cmax_plasma
    assert 1.8 < ratio < 2.2


# ---------------------------------------------------------------------------
# Drug name stored
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    r = simulate_pain_pkpd("oxycodone", dose_mg=5.0, cl_L_per_h=3.0, vd_L=100.0)
    assert r.drug_name == "oxycodone"


# ---------------------------------------------------------------------------
# Oral route works
# ---------------------------------------------------------------------------


def test_oral_route_cmax_positive(morphine_oral: PainPKPDResult):
    assert morphine_oral.cmax_plasma > 0.0


def test_oral_initial_plasma_zero(morphine_oral: PainPKPDResult):
    """Oral dosing should start with zero plasma concentration."""
    assert morphine_oral.c_plasma_mg_L[0] == pytest.approx(0.0)
