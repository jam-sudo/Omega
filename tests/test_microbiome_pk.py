"""Tests for Phase 632 — Microbiome PK."""

import pytest

from omega_pbpk.core.microbiome_pk import (
    MicrobiomePKResult,
    estimate_microbial_clearance,
    gut_lumen_pk,
    simulate_microbiome_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    gut_transit_time_h=24.0,
    microbial_cl_per_h=0.5,
    f_bioactivated=0.1,
    has_nitro=False,
    has_azo=False,
    has_glucuronide_conjugate=False,
    absorption_fraction=0.3,
    logP=2.0,
    colonocyte_clint_mL_per_min=0.0,
)


@pytest.fixture
def result():
    return simulate_microbiome_pk(**BASE_KWARGS)


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_result(result):
    assert isinstance(result, MicrobiomePKResult)


def test_microbial_metabolism_in_0_100(result):
    assert 0.0 <= result.microbiome_metabolism_pct <= 100.0


def test_bioactivation_in_0_100(result):
    assert 0.0 <= result.bioactivation_pct <= 100.0


def test_gut_exposure_positive(result):
    assert result.gut_exposure_mg_h > 0.0


def test_systemic_reduction_nonneg(result):
    assert result.systemic_reduction_pct >= 0.0


def test_gut_half_life_positive(result):
    assert result.gut_half_life_h > 0.0


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------


def test_nitroreductase_risk_with_nitro():
    r = simulate_microbiome_pk(**{**BASE_KWARGS, "has_nitro": True})
    assert r.nitroreductase_risk is True


def test_azoreductase_risk_with_azo():
    r = simulate_microbiome_pk(**{**BASE_KWARGS, "has_azo": True})
    assert r.azoreductase_risk is True


def test_glucuronidase_risk():
    r = simulate_microbiome_pk(**{**BASE_KWARGS, "has_glucuronide_conjugate": True})
    assert r.glucuronidase_risk is True


# ---------------------------------------------------------------------------
# Enterotype sensitivity
# ---------------------------------------------------------------------------


def test_enterotype_high_for_nitro():
    r = simulate_microbiome_pk(**{**BASE_KWARGS, "has_nitro": True})
    assert r.enterotype_sensitivity == "high"


def test_enterotype_moderate_for_glucuronide():
    r = simulate_microbiome_pk(**{**BASE_KWARGS, "has_glucuronide_conjugate": True})
    assert r.enterotype_sensitivity == "moderate"


def test_enterotype_low_for_clean():
    r = simulate_microbiome_pk(**BASE_KWARGS)
    assert r.enterotype_sensitivity == "low"


# ---------------------------------------------------------------------------
# Dose/parameter sensitivity
# ---------------------------------------------------------------------------


def test_high_microbial_cl_more_metabolism():
    r_low = simulate_microbiome_pk(**{**BASE_KWARGS, "microbial_cl_per_h": 0.1})
    r_high = simulate_microbiome_pk(**{**BASE_KWARGS, "microbial_cl_per_h": 2.0})
    assert r_high.microbiome_metabolism_pct > r_low.microbiome_metabolism_pct


def test_long_transit_more_metabolism():
    r_short = simulate_microbiome_pk(**{**BASE_KWARGS, "gut_transit_time_h": 6.0})
    r_long = simulate_microbiome_pk(**{**BASE_KWARGS, "gut_transit_time_h": 72.0})
    assert r_long.microbiome_metabolism_pct > r_short.microbiome_metabolism_pct


# ---------------------------------------------------------------------------
# gut_lumen_pk
# ---------------------------------------------------------------------------


def test_gut_lumen_pk_returns_dict():
    result = gut_lumen_pk(100.0, 24.0, 0.5)
    assert isinstance(result, dict)


def test_gut_lumen_pk_keys():
    result = gut_lumen_pk(100.0, 24.0, 0.5)
    for key in ("auc", "c_max", "t_half", "remaining_mg"):
        assert key in result


def test_gut_lumen_pk_cmax_equals_dose_per_V():
    # V_gut = 1 L
    result = gut_lumen_pk(200.0, 24.0, 0.5)
    assert abs(result["c_max"] - 200.0) < 1e-9


def test_gut_lumen_remaining_decreases():
    result = gut_lumen_pk(100.0, 24.0, 0.5)
    assert result["remaining_mg"] < 100.0


# ---------------------------------------------------------------------------
# estimate_microbial_clearance
# ---------------------------------------------------------------------------


def test_estimate_microbial_cl_positive():
    cl = estimate_microbial_clearance(logP=2.0)
    assert cl > 0.0


def test_nitro_increases_cl():
    cl_base = estimate_microbial_clearance(logP=2.0, has_nitro=False)
    cl_nitro = estimate_microbial_clearance(logP=2.0, has_nitro=True)
    assert cl_nitro > cl_base


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_microbiome_pk(**{**BASE_KWARGS, "dose_mg": -10.0})


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty(result):
    assert len(result.notes) > 0
