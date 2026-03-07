"""Tests for Phase 852: Drug release from liposomal formulations.

Covers:
  - Return type is LiposomeReleasePKResult
  - c_encapsulated decreases monotonically over time
  - c_released starts at 0, increases then decreases
  - PEGylated liposomes have a longer t_half than conventional
  - encapsulated_fraction_at_24h is in [0, 1]
  - compare_liposome_types returns exactly 4 results
  - release_mechanism is a valid value
  - Validation errors for invalid inputs
"""

from __future__ import annotations

import pytest

from omega_pbpk.core.liposome_release_pk import (
    LiposomeReleasePKResult,
    compare_liposome_types,
    simulate_liposome_release,
)

# ---------------------------------------------------------------------------
# Valid release mechanism values
# ---------------------------------------------------------------------------

_VALID_MECHANISMS = {"passive_diffusion", "triggered", "pH_triggered"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(liposome_type: str = "conventional", **kwargs) -> LiposomeReleasePKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        liposome_type=liposome_type,
        t_end_h=72.0,
        dt_h=0.5,
    )
    defaults.update(kwargs)
    return simulate_liposome_release(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _sim()
    assert isinstance(result, LiposomeReleasePKResult)


# ---------------------------------------------------------------------------
# c_encapsulated decreases monotonically
# ---------------------------------------------------------------------------


def test_c_encapsulated_decreases_monotonically():
    result = _sim(liposome_type="conventional", dt_h=0.1)
    enc = result.c_encapsulated_mg_L
    for i in range(1, len(enc)):
        assert enc[i] <= enc[i - 1] + 1e-9, f"Not monotone at step {i}"


def test_c_encapsulated_decreases_pegylated():
    result = _sim(liposome_type="pegylated", dt_h=0.1)
    enc = result.c_encapsulated_mg_L
    for i in range(1, len(enc)):
        assert enc[i] <= enc[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# c_released starts at 0, increases then decreases
# ---------------------------------------------------------------------------


def test_c_released_starts_at_zero():
    result = _sim()
    assert result.c_released_mg_L[0] == pytest.approx(0.0)


def test_c_released_has_peak_then_declines():
    """Free drug should rise to a peak and then fall back."""
    result = _sim(liposome_type="conventional", dt_h=0.1)
    rel = result.c_released_mg_L
    cmax_idx = rel.index(max(rel))
    # At least one step after the peak it should decrease
    assert cmax_idx > 0, "Peak should not be at t=0"
    assert rel[-1] < rel[cmax_idx], "Drug should decline after peak"


# ---------------------------------------------------------------------------
# PEGylated t_half > conventional
# ---------------------------------------------------------------------------


def test_pegylated_longer_t_half_than_conventional():
    conv = _sim(liposome_type="conventional")
    peg = _sim(liposome_type="pegylated")
    assert peg.t_half_liposome_h > conv.t_half_liposome_h


# ---------------------------------------------------------------------------
# encapsulated_fraction_at_24h in [0, 1]
# ---------------------------------------------------------------------------


def test_encapsulated_fraction_at_24h_in_range():
    for ltype in ("conventional", "pegylated", "thermosensitive", "ph_sensitive"):
        result = _sim(liposome_type=ltype)
        frac = result.encapsulated_fraction_at_24h
        assert 0.0 <= frac <= 1.0, f"Fraction out of range for {ltype}: {frac}"


def test_pegylated_higher_encapsulated_fraction_at_24h():
    """PEGylated should retain more drug at 24 h than conventional."""
    conv = _sim(liposome_type="conventional")
    peg = _sim(liposome_type="pegylated")
    assert peg.encapsulated_fraction_at_24h > conv.encapsulated_fraction_at_24h


# ---------------------------------------------------------------------------
# compare_liposome_types
# ---------------------------------------------------------------------------


def test_compare_liposome_types_returns_four():
    results = compare_liposome_types(drug_name="Drug", dose_mg=10.0)
    assert len(results) == 4


def test_compare_liposome_types_sorted_by_auc():
    results = compare_liposome_types(drug_name="Drug", dose_mg=10.0)
    aucs = [r.auc_free_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_liposome_types_all_types_present():
    results = compare_liposome_types(drug_name="Drug", dose_mg=10.0)
    types_found = {r.liposome_type for r in results}
    assert types_found == {"conventional", "pegylated", "thermosensitive", "ph_sensitive"}


# ---------------------------------------------------------------------------
# release_mechanism valid values
# ---------------------------------------------------------------------------


def test_release_mechanism_conventional():
    result = _sim(liposome_type="conventional")
    assert result.release_mechanism in _VALID_MECHANISMS


def test_release_mechanism_thermosensitive():
    result = _sim(liposome_type="thermosensitive")
    assert result.release_mechanism == "triggered"


def test_release_mechanism_ph_sensitive():
    result = _sim(liposome_type="ph_sensitive")
    assert result.release_mechanism == "pH_triggered"


def test_release_mechanism_pegylated():
    result = _sim(liposome_type="pegylated")
    assert result.release_mechanism == "passive_diffusion"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def test_cmax_free_positive():
    result = _sim()
    assert result.cmax_free_mg_L > 0.0


def test_tmax_free_positive():
    result = _sim()
    assert result.tmax_free_h > 0.0


def test_auc_free_positive():
    result = _sim()
    assert result.auc_free_mg_h_per_L > 0.0


def test_auc_total_greater_than_or_equal_auc_free():
    result = _sim()
    assert result.auc_total_mg_h_per_L >= result.auc_free_mg_h_per_L - 1e-6


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_liposome_release(drug_name="D", dose_mg=0.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_liposome_release(drug_name="D", dose_mg=-5.0)


def test_error_cl_free_zero():
    with pytest.raises(ValueError, match="cl_free_L_per_h"):
        simulate_liposome_release(drug_name="D", dose_mg=10.0, cl_free_L_per_h=0.0)


def test_error_vd_free_zero():
    with pytest.raises(ValueError, match="vd_free_L"):
        simulate_liposome_release(drug_name="D", dose_mg=10.0, vd_free_L=0.0)


def test_error_invalid_liposome_type():
    with pytest.raises(ValueError, match="liposome_type"):
        simulate_liposome_release(drug_name="D", dose_mg=10.0, liposome_type="nanoparticle")
