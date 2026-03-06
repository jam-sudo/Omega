"""Tests for GI transit and regional absorption model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.gi_transit import (
    GITransitResult,
    compare_food_states,
    simulate_gi_transit,
)

# ---- Helpers ----


def _default_result(**kwargs) -> GITransitResult:
    defaults = dict(drug_name="TestDrug", dose_mg=100.0, logP=2.0, pka=4.0)
    defaults.update(kwargs)
    return simulate_gi_transit(**defaults)


# ---- Tests ----


def test_returns_gi_transit_result():
    r = _default_result()
    assert isinstance(r, GITransitResult)


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_gi_transit("X", dose_mg=0, logP=2.0, pka=4.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_gi_transit("X", 100.0, logP=2.0, pka=4.0, cl_L_per_h=0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_gi_transit("X", 100.0, logP=2.0, pka=4.0, vd_L=0)


def test_validation_solubility_zero():
    with pytest.raises(ValueError, match="solubility"):
        simulate_gi_transit("X", 100.0, logP=2.0, pka=4.0, solubility_mg_mL=0)


def test_fa_total_in_range():
    r = _default_result()
    assert 0.0 <= r.fa_total <= 1.0


def test_cmax_positive():
    r = _default_result()
    assert r.cmax_mg_L > 0


def test_auc_positive():
    r = _default_result()
    assert r.auc_0_t > 0


def test_high_permeability_high_fa():
    """Neutral drug with high logP should absorb well."""
    r = _default_result(logP=3.0, pka=6.0)  # neutral classification
    assert r.fa_total > 0.5


def test_low_solubility_rate_limiting():
    r = _default_result(solubility_mg_mL=0.001)
    assert r.rate_limiting == "solubility"


def test_low_permeability_rate_limiting():
    r = _default_result(logP=-3.0, mw=600.0)
    assert r.rate_limiting == "permeability"


def test_fa_cumulative_non_decreasing():
    r = _default_result()
    for i in range(1, len(r.fa_cumulative)):
        assert r.fa_cumulative[i] >= r.fa_cumulative[i - 1] - 1e-12


def test_fa_cumulative_last_approx_fa_total():
    r = _default_result()
    assert abs(r.fa_cumulative[-1] - r.fa_total) < 1e-6


def test_compare_food_states_returns_two():
    fasted, fed = compare_food_states("TestDrug", 100.0, 2.0, 4.0)
    assert isinstance(fasted, GITransitResult)
    assert isinstance(fed, GITransitResult)


def test_fed_tmax_later_than_fasted():
    fasted, fed = compare_food_states("TestDrug", 100.0, 2.0, 4.0)
    assert fed.tmax_h >= fasted.tmax_h


def test_f_abs_si_leq_fa_total():
    r = _default_result()
    assert r.f_abs_small_intestine <= r.fa_total + 1e-9


def test_tmax_in_range():
    r = _default_result()
    assert 0.0 < r.tmax_h < r.times_h[-1]


def test_rate_limiting_valid_values():
    r = _default_result()
    assert r.rate_limiting in {"solubility", "permeability", "transit"}


def test_dose_proportionality_cmax():
    r1 = _default_result(dose_mg=50.0, solubility_mg_mL=100.0)
    r2 = _default_result(dose_mg=100.0, solubility_mg_mL=100.0)
    ratio = r2.cmax_mg_L / r1.cmax_mg_L
    assert 1.5 < ratio < 2.5  # approximately 2x


def test_high_logP_better_absorption():
    r_low = _default_result(logP=0.5)
    r_high = _default_result(logP=3.0)
    assert r_high.fa_total > r_low.fa_total


def test_colon_absorbs_less_than_si():
    """Colon (index 6) should absorb less than total SI (indices 1-5)."""
    r = _default_result()
    si_total = sum(r.segment_absorbed_pct[1:6])
    colon = r.segment_absorbed_pct[6]
    assert colon < si_total


def test_times_length():
    r = _default_result(t_end_h=24.0, dt_h=0.05)
    expected = int(24.0 / 0.05) + 1
    assert len(r.times_h) == expected


def test_notes_contain_drug_name_and_food():
    r = _default_result(drug_name="Metformin", food_state="fasted")
    assert "Metformin" in r.notes
    assert "fasted" in r.notes


def test_auc_increases_with_dose():
    r1 = _default_result(dose_mg=50.0)
    r2 = _default_result(dose_mg=100.0)
    assert r2.auc_0_t > r1.auc_0_t


def test_segment_absorbed_pct_length():
    r = _default_result()
    assert len(r.segment_absorbed_pct) == 7


def test_cp_all_non_negative():
    r = _default_result()
    assert all(v >= 0.0 for v in r.cp_mg_L)
