"""Tests for Phase 1080 — Drug Crystallinity Effects on Dissolution."""

import pytest

from omega_pbpk.prediction.crystallinity_dissolution import (
    CrystallinityDissolutionResult,
    compare_crystallinity_levels,
    simulate_crystallinity_dissolution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_ARGS = dict(
    drug_name="TestDrug",
    dose_mg=50.0,
    crystallinity_pct=50.0,
    solubility_mg_mL=0.1,
    volume_lumen_mL=250.0,
    t_end_h=8.0,
    dt_h=0.05,
)

AMOR_ARGS = dict(
    drug_name="Amorphous",
    dose_mg=10.0,
    crystallinity_pct=0.0,
    solubility_mg_mL=0.1,
    volume_lumen_mL=250.0,
    t_end_h=8.0,
    dt_h=0.05,
)

CRYST_ARGS = dict(
    drug_name="Crystalline",
    dose_mg=10.0,
    crystallinity_pct=100.0,
    solubility_mg_mL=0.1,
    volume_lumen_mL=250.0,
    t_end_h=8.0,
    dt_h=0.05,
)


# ---------------------------------------------------------------------------
# Test 1: Return type
# ---------------------------------------------------------------------------
def test_return_type():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert isinstance(result, CrystallinityDissolutionResult)


# ---------------------------------------------------------------------------
# Test 2: 0% crystalline → higher f_dissolved_at_4h than 100% crystalline
# ---------------------------------------------------------------------------
def test_amorphous_dissolves_faster_than_crystalline():
    amor = simulate_crystallinity_dissolution(**AMOR_ARGS)
    cryst = simulate_crystallinity_dissolution(**CRYST_ARGS)
    assert amor.f_dissolved_at_4h > cryst.f_dissolved_at_4h


# ---------------------------------------------------------------------------
# Test 3: a_solid starts at dose_mg
# ---------------------------------------------------------------------------
def test_a_solid_starts_at_dose():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert abs(result.a_solid_mg[0] - DEFAULT_ARGS["dose_mg"]) < 1e-9


# ---------------------------------------------------------------------------
# Test 4: a_dissolved starts at 0
# ---------------------------------------------------------------------------
def test_a_dissolved_starts_at_zero():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert result.a_dissolved_mg[0] == 0.0


# ---------------------------------------------------------------------------
# Test 5: f_dissolved_at_4h >= f_dissolved_at_1h (non-decreasing dissolution)
# ---------------------------------------------------------------------------
def test_dissolution_non_decreasing():
    result = simulate_crystallinity_dissolution(**AMOR_ARGS)
    assert result.f_dissolved_at_4h >= result.f_dissolved_at_1h


# ---------------------------------------------------------------------------
# Test 6: f_dissolved_complete in [0, 1]
# ---------------------------------------------------------------------------
def test_f_dissolved_complete_range():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert 0.0 <= result.f_dissolved_complete <= 1.0


# ---------------------------------------------------------------------------
# Test 7: absorption_enhancement_factor > 0
# ---------------------------------------------------------------------------
def test_absorption_enhancement_positive():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert result.absorption_enhancement_factor > 0.0


# ---------------------------------------------------------------------------
# Test 8: Amorphous has higher absorption_enhancement_factor than crystalline
# ---------------------------------------------------------------------------
def test_amorphous_higher_enhancement():
    amor = simulate_crystallinity_dissolution(**AMOR_ARGS)
    cryst = simulate_crystallinity_dissolution(**CRYST_ARGS)
    assert amor.absorption_enhancement_factor >= cryst.absorption_enhancement_factor


# ---------------------------------------------------------------------------
# Test 9: compare_crystallinity_levels returns correct count
# ---------------------------------------------------------------------------
def test_compare_count():
    c_list = [0.0, 25.0, 50.0, 75.0, 100.0]
    results = compare_crystallinity_levels("Drug", 10.0, c_list)
    assert len(results) == len(c_list)


# ---------------------------------------------------------------------------
# Test 10: compare_crystallinity_levels sorted by f_dissolved_at_4h descending
# ---------------------------------------------------------------------------
def test_compare_sorted_descending():
    c_list = [100.0, 50.0, 0.0]
    results = compare_crystallinity_levels("Drug", 10.0, c_list)
    vals = [r.f_dissolved_at_4h for r in results]
    assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# Test 11: t_80pct_dissolved is finite for amorphous (enough solubility)
# ---------------------------------------------------------------------------
def test_t80_finite_for_amorphous_high_sol():
    result = simulate_crystallinity_dissolution(
        drug_name="Drug",
        dose_mg=5.0,
        crystallinity_pct=0.0,
        solubility_mg_mL=1.0,  # high solubility — will fully dissolve
        volume_lumen_mL=250.0,
        t_end_h=8.0,
        dt_h=0.05,
    )
    assert result.t_80pct_dissolved_h < 999.0


# ---------------------------------------------------------------------------
# Test 12: Validation error — bad crystallinity (> 100)
# ---------------------------------------------------------------------------
def test_validation_crystallinity_over_100():
    with pytest.raises(ValueError, match="crystallinity"):
        simulate_crystallinity_dissolution(**{**DEFAULT_ARGS, "crystallinity_pct": 110.0})


# ---------------------------------------------------------------------------
# Test 13: Validation error — bad crystallinity (< 0)
# ---------------------------------------------------------------------------
def test_validation_crystallinity_negative():
    with pytest.raises(ValueError, match="crystallinity"):
        simulate_crystallinity_dissolution(**{**DEFAULT_ARGS, "crystallinity_pct": -5.0})


# ---------------------------------------------------------------------------
# Test 14: Validation error — dose <= 0
# ---------------------------------------------------------------------------
def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_crystallinity_dissolution(**{**DEFAULT_ARGS, "dose_mg": 0.0})


# ---------------------------------------------------------------------------
# Test 15: Validation error — bad solubility
# ---------------------------------------------------------------------------
def test_validation_solubility_zero():
    with pytest.raises(ValueError, match="solubility"):
        simulate_crystallinity_dissolution(**{**DEFAULT_ARGS, "solubility_mg_mL": 0.0})


# ---------------------------------------------------------------------------
# Test 16: 100% amorphous with low solubility — supersaturation may occur
# ---------------------------------------------------------------------------
def test_amorphous_low_sol_supersaturation():
    # Low solubility → S_max is small → dose >> S_max → supersaturation likely
    result = simulate_crystallinity_dissolution(
        drug_name="Supersaturate",
        dose_mg=100.0,
        crystallinity_pct=0.0,
        solubility_mg_mL=0.01,
        volume_lumen_mL=100.0,
        k_precip_per_h=0.1,
        t_end_h=8.0,
        dt_h=0.05,
    )
    # dissolved should be non-trivially positive
    assert result.a_dissolved_mg[-1] > 0.0
    # note supersaturation flag in notes
    assert "yes" in result.notes


# ---------------------------------------------------------------------------
# Test 17: 100% crystalline has t_80 = 999 when dose >> solubility
# ---------------------------------------------------------------------------
def test_crystalline_large_dose_no_t80():
    result = simulate_crystallinity_dissolution(
        drug_name="Crystalline",
        dose_mg=1000.0,
        crystallinity_pct=100.0,
        solubility_mg_mL=0.01,
        volume_lumen_mL=100.0,
        t_end_h=8.0,
        dt_h=0.05,
    )
    # With dose=1000mg and S_max=1mg, very unlikely to reach 80%
    assert result.t_80pct_dissolved_h >= 999.0


# ---------------------------------------------------------------------------
# Test 18: dissolution_fraction_list has same length as times_h
# ---------------------------------------------------------------------------
def test_dissolution_fraction_list_length():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert len(result.dissolution_fraction_list) == len(result.times_h)


# ---------------------------------------------------------------------------
# Test 19: a_solid + a_dissolved conservation (mass balance check)
# ---------------------------------------------------------------------------
def test_mass_balance():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    dose = DEFAULT_ARGS["dose_mg"]
    for s, d in zip(result.a_solid_mg, result.a_dissolved_mg):
        total = s + d
        assert abs(total - dose) < 0.5, f"Mass balance violated: {s:.3f} + {d:.3f} = {total:.3f}"


# ---------------------------------------------------------------------------
# Test 20: 50% crystallinity has intermediate f_dissolved_at_4h
# ---------------------------------------------------------------------------
def test_intermediate_crystallinity_intermediate_dissolution():
    amor = simulate_crystallinity_dissolution(**AMOR_ARGS)
    cryst = simulate_crystallinity_dissolution(**CRYST_ARGS)
    mid = simulate_crystallinity_dissolution(**{**AMOR_ARGS, "crystallinity_pct": 50.0})
    assert cryst.f_dissolved_at_4h <= mid.f_dissolved_at_4h <= amor.f_dissolved_at_4h


# ---------------------------------------------------------------------------
# Test 21: notes field is a non-empty string
# ---------------------------------------------------------------------------
def test_notes_non_empty():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Test 22: times_h starts at 0 and ends near t_end_h
# ---------------------------------------------------------------------------
def test_times_range():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert result.times_h[0] == 0.0
    assert abs(result.times_h[-1] - DEFAULT_ARGS["t_end_h"]) < 0.1


# ---------------------------------------------------------------------------
# Test 23: f_dissolved_at_1h in [0, 1]
# ---------------------------------------------------------------------------
def test_f_dissolved_1h_range():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert 0.0 <= result.f_dissolved_at_1h <= 1.0


# ---------------------------------------------------------------------------
# Test 24: f_dissolved_at_4h in [0, 1]
# ---------------------------------------------------------------------------
def test_f_dissolved_4h_range():
    result = simulate_crystallinity_dissolution(**DEFAULT_ARGS)
    assert 0.0 <= result.f_dissolved_at_4h <= 1.0
