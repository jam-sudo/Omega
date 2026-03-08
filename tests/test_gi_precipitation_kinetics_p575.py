"""Tests for Phase 575 - Drug Precipitation Kinetics in GI Tract."""

from omega_pbpk.core.gi_precipitation_kinetics_p575 import (
    GIPrecipitationResult,
    compare_polymer_effect,
    simulate_gi_precipitation,
)


def test_return_type():
    result = simulate_gi_precipitation("DrugA", 100.0, 0.1)
    assert isinstance(result, GIPrecipitationResult)


def test_list_lengths():
    result = simulate_gi_precipitation("DrugA", 100.0, 0.1, t_end_min=60.0, dt_min=0.5)
    expected_len = int(60.0 / 0.5) + 1
    assert len(result.times_min) == expected_len
    assert len(result.c_dissolved_mg_mL) == expected_len
    assert len(result.m_precipitated_mg) == expected_len
    assert len(result.dissolved_pct) == expected_len


def test_below_solubility_no_precipitation():
    # dose/volume = 10/250 = 0.04 mg/mL, solubility = 1.0 mg/mL
    result = simulate_gi_precipitation("DrugB", 10.0, 1.0, gi_volume_mL=250.0)
    assert all(m < 0.01 for m in result.m_precipitated_mg)
    assert result.precipitation_onset_min == float("inf")
    assert result.t_critical_min == float("inf")


def test_below_solubility_notes():
    result = simulate_gi_precipitation("DrugB", 10.0, 1.0)
    assert "no precipitation" in result.notes.lower()


def test_above_solubility_precipitation_occurs():
    # dose/volume = 500/250 = 2.0 mg/mL, solubility = 0.1 mg/mL
    result = simulate_gi_precipitation("DrugC", 500.0, 0.1)
    assert result.m_precipitated_mg[-1] > 0.0
    assert result.precipitation_onset_min < float("inf")


def test_polymer_reduces_precipitation():
    no_poly = simulate_gi_precipitation("DrugD", 500.0, 0.1, polymer_present=False)
    with_poly = simulate_gi_precipitation("DrugD", 500.0, 0.1, polymer_present=True)
    assert with_poly.m_precipitated_mg[-1] < no_poly.m_precipitated_mg[-1]


def test_polymer_inhibition_factor():
    no_poly = simulate_gi_precipitation("DrugD", 500.0, 0.1, polymer_present=False)
    with_poly = simulate_gi_precipitation("DrugD", 500.0, 0.1, polymer_present=True)
    assert no_poly.inhibition_factor == 1.0
    assert with_poly.inhibition_factor == 0.5


def test_fa_effective_range():
    result = simulate_gi_precipitation("DrugE", 500.0, 0.1)
    assert 0.0 <= result.fa_effective <= 1.0


def test_fa_effective_below_solubility_high():
    result = simulate_gi_precipitation("DrugF", 10.0, 1.0)
    assert result.fa_effective > 0.0


def test_precipitation_onset_non_negative():
    result = simulate_gi_precipitation("DrugG", 500.0, 0.1)
    assert result.precipitation_onset_min >= 0.0


def test_higher_k_precip_more_precipitation():
    low_k = simulate_gi_precipitation("DrugH", 500.0, 0.1, k_precip_per_min=0.01)
    high_k = simulate_gi_precipitation("DrugH", 500.0, 0.1, k_precip_per_min=0.2)
    assert high_k.m_precipitated_mg[-1] > low_k.m_precipitated_mg[-1]


def test_dissolved_pct_starts_at_100():
    result = simulate_gi_precipitation("DrugI", 500.0, 0.1)
    assert abs(result.dissolved_pct[0] - 100.0) < 1e-6


def test_dissolved_pct_decreases():
    result = simulate_gi_precipitation("DrugJ", 500.0, 0.1)
    assert result.dissolved_pct[-1] < result.dissolved_pct[0]


def test_drug_name_preserved():
    result = simulate_gi_precipitation("MyDrug", 100.0, 0.1)
    assert result.drug_name == "MyDrug"


def test_dose_preserved():
    result = simulate_gi_precipitation("DrugK", 200.0, 0.1)
    assert result.dose_mg == 200.0


def test_gi_volume_preserved():
    result = simulate_gi_precipitation("DrugL", 100.0, 0.1, gi_volume_mL=300.0)
    assert result.gi_volume_mL == 300.0


def test_times_start_at_zero():
    result = simulate_gi_precipitation("DrugM", 100.0, 0.1)
    assert result.times_min[0] == 0.0


def test_times_end_at_t_end():
    result = simulate_gi_precipitation("DrugN", 100.0, 0.1, t_end_min=60.0, dt_min=0.5)
    assert abs(result.times_min[-1] - 60.0) < 1e-6


def test_compare_polymer_effect_returns_list():
    results = compare_polymer_effect("DrugO", 500.0, 0.1)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_polymer_effect_sorted_by_fa():
    results = compare_polymer_effect("DrugP", 500.0, 0.1)
    assert results[0].fa_effective >= results[1].fa_effective


def test_compare_polymer_effect_polymer_better():
    results = compare_polymer_effect("DrugQ", 500.0, 0.1)
    # Polymer should have higher fa_effective (less precipitation)
    polymer_result = [r for r in results if r.inhibition_factor == 0.5][0]
    no_polymer_result = [r for r in results if r.inhibition_factor == 1.0][0]
    assert polymer_result.fa_effective > no_polymer_result.fa_effective


def test_t_critical_after_onset():
    result = simulate_gi_precipitation("DrugR", 500.0, 0.1, k_precip_per_min=0.1, t_end_min=240.0)
    if result.t_critical_min < float("inf"):
        assert result.t_critical_min >= result.precipitation_onset_min
