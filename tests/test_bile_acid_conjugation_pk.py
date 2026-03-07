"""Tests for Phase 854 — Bile Acid Conjugation and Absorption PK."""

import pytest

from omega_pbpk.core.bile_acid_conjugation_pk import (
    BileAcidConjugationPKResult,
    compare_cleavage_rates,
    simulate_bile_acid_conjugation_pk,
)

# ─── Basic return type & field preservation ───────────────────────────────────


def test_returns_correct_type():
    result = simulate_bile_acid_conjugation_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, BileAcidConjugationPKResult)


def test_fields_preserved():
    result = simulate_bile_acid_conjugation_pk(
        "BileConj",
        dose_mg=50.0,
        ka_conj_per_h=0.8,
        k_cleavage_per_h=0.2,
    )
    assert result.drug_name == "BileConj"
    assert result.dose_mg == 50.0
    assert result.ka_conj_per_h == 0.8
    assert result.k_cleavage_per_h == 0.2


# ─── Trajectory arrays ───────────────────────────────────────────────────────


def test_trajectory_lengths_match():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, t_end_h=12.0, dt_h=0.5)
    n = len(result.times_h)
    assert len(result.c_plasma_drug_mg_L) == n
    assert len(result.a_plasma_conj_mg) == n
    assert len(result.a_gut_conjugate_mg) == n


def test_trajectory_starts_at_zero():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0)
    assert result.times_h[0] == 0.0
    assert result.c_plasma_drug_mg_L[0] == 0.0
    assert result.a_plasma_conj_mg[0] == 0.0
    assert result.a_gut_conjugate_mg[0] == 100.0


def test_trajectory_all_non_negative():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0)
    assert all(c >= 0.0 for c in result.c_plasma_drug_mg_L)
    assert all(a >= 0.0 for a in result.a_plasma_conj_mg)
    assert all(a >= 0.0 for a in result.a_gut_conjugate_mg)


# ─── PK metrics ──────────────────────────────────────────────────────────────


def test_cmax_positive():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, k_cleavage_per_h=0.2)
    assert result.cmax_drug_mg_L > 0.0


def test_auc_positive():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, k_cleavage_per_h=0.2)
    assert result.auc_drug_mg_h_per_L > 0.0


def test_tmax_within_simulation():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, t_end_h=24.0)
    assert 0.0 <= result.tmax_drug_h <= 24.0


def test_cmax_equals_max_of_trajectory():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0)
    assert abs(result.cmax_drug_mg_L - max(result.c_plasma_drug_mg_L)) < 1e-12


# ─── Mechanistic effects ─────────────────────────────────────────────────────


def test_higher_cleavage_higher_cmax():
    low_clv = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, k_cleavage_per_h=0.05)
    high_clv = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, k_cleavage_per_h=0.8)
    assert high_clv.cmax_drug_mg_L > low_clv.cmax_drug_mg_L


def test_higher_ka_conj_more_absorbed():
    slow = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, ka_conj_per_h=0.1)
    fast = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, ka_conj_per_h=2.0)
    assert fast.auc_drug_mg_h_per_L > slow.auc_drug_mg_h_per_L


def test_zero_cleavage_no_drug_released():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, k_cleavage_per_h=0.0)
    assert result.cmax_drug_mg_L == 0.0
    assert result.auc_drug_mg_h_per_L == 0.0


def test_ehc_recycling_increases_auc():
    no_recycle = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, recycling_cycles=0)
    with_recycle = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, recycling_cycles=3)
    assert with_recycle.auc_drug_mg_h_per_L > no_recycle.auc_drug_mg_h_per_L


def test_dose_proportionality():
    r1 = simulate_bile_acid_conjugation_pk("D", dose_mg=50.0)
    r2 = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0)
    ratio = r2.cmax_drug_mg_L / r1.cmax_drug_mg_L
    assert abs(ratio - 2.0) < 0.1


# ─── Summary fractions ───────────────────────────────────────────────────────


def test_f_absorbed_conj_in_unit_interval():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0)
    assert 0.0 <= result.f_absorbed_conj <= 1.0


def test_f_converted_to_drug_in_unit_interval():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0)
    assert 0.0 <= result.f_converted_to_drug <= 1.0


def test_f_converted_zero_when_no_cleavage():
    result = simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, k_cleavage_per_h=0.0)
    assert result.f_converted_to_drug == 0.0


# ─── compare_cleavage_rates ───────────────────────────────────────────────────


def test_compare_returns_list():
    results = compare_cleavage_rates("D", 100.0, [0.1, 0.5, 1.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_returns_correct_types():
    results = compare_cleavage_rates("D", 100.0, [0.1, 0.5])
    assert all(isinstance(r, BileAcidConjugationPKResult) for r in results)


def test_compare_sorted_descending_cmax():
    results = compare_cleavage_rates("D", 100.0, [0.05, 0.2, 0.5, 1.0])
    for i in range(len(results) - 1):
        assert results[i].cmax_drug_mg_L >= results[i + 1].cmax_drug_mg_L


# ─── Validation errors ────────────────────────────────────────────────────────


def test_invalid_dose_mg():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_bile_acid_conjugation_pk("D", dose_mg=0.0)


def test_invalid_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_bile_acid_conjugation_pk("D", dose_mg=-10.0)


def test_invalid_ka_conj():
    with pytest.raises(ValueError, match="ka_conj_per_h must be > 0"):
        simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, ka_conj_per_h=0.0)


def test_invalid_k_cleavage_negative():
    with pytest.raises(ValueError, match="k_cleavage_per_h must be >= 0"):
        simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, k_cleavage_per_h=-0.1)


def test_invalid_recycling_cycles_negative():
    with pytest.raises(ValueError, match="recycling_cycles must be >= 0"):
        simulate_bile_acid_conjugation_pk("D", dose_mg=100.0, recycling_cycles=-1)
