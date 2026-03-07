"""Tests for Phase 1094 — Drug Synovial Joint PK Extended (Arthritis Therapeutics)."""

import pytest

from omega_pbpk.core.synovial_joint_pk import (
    SynovialJointPKResult,
    compare_inflammation_states,
    simulate_synovial_joint_pk,
)


# ---------------------------------------------------------------------------
# Basic return-type and structure tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_synovial_joint_pk("methotrexate", dose_mg=10.0)
    assert isinstance(result, SynovialJointPKResult)


def test_drug_name_preserved():
    result = simulate_synovial_joint_pk("dexamethasone", dose_mg=5.0)
    assert result.drug_name == "dexamethasone"


def test_dose_preserved():
    result = simulate_synovial_joint_pk("adalimumab", dose_mg=40.0)
    assert result.dose_mg == pytest.approx(40.0)


def test_inflammation_factor_preserved():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0, inflammation_factor=2.5)
    assert result.inflammation_factor == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Initial condition tests
# ---------------------------------------------------------------------------


def test_c_synovial_starts_at_zero():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert result.c_synovial_mg_L[0] == pytest.approx(0.0)


def test_c_cartilage_starts_at_zero():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert result.c_cartilage_mg_L[0] == pytest.approx(0.0)


def test_c_plasma_starts_at_dose_over_vd():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0, vd_plasma_L=5.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(10.0 / 5.0)


# ---------------------------------------------------------------------------
# PK metric tests
# ---------------------------------------------------------------------------


def test_cmax_synovial_positive():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert result.cmax_synovial > 0.0


def test_auc_synovial_positive():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert result.auc_synovial > 0.0


def test_auc_cartilage_positive():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert result.auc_cartilage > 0.0


def test_synovial_to_plasma_ratio_positive():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert result.synovial_to_plasma_ratio > 0.0


def test_tmax_synovial_in_valid_range():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0, t_end_h=48.0)
    assert 0.0 <= result.tmax_synovial_h <= 48.0


def test_t_above_mic_non_negative():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert result.t_above_mic_synovial_h >= 0.0


def test_notes_is_string():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Inflammation factor effects
# ---------------------------------------------------------------------------


def test_higher_inflammation_factor_yields_higher_auc_synovial():
    """Inflamed joint (higher k_ps) should result in more synovial drug exposure."""
    r_low = simulate_synovial_joint_pk("drug_x", dose_mg=10.0, inflammation_factor=1.0)
    r_high = simulate_synovial_joint_pk("drug_x", dose_mg=10.0, inflammation_factor=3.0)
    assert r_high.auc_synovial > r_low.auc_synovial


def test_higher_inflammation_factor_yields_higher_cmax_synovial():
    r_low = simulate_synovial_joint_pk("drug_x", dose_mg=10.0, inflammation_factor=1.0)
    r_high = simulate_synovial_joint_pk("drug_x", dose_mg=10.0, inflammation_factor=3.0)
    assert r_high.cmax_synovial > r_low.cmax_synovial


# ---------------------------------------------------------------------------
# joint_penetration_efficacy classification
# ---------------------------------------------------------------------------


def test_joint_penetration_efficacy_valid_values():
    result = simulate_synovial_joint_pk("drug_x", dose_mg=10.0)
    assert result.joint_penetration_efficacy in {"poor", "moderate", "good"}


# ---------------------------------------------------------------------------
# compare_inflammation_states tests
# ---------------------------------------------------------------------------


def test_compare_inflammation_states_returns_correct_count():
    factors = [1.0, 2.0, 3.0]
    results = compare_inflammation_states("drug_x", dose_mg=10.0, inflammation_factors=factors)
    assert len(results) == 3


def test_compare_inflammation_states_all_are_results():
    factors = [1.0, 2.0]
    results = compare_inflammation_states("drug_x", dose_mg=10.0, inflammation_factors=factors)
    for r in results:
        assert isinstance(r, SynovialJointPKResult)


def test_compare_inflammation_states_sorted_descending():
    factors = [1.0, 2.0, 3.0, 4.0]
    results = compare_inflammation_states("drug_x", dose_mg=10.0, inflammation_factors=factors)
    aucs = [r.auc_synovial for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_inflammation_states_single_factor():
    results = compare_inflammation_states("drug_x", dose_mg=10.0, inflammation_factors=[1.5])
    assert len(results) == 1
    assert isinstance(results[0], SynovialJointPKResult)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_synovial_joint_pk("drug_x", dose_mg=0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_synovial_joint_pk("drug_x", dose_mg=-5.0)


def test_invalid_inflammation_factor_below_one():
    with pytest.raises(ValueError, match="inflammation_factor"):
        simulate_synovial_joint_pk("drug_x", dose_mg=10.0, inflammation_factor=0.5)


def test_invalid_inflammation_factor_above_five():
    with pytest.raises(ValueError, match="inflammation_factor"):
        simulate_synovial_joint_pk("drug_x", dose_mg=10.0, inflammation_factor=6.0)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl"):
        simulate_synovial_joint_pk("drug_x", dose_mg=10.0, cl_L_per_h=0.0)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd"):
        simulate_synovial_joint_pk("drug_x", dose_mg=10.0, vd_plasma_L=-1.0)
