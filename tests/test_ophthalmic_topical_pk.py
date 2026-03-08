"""
Tests for Phase 439: Ocular topical drug delivery PK model.
"""

import pytest

from omega_pbpk.core.ophthalmic_topical_pk import (
    OphthalmicTopicalResult,
    compare_logp_ocular,
    simulate_ophthalmic_topical_pk,
)

# --------------------------------------------------------------------------- #
# Basic return type and structure
# --------------------------------------------------------------------------- #


def test_returns_correct_type():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert isinstance(result, OphthalmicTopicalResult)


def test_times_start_at_zero():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert result.times_min[0] == pytest.approx(0.0)


def test_arrays_consistent_length():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    n = len(result.times_min)
    assert len(result.c_tear_mg_mL) == n
    assert len(result.c_cornea_mg_g) == n
    assert len(result.c_aqueous_mg_mL) == n


def test_tear_starts_at_dose_over_volume():
    dose_mg = 0.05
    drop_volume_uL = 30.0
    result = simulate_ophthalmic_topical_pk(
        "timolol", dose_mg, logp=0.0, drop_volume_uL=drop_volume_uL
    )
    V_tear_mL = (7.0 + drop_volume_uL) / 1000.0
    expected_c0 = dose_mg / V_tear_mL
    assert result.c_tear_mg_mL[0] == pytest.approx(expected_c0, rel=1e-6)


def test_cornea_starts_at_zero():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert result.c_cornea_mg_g[0] == pytest.approx(0.0)


def test_aqueous_starts_at_zero():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert result.c_aqueous_mg_mL[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# PK metric correctness
# --------------------------------------------------------------------------- #


def test_cmax_aqueous_positive():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert result.cmax_aqueous_mg_mL > 0.0


def test_auc_aqueous_positive():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert result.auc_aqueous_mg_h_per_mL > 0.0


def test_t_half_tear_short():
    """Tear half-life should be << 5 min due to fast drainage (k_drain = 0.7/min)."""
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert result.t_half_tear_min < 5.0


def test_drainage_loss_dominates():
    """k_drain = 0.7 >> k_cornea + k_noncorneal for typical logP → drainage > 50%."""
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert result.drainage_loss_pct > 50.0


def test_ocular_bioavailability_in_range():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert 0.0 <= result.ocular_bioavailability_pct <= 100.0


def test_tmax_aqueous_after_t0():
    result = simulate_ophthalmic_topical_pk("timolol", 0.05, logp=0.0)
    assert result.tmax_aqueous_min > 0.0


# --------------------------------------------------------------------------- #
# Physiological expectations
# --------------------------------------------------------------------------- #


def test_higher_logp_increases_aqueous_penetration():
    """Higher (moderate) logP → greater corneal permeability → more drug in aqueous."""
    r_low = simulate_ophthalmic_topical_pk("drug", 0.05, logp=-2.0)
    r_high = simulate_ophthalmic_topical_pk("drug", 0.05, logp=2.0)
    assert r_high.auc_aqueous_mg_h_per_mL > r_low.auc_aqueous_mg_h_per_mL


def test_larger_drop_volume_lower_initial_tear_conc():
    """Larger drop volume → more dilution → lower initial tear concentration."""
    r_small = simulate_ophthalmic_topical_pk("drug", 0.05, logp=0.0, drop_volume_uL=10.0)
    r_large = simulate_ophthalmic_topical_pk("drug", 0.05, logp=0.0, drop_volume_uL=50.0)
    assert r_small.c_tear_mg_mL[0] > r_large.c_tear_mg_mL[0]


def test_dose_linearity_cmax_aqueous():
    """2x dose → ~2x cmax_aqueous (linear system)."""
    r1 = simulate_ophthalmic_topical_pk("drug", 0.05, logp=0.0)
    r2 = simulate_ophthalmic_topical_pk("drug", 0.10, logp=0.0)
    ratio = r2.cmax_aqueous_mg_mL / r1.cmax_aqueous_mg_mL
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_dose_linearity_auc():
    """2x dose → ~2x auc_aqueous."""
    r1 = simulate_ophthalmic_topical_pk("drug", 0.05, logp=0.0)
    r2 = simulate_ophthalmic_topical_pk("drug", 0.10, logp=0.0)
    ratio = r2.auc_aqueous_mg_h_per_mL / r1.auc_aqueous_mg_h_per_mL
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_drug_name_preserved():
    result = simulate_ophthalmic_topical_pk("LatanoProst", 0.05, logp=2.5)
    assert result.drug_name == "LatanoProst"


# --------------------------------------------------------------------------- #
# compare_logp_ocular
# --------------------------------------------------------------------------- #


def test_compare_logp_returns_correct_length():
    logp_values = [-2.0, 0.0, 1.0, 2.0, 3.0]
    results = compare_logp_ocular("drug", 0.05, logp_values)
    assert len(results) == len(logp_values)


def test_compare_logp_sorted_descending():
    logp_values = [-2.0, 0.0, 1.0, 2.0, 3.0]
    results = compare_logp_ocular("drug", 0.05, logp_values)
    aucs = [r.auc_aqueous_mg_h_per_mL for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_logp_all_results_type():
    logp_values = [-1.0, 0.0, 1.0]
    results = compare_logp_ocular("drug", 0.05, logp_values)
    for r in results:
        assert isinstance(r, OphthalmicTopicalResult)


# --------------------------------------------------------------------------- #
# Validation errors
# --------------------------------------------------------------------------- #


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_ophthalmic_topical_pk("drug", 0.0, logp=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_ophthalmic_topical_pk("drug", -0.1, logp=0.0)


def test_validation_logp_too_low():
    with pytest.raises(ValueError, match="logp must be between"):
        simulate_ophthalmic_topical_pk("drug", 0.05, logp=-6.0)


def test_validation_logp_too_high():
    with pytest.raises(ValueError, match="logp must be between"):
        simulate_ophthalmic_topical_pk("drug", 0.05, logp=11.0)
