"""Tests for Phase 860 — mRNA PK/PD Model."""

import pytest

from omega_pbpk.core.mrna_pkpd import (
    compare_mrna_formulations,
    mRNAPKPDResult,
    simulate_mrna_pkpd,
)

# ---------------------------------------------------------------------------
# Basic return type and field preservation
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_mrna_pkpd("mRNA-1273", 0.1)
    assert isinstance(result, mRNAPKPDResult)


def test_drug_name_preserved():
    result = simulate_mrna_pkpd("mRNA-drug", 0.05)
    assert result.drug_name == "mRNA-drug"


def test_dose_mg_preserved():
    result = simulate_mrna_pkpd("mRNA-drug", 0.2)
    assert result.dose_mg == 0.2


def test_route_preserved():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1, route="sc")
    assert result.route == "sc"


# ---------------------------------------------------------------------------
# Time series fields
# ---------------------------------------------------------------------------


def test_times_h_is_list():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1, t_end_h=48.0, dt_h=1.0)
    assert isinstance(result.times_h, list)
    assert len(result.times_h) > 0


def test_c_mrna_same_length_as_times():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1, t_end_h=48.0, dt_h=1.0)
    assert len(result.c_mrna_mg_L) == len(result.times_h)


def test_c_protein_same_length_as_times():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1, t_end_h=48.0, dt_h=1.0)
    assert len(result.c_protein_au) == len(result.times_h)


# ---------------------------------------------------------------------------
# PK/PD metrics
# ---------------------------------------------------------------------------


def test_cmax_mrna_positive():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1)
    assert result.cmax_mrna_mg_L > 0.0


def test_cmax_protein_positive():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1)
    assert result.cmax_protein_au > 0.0


def test_auc_mrna_positive():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1)
    assert result.auc_mrna_mg_h_per_L > 0.0


def test_auc_protein_positive():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1)
    assert result.auc_protein_au_h > 0.0


def test_protein_duration_above_half_max_positive():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1, t_end_h=168.0)
    assert result.protein_duration_above_half_max_h > 0.0


# ---------------------------------------------------------------------------
# Temporal dynamics
# ---------------------------------------------------------------------------


def test_protein_peaks_after_mrna():
    """Protein expression peaks after mRNA due to translation delay."""
    result = simulate_mrna_pkpd(
        "mRNA-drug",
        0.1,
        k_uptake_per_h=0.3,
        t_half_mrna_h=8.0,
        t_half_protein_h=48.0,
        t_end_h=168.0,
        dt_h=0.5,
    )
    assert result.tmax_protein_h > result.tmax_mrna_h


# ---------------------------------------------------------------------------
# Pharmacological relationships
# ---------------------------------------------------------------------------


def test_longer_mrna_halflife_higher_auc_protein():
    """Longer mRNA half-life → more protein expressed (higher AUC)."""
    result_short = simulate_mrna_pkpd("mRNA-drug", 0.1, t_half_mrna_h=4.0, t_end_h=168.0)
    result_long = simulate_mrna_pkpd("mRNA-drug", 0.1, t_half_mrna_h=24.0, t_end_h=168.0)
    assert result_long.auc_protein_au_h > result_short.auc_protein_au_h


def test_longer_protein_halflife_longer_duration():
    """Longer protein half-life → longer duration above half-max."""
    result_short = simulate_mrna_pkpd("mRNA-drug", 0.1, t_half_protein_h=24.0, t_end_h=168.0)
    result_long = simulate_mrna_pkpd("mRNA-drug", 0.1, t_half_protein_h=120.0, t_end_h=168.0)
    assert (
        result_long.protein_duration_above_half_max_h
        > result_short.protein_duration_above_half_max_h
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_iv_route_works():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1, route="iv")
    assert result.cmax_mrna_mg_L > 0.0
    assert result.route == "iv"


def test_im_route_works():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1, route="im")
    assert result.cmax_mrna_mg_L > 0.0


def test_sc_route_works():
    result = simulate_mrna_pkpd("mRNA-drug", 0.1, route="sc")
    assert result.cmax_mrna_mg_L > 0.0


# ---------------------------------------------------------------------------
# compare_mrna_formulations
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    formulations = [
        {"name": "F1", "t_half_mrna_h": 8.0, "k_uptake_per_h": 0.3},
        {"name": "F2", "t_half_mrna_h": 16.0, "k_uptake_per_h": 0.5},
    ]
    results = compare_mrna_formulations("mRNA-drug", 0.1, formulations)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_sorted_by_auc_protein_descending():
    formulations = [
        {"name": "Slow", "t_half_mrna_h": 4.0, "k_uptake_per_h": 0.2},
        {"name": "Medium", "t_half_mrna_h": 8.0, "k_uptake_per_h": 0.3},
        {"name": "Fast", "t_half_mrna_h": 16.0, "k_uptake_per_h": 0.5},
    ]
    results = compare_mrna_formulations("mRNA-drug", 0.1, formulations)
    aucs = [r.auc_protein_au_h for r in results]
    assert aucs == sorted(aucs, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_mrna_pkpd("mRNA-drug", 0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_mrna_pkpd("mRNA-drug", -0.1)


def test_validation_t_half_mrna_zero():
    with pytest.raises(ValueError, match="t_half_mrna_h must be > 0"):
        simulate_mrna_pkpd("mRNA-drug", 0.1, t_half_mrna_h=0.0)


def test_validation_t_half_protein_zero():
    with pytest.raises(ValueError, match="t_half_protein_h must be > 0"):
        simulate_mrna_pkpd("mRNA-drug", 0.1, t_half_protein_h=0.0)


def test_validation_vd_mrna_zero():
    with pytest.raises(ValueError, match="vd_mrna_L must be > 0"):
        simulate_mrna_pkpd("mRNA-drug", 0.1, vd_mrna_L=0.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route must be"):
        simulate_mrna_pkpd("mRNA-drug", 0.1, route="oral")
