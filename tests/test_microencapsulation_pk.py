"""Tests for microencapsulation_pk — Phase 1026."""

import pytest

from omega_pbpk.prediction.microencapsulation_pk import (
    MicroencapsulationResult,
    compare_polymers,
    predict_microencapsulation_pk,
)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_microencapsulation_pk("ibuprofen", mw_Da=206.0, logp=3.5)
    assert isinstance(result, MicroencapsulationResult)


def test_frozen_dataclass():
    result = predict_microencapsulation_pk("ibuprofen", mw_Da=206.0, logp=3.5)
    with pytest.raises((AttributeError, TypeError)):
        result.t50_h = 999.0  # type: ignore[misc]


def test_t50_positive():
    result = predict_microencapsulation_pk("drug_a", mw_Da=300.0, logp=2.0)
    assert result.t50_h > 0.0


def test_t90_greater_than_t50():
    result = predict_microencapsulation_pk("drug_a", mw_Da=300.0, logp=2.0)
    assert result.t90_h > result.t50_h


def test_burst_release_in_range():
    result = predict_microencapsulation_pk("drug_b", mw_Da=250.0, logp=1.5)
    assert 0.0 <= result.burst_release_pct <= 30.0


def test_release_mechanism_valid():
    result = predict_microencapsulation_pk("drug_c", mw_Da=300.0, logp=2.0)
    assert result.release_mechanism in {"diffusion", "erosion", "combined"}


def test_rate_constant_positive():
    result = predict_microencapsulation_pk("drug_d", mw_Da=300.0, logp=2.0)
    assert result.release_rate_constant_per_h > 0.0


def test_bioavailability_in_range():
    result = predict_microencapsulation_pk("drug_e", mw_Da=300.0, logp=2.0)
    assert 0.3 <= result.predicted_bioavailability <= 0.95


# ---------------------------------------------------------------------------
# Polymer-specific mechanism checks
# ---------------------------------------------------------------------------


def test_gelatin_mechanism_erosion():
    result = predict_microencapsulation_pk("drug_f", mw_Da=300.0, logp=1.0, polymer="gelatin")
    assert result.release_mechanism == "erosion"


def test_ethylcellulose_mechanism_diffusion():
    result = predict_microencapsulation_pk(
        "drug_g", mw_Da=300.0, logp=1.0, polymer="ethylcellulose"
    )
    assert result.release_mechanism == "diffusion"


def test_plga_mechanism_combined():
    result = predict_microencapsulation_pk("drug_h", mw_Da=300.0, logp=1.0, polymer="PLGA")
    assert result.release_mechanism == "combined"


# ---------------------------------------------------------------------------
# PLGA vs Eudragit relative t50
# ---------------------------------------------------------------------------


def test_plga_slower_than_eudragit():
    r_plga = predict_microencapsulation_pk("drug_i", mw_Da=300.0, logp=1.0, polymer="PLGA")
    r_eud = predict_microencapsulation_pk("drug_i", mw_Da=300.0, logp=1.0, polymer="Eudragit")
    assert r_plga.t50_h > r_eud.t50_h


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def test_larger_particle_size_increases_t50():
    r_small = predict_microencapsulation_pk(
        "drug_j", mw_Da=300.0, logp=1.0, particle_size_um=20.0
    )
    r_large = predict_microencapsulation_pk(
        "drug_j", mw_Da=300.0, logp=1.0, particle_size_um=200.0
    )
    assert r_large.t50_h > r_small.t50_h


def test_higher_logp_increases_t50():
    r_low = predict_microencapsulation_pk("drug_k", mw_Da=300.0, logp=0.0)
    r_high = predict_microencapsulation_pk("drug_k", mw_Da=300.0, logp=5.0)
    assert r_high.t50_h > r_low.t50_h


# ---------------------------------------------------------------------------
# compare_polymers
# ---------------------------------------------------------------------------


def test_compare_polymers_returns_five():
    results = compare_polymers("aspirin", mw_Da=180.0, logp=1.2)
    assert len(results) == 5


def test_compare_polymers_sorted_by_t50():
    results = compare_polymers("aspirin", mw_Da=180.0, logp=1.2)
    t50s = [r.t50_h for r in results]
    assert t50s == sorted(t50s)


def test_compare_polymers_all_microencapsulation_results():
    results = compare_polymers("paracetamol", mw_Da=151.0, logp=0.5)
    assert all(isinstance(r, MicroencapsulationResult) for r in results)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = predict_microencapsulation_pk("drug_l", mw_Da=300.0, logp=2.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_microencapsulation_pk("drug", mw_Da=0.0, logp=1.0)


def test_validation_particle_size_zero():
    with pytest.raises(ValueError, match="particle_size_um"):
        predict_microencapsulation_pk("drug", mw_Da=200.0, logp=1.0, particle_size_um=0.0)


def test_validation_loading_zero():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_microencapsulation_pk("drug", mw_Da=200.0, logp=1.0, drug_loading_pct=0.0)


def test_validation_loading_100():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_microencapsulation_pk("drug", mw_Da=200.0, logp=1.0, drug_loading_pct=100.0)


def test_validation_invalid_polymer():
    with pytest.raises(ValueError, match="polymer"):
        predict_microencapsulation_pk("drug", mw_Da=200.0, logp=1.0, polymer="unknown_polymer")
