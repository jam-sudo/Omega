"""Tests for Phase 371: membrane partitioning prediction."""

import math

import pytest

from omega_pbpk.prediction.membrane_partitioning import (
    predict_membrane_partitioning,
    screen_membrane_partitioning,
)

# ---------------------------------------------------------------------------
# Basic construction / frozen dataclass
# ---------------------------------------------------------------------------


def test_result_is_frozen_dataclass():
    res = predict_membrane_partitioning("TestDrug", logP=2.0, mw=300.0, psa=60.0, n_hbd=2)
    with pytest.raises((AttributeError, TypeError)):
        res.logP = 5.0  # type: ignore[misc]


def test_result_attributes_accessible():
    res = predict_membrane_partitioning("X", logP=1.5, mw=250.0, psa=40.0, n_hbd=1)
    assert res.compound_name == "X"
    assert res.logP == 1.5
    assert res.mw == 250.0
    assert res.psa == 40.0
    assert res.n_hbd == 1


# ---------------------------------------------------------------------------
# Km/w values
# ---------------------------------------------------------------------------


def test_km_w_neutral_positive():
    res = predict_membrane_partitioning("DrugA", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert res.km_w_neutral > 0


def test_km_w_ionized_positive():
    res = predict_membrane_partitioning("DrugA", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert res.km_w_ionized > 0


def test_km_w_effective_positive():
    res = predict_membrane_partitioning("DrugA", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert res.km_w_effective > 0


def test_log_km_w_neutral_finite():
    res = predict_membrane_partitioning("DrugA", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert math.isfinite(res.log_km_w_neutral)


def test_log_km_w_effective_finite():
    res = predict_membrane_partitioning("DrugA", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert math.isfinite(res.log_km_w_effective)


def test_ionized_lower_than_neutral():
    res = predict_membrane_partitioning("DrugA", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert res.km_w_ionized < res.km_w_neutral


def test_ionized_exactly_3_log_units_below_neutral():
    res = predict_membrane_partitioning("DrugA", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert abs(res.log_km_w_neutral - res.log_km_w_ionized - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# LogP effect
# ---------------------------------------------------------------------------


def test_high_logP_higher_km_w():
    low = predict_membrane_partitioning("LowP", logP=1.0, mw=300.0, psa=50.0, n_hbd=2)
    high = predict_membrane_partitioning("HighP", logP=4.0, mw=300.0, psa=50.0, n_hbd=2)
    assert high.km_w_neutral > low.km_w_neutral


# ---------------------------------------------------------------------------
# PSA and HBD effects
# ---------------------------------------------------------------------------


def test_high_psa_lower_km_w():
    low_psa = predict_membrane_partitioning("LowPSA", logP=3.0, mw=300.0, psa=20.0, n_hbd=1)
    high_psa = predict_membrane_partitioning("HighPSA", logP=3.0, mw=300.0, psa=120.0, n_hbd=1)
    assert low_psa.km_w_neutral > high_psa.km_w_neutral


def test_high_hbd_lower_km_w():
    low_hbd = predict_membrane_partitioning("LowHBD", logP=3.0, mw=300.0, psa=50.0, n_hbd=0)
    high_hbd = predict_membrane_partitioning("HighHBD", logP=3.0, mw=300.0, psa=50.0, n_hbd=5)
    assert low_hbd.km_w_neutral > high_hbd.km_w_neutral


# ---------------------------------------------------------------------------
# Neutral compound (no pKa): effective == neutral
# ---------------------------------------------------------------------------


def test_neutral_compound_effective_equals_neutral():
    res = predict_membrane_partitioning("Neutral", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert abs(res.km_w_effective - res.km_w_neutral) < 1e-9


# ---------------------------------------------------------------------------
# Ionized compound: effective Km/w reduced
# ---------------------------------------------------------------------------


def test_acid_at_low_pka_reduces_effective_km_w():
    """Acid with pKa << 7.4 is mostly ionized → effective Km/w << neutral."""
    neutral = predict_membrane_partitioning("Neutral", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    acid = predict_membrane_partitioning(
        "Acid", logP=3.0, mw=300.0, psa=50.0, n_hbd=2, pka_acid=3.0
    )
    assert acid.km_w_effective < neutral.km_w_effective


def test_base_at_high_pka_reduces_effective_km_w():
    """Base with pKa >> 7.4 is mostly ionized → effective Km/w << neutral."""
    neutral = predict_membrane_partitioning("Neutral", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    base = predict_membrane_partitioning(
        "Base", logP=3.0, mw=300.0, psa=50.0, n_hbd=2, pka_base=10.0
    )
    assert base.km_w_effective < neutral.km_w_effective


def test_base_at_low_pka_mostly_neutral():
    """Base with pKa << 7.4 is mostly un-ionized → effective ≈ neutral."""
    neutral = predict_membrane_partitioning("Neutral", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    base = predict_membrane_partitioning(
        "Base", logP=3.0, mw=300.0, psa=50.0, n_hbd=2, pka_base=4.0
    )
    # effective should be very close to neutral (fi ≈ 0)
    assert abs(base.km_w_effective - neutral.km_w_effective) / neutral.km_w_effective < 0.01


# ---------------------------------------------------------------------------
# Permeability
# ---------------------------------------------------------------------------


def test_perm_cm_per_s_positive():
    res = predict_membrane_partitioning("DrugA", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert res.perm_cm_per_s > 0


def test_high_logP_higher_permeability():
    low = predict_membrane_partitioning("LowP", logP=0.5, mw=300.0, psa=80.0, n_hbd=3)
    high = predict_membrane_partitioning("HighP", logP=5.0, mw=300.0, psa=20.0, n_hbd=0)
    assert high.perm_cm_per_s > low.perm_cm_per_s


# ---------------------------------------------------------------------------
# Permeability classification
# ---------------------------------------------------------------------------


def test_high_perm_class():
    # Very lipophilic, low PSA/HBD, small MW → high permeability
    res = predict_membrane_partitioning("HighPerm", logP=5.0, mw=200.0, psa=10.0, n_hbd=0)
    assert res.membrane_permeability_class == "high"


def test_low_perm_class():
    # Very hydrophilic, high PSA/HBD, large MW → low permeability
    res = predict_membrane_partitioning("LowPerm", logP=-2.0, mw=600.0, psa=200.0, n_hbd=10)
    assert res.membrane_permeability_class == "low"


def test_perm_class_values():
    res = predict_membrane_partitioning("Drug", logP=2.0, mw=300.0, psa=60.0, n_hbd=2)
    assert res.membrane_permeability_class in {"high", "moderate", "low"}


# ---------------------------------------------------------------------------
# CNS membrane potential
# ---------------------------------------------------------------------------


def test_cns_membrane_potential_true_for_high_km_w():
    # High logP + low PSA → km_w_effective > 100
    res = predict_membrane_partitioning("CNSDrug", logP=5.0, mw=250.0, psa=15.0, n_hbd=0)
    # km_w_neutral = 10^(1.09*5 - 0.01*15 - 0) = 10^5.3 >> 100
    assert res.km_w_effective > 100
    assert res.cns_membrane_potential is True


def test_cns_membrane_potential_false_for_low_km_w():
    # Very low logP → km_w << 100
    res = predict_membrane_partitioning("HydrophilicDrug", logP=-2.0, mw=300.0, psa=100.0, n_hbd=5)
    assert res.km_w_effective < 100
    assert res.cns_membrane_potential is False


# ---------------------------------------------------------------------------
# Lipid accumulation
# ---------------------------------------------------------------------------


def test_lipid_accumulation_positive():
    res = predict_membrane_partitioning("Drug", logP=3.0, mw=300.0, psa=50.0, n_hbd=2)
    assert res.lipid_accumulation_factor >= 0


# ---------------------------------------------------------------------------
# screen_membrane_partitioning
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"name": "A", "logP": 1.0, "mw": 200.0, "psa": 50.0, "n_hbd": 2},
        {"name": "B", "logP": 4.0, "mw": 300.0, "psa": 30.0, "n_hbd": 1},
        {"name": "C", "logP": 2.5, "mw": 250.0, "psa": 70.0, "n_hbd": 3},
    ]
    results = screen_membrane_partitioning(compounds)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_descending():
    compounds = [
        {"name": "Low", "logP": 0.5, "mw": 350.0, "psa": 90.0, "n_hbd": 4},
        {"name": "High", "logP": 5.0, "mw": 250.0, "psa": 20.0, "n_hbd": 0},
        {"name": "Mid", "logP": 2.5, "mw": 300.0, "psa": 55.0, "n_hbd": 2},
    ]
    results = screen_membrane_partitioning(compounds)
    km_ws = [r.km_w_effective for r in results]
    assert km_ws == sorted(km_ws, reverse=True)


def test_screen_first_is_highest_membrane_affinity():
    compounds = [
        {"name": "Lipophilic", "logP": 5.0, "mw": 250.0, "psa": 10.0, "n_hbd": 0},
        {"name": "Hydrophilic", "logP": -1.0, "mw": 400.0, "psa": 150.0, "n_hbd": 8},
    ]
    results = screen_membrane_partitioning(compounds)
    assert results[0].compound_name == "Lipophilic"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw must be"):
        predict_membrane_partitioning("Bad", logP=2.0, mw=0.0, psa=50.0, n_hbd=2)


def test_negative_psa_raises():
    with pytest.raises(ValueError, match="psa must be"):
        predict_membrane_partitioning("Bad", logP=2.0, mw=300.0, psa=-1.0, n_hbd=2)


def test_negative_n_hbd_raises():
    with pytest.raises(ValueError, match="n_hbd must be"):
        predict_membrane_partitioning("Bad", logP=2.0, mw=300.0, psa=50.0, n_hbd=-1)


def test_invalid_membrane_type_raises():
    with pytest.raises(ValueError, match="membrane_type"):
        predict_membrane_partitioning(
            "Bad", logP=2.0, mw=300.0, psa=50.0, n_hbd=2, membrane_type="INVALID"
        )
