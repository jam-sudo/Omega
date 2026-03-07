"""Tests for Phase 198 — mAb Solution Viscosity Predictor."""

import pytest

from omega_pbpk.prediction.mab_viscosity_predictor_p198 import (
    MABViscosityResult,
    predict_mab_viscosity,
    screen_mab_concentrations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def typical_result():
    """Typical IgG at 100 mg/mL, pH away from pI."""
    return predict_mab_viscosity(
        drug_name="TestMAb",
        conc_mg_mL=100.0,
        mw_kDa=150.0,
        pi_isoelectric=7.0,
        ph=5.0,
        ionic_strength_mM=150.0,
    )


@pytest.fixture
def near_pi_result():
    """mAb formulated near its pI (pH ≈ pI → high viscosity)."""
    return predict_mab_viscosity(
        drug_name="TestMAb",
        conc_mg_mL=100.0,
        mw_kDa=150.0,
        pi_isoelectric=6.0,
        ph=6.3,  # |pH - pI| = 0.3 < 1 → charge_factor = 1.5
        ionic_strength_mM=150.0,
    )


@pytest.fixture
def far_from_pi_result():
    """mAb formulated far from its pI (lower viscosity)."""
    return predict_mab_viscosity(
        drug_name="TestMAb",
        conc_mg_mL=100.0,
        mw_kDa=150.0,
        pi_isoelectric=9.0,
        ph=4.0,  # |pH - pI| = 5.0 > 4 → charge_factor = 0.8
        ionic_strength_mM=150.0,
    )


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type(typical_result):
    assert isinstance(typical_result, MABViscosityResult)


def test_drug_name_preserved(typical_result):
    assert typical_result.drug_name == "TestMAb"


def test_conc_preserved(typical_result):
    assert typical_result.conc_mg_mL == 100.0


def test_mw_preserved(typical_result):
    assert typical_result.mw_kDa == 150.0


def test_pi_preserved(typical_result):
    assert typical_result.pi_isoelectric == 7.0


def test_ph_preserved(typical_result):
    assert typical_result.ph == 5.0


def test_ionic_strength_preserved(typical_result):
    assert typical_result.ionic_strength_mM == 150.0


def test_notes_is_str(typical_result):
    assert isinstance(typical_result.notes, str)


def test_result_is_frozen():
    """MABViscosityResult must be frozen dataclass."""
    result = predict_mab_viscosity(
        drug_name="X",
        conc_mg_mL=50.0,
        mw_kDa=148.0,
        pi_isoelectric=8.5,
        ph=6.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        result.viscosity_mPas = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Physical correctness
# ---------------------------------------------------------------------------


def test_viscosity_positive(typical_result):
    assert typical_result.viscosity_mPas > 0.0


def test_higher_conc_higher_viscosity():
    """Ross-Minton model: viscosity increases with concentration."""
    low = predict_mab_viscosity("X", 50.0, 150.0, 7.0, 5.0)
    high = predict_mab_viscosity("X", 150.0, 150.0, 7.0, 5.0)
    assert high.viscosity_mPas > low.viscosity_mPas


def test_near_pi_higher_viscosity_than_far_from_pi(near_pi_result, far_from_pi_result):
    """pH near pI → higher charge_factor → higher viscosity."""
    assert near_pi_result.viscosity_mPas > far_from_pi_result.viscosity_mPas


def test_k_rm_positive(typical_result):
    assert typical_result.k_rm > 0.0


def test_max_injectable_conc_positive(typical_result):
    assert typical_result.max_injectable_conc_mg_mL > 0.0


def test_max_injectable_conc_less_than_conc_max(typical_result):
    assert typical_result.max_injectable_conc_mg_mL < 500.0


# ---------------------------------------------------------------------------
# Viscosity class
# ---------------------------------------------------------------------------

VALID_CLASSES = {"injectable", "challenging", "problematic"}


def test_viscosity_class_valid(typical_result):
    assert typical_result.viscosity_class in VALID_CLASSES


def test_low_conc_injectable_class():
    result = predict_mab_viscosity("X", 1.0, 150.0, 7.0, 5.0)
    assert result.viscosity_class == "injectable"


def test_very_high_conc_problematic_class():
    result = predict_mab_viscosity(
        "HighViscMAb",
        300.0,
        200.0,
        7.0,
        7.0,  # at pI → max k_RM
        ionic_strength_mM=0.0,
    )
    # Should be in challenging or problematic range
    assert result.viscosity_class in {"challenging", "problematic"}


# ---------------------------------------------------------------------------
# screen_mab_concentrations
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_mab_concentrations("TestMAb", 150.0, 7.0, 5.0, [50.0, 100.0, 150.0])
    assert isinstance(results, list)


def test_screen_correct_length():
    results = screen_mab_concentrations("TestMAb", 150.0, 7.0, 5.0, [50.0, 100.0, 150.0, 200.0])
    assert len(results) == 4


def test_screen_sorted_by_conc_ascending():
    concs = [200.0, 50.0, 150.0, 10.0]
    results = screen_mab_concentrations("X", 148.0, 8.5, 6.0, concs)
    for i in range(len(results) - 1):
        assert results[i].conc_mg_mL <= results[i + 1].conc_mg_mL


def test_screen_all_are_mab_viscosity_results():
    results = screen_mab_concentrations("X", 148.0, 8.5, 6.0, [10.0, 50.0, 100.0])
    for r in results:
        assert isinstance(r, MABViscosityResult)


def test_screen_viscosity_increases_with_conc():
    results = screen_mab_concentrations("X", 150.0, 7.0, 5.0, [10.0, 50.0, 100.0, 200.0])
    for i in range(len(results) - 1):
        assert results[i].viscosity_mPas <= results[i + 1].viscosity_mPas


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_conc_zero_raises():
    with pytest.raises(ValueError, match="conc_mg_mL"):
        predict_mab_viscosity("X", 0.0, 150.0, 7.0, 5.0)


def test_conc_negative_raises():
    with pytest.raises(ValueError, match="conc_mg_mL"):
        predict_mab_viscosity("X", -10.0, 150.0, 7.0, 5.0)


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_kDa"):
        predict_mab_viscosity("X", 100.0, 0.0, 7.0, 5.0)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_kDa"):
        predict_mab_viscosity("X", 100.0, -50.0, 7.0, 5.0)


def test_ionic_strength_negative_raises():
    with pytest.raises(ValueError, match="ionic_strength_mM"):
        predict_mab_viscosity("X", 100.0, 150.0, 7.0, 5.0, ionic_strength_mM=-1.0)
