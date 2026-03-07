"""
Tests for Phase 941: pH-solubility profile prediction.
"""

import math
import pytest

from omega_pbpk.prediction.drug_solubility_ph import (
    SolubilityPhProfileResult,
    predict_solubility_ph_profile,
    screen_solubility_profiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def acid_result():
    return predict_solubility_ph_profile(
        drug_name="AcidDrug",
        logp=2.0,
        pka=4.5,
        ionization_type="acid",
        dose_mg=100.0,
    )


@pytest.fixture
def base_result():
    return predict_solubility_ph_profile(
        drug_name="BaseDrug",
        logp=2.0,
        pka=8.5,
        ionization_type="base",
        dose_mg=100.0,
    )


@pytest.fixture
def neutral_result():
    return predict_solubility_ph_profile(
        drug_name="NeutralDrug",
        logp=1.5,
        pka=7.0,
        ionization_type="neutral",
        dose_mg=100.0,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

def test_return_type(acid_result):
    assert isinstance(acid_result, SolubilityPhProfileResult)


def test_ph_values_length_default(acid_result):
    assert len(acid_result.ph_values) == 37


def test_solubility_length_default(acid_result):
    assert len(acid_result.solubility_mg_mL) == 37


def test_all_solubility_positive(acid_result):
    assert all(s > 0 for s in acid_result.solubility_mg_mL)


def test_intrinsic_solubility_positive(acid_result):
    assert acid_result.intrinsic_solubility_mg_mL > 0


def test_ph_max_solubility_in_range(acid_result):
    assert 1.0 <= acid_result.ph_max_solubility <= 10.0


def test_max_ge_min_solubility(acid_result):
    assert acid_result.max_solubility_mg_mL >= acid_result.min_solubility_mg_mL


def test_solubility_at_gastric_ph_positive(acid_result):
    assert acid_result.solubility_at_gastric_ph > 0


def test_solubility_at_intestinal_ph_positive(acid_result):
    assert acid_result.solubility_at_intestinal_ph > 0


def test_solubility_at_colonic_ph_positive(acid_result):
    assert acid_result.solubility_at_colonic_ph > 0


# ---------------------------------------------------------------------------
# Acid / Base / Neutral behavior
# ---------------------------------------------------------------------------

def test_acid_higher_solubility_at_high_ph(acid_result):
    """Acid drug: higher solubility at high pH (more ionized)."""
    assert acid_result.solubility_at_colonic_ph > acid_result.solubility_at_gastric_ph


def test_base_higher_solubility_at_low_ph(base_result):
    """Base drug: higher solubility at low pH (more ionized)."""
    assert base_result.solubility_at_gastric_ph > base_result.solubility_at_colonic_ph


def test_neutral_all_equal_intrinsic(neutral_result):
    """Neutral drug: all solubilities equal to intrinsic."""
    s0 = neutral_result.intrinsic_solubility_mg_mL
    assert abs(neutral_result.solubility_at_gastric_ph - s0) < 1e-10
    assert abs(neutral_result.solubility_at_intestinal_ph - s0) < 1e-10
    assert abs(neutral_result.solubility_at_colonic_ph - s0) < 1e-10


def test_neutral_gut_solubility_ratio_approx_1(neutral_result):
    assert abs(neutral_result.gut_solubility_ratio - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------

def test_gut_solubility_ratio_positive(acid_result):
    assert acid_result.gut_solubility_ratio > 0


def test_bcs_solubility_class_valid(acid_result):
    assert acid_result.bcs_solubility_class in {"high", "low"}


def test_notes_non_empty(acid_result):
    assert len(acid_result.notes) > 0


def test_drug_name_preserved(acid_result):
    assert acid_result.drug_name == "AcidDrug"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_validation_dose_mg_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_solubility_ph_profile("Drug", 2.0, 5.0, "acid", dose_mg=0.0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_solubility_ph_profile("Drug", 2.0, 5.0, "acid", dose_mg=-10.0)


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_solubility_ph_profile("Drug", 2.0, 5.0, "amphoteric")


def test_validation_n_points_too_small():
    with pytest.raises(ValueError, match="n_points"):
        predict_solubility_ph_profile("Drug", 2.0, 5.0, "acid", n_points=1)


def test_validation_ph_range_not_increasing():
    with pytest.raises(ValueError, match="ph_range"):
        predict_solubility_ph_profile("Drug", 2.0, 5.0, "acid", ph_range=(7.0, 3.0))


def test_validation_ph_range_equal():
    with pytest.raises(ValueError, match="ph_range"):
        predict_solubility_ph_profile("Drug", 2.0, 5.0, "acid", ph_range=(5.0, 5.0))


# ---------------------------------------------------------------------------
# screen_solubility_profiles
# ---------------------------------------------------------------------------

def test_screen_returns_correct_length():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "pka": 4.0, "ionization_type": "acid"},
        {"drug_name": "B", "logp": 3.0, "pka": 8.0, "ionization_type": "base"},
        {"drug_name": "C", "logp": 0.5, "pka": 7.0, "ionization_type": "neutral"},
    ]
    results = screen_solubility_profiles(compounds, target_ph=6.8)
    assert len(results) == 3


def test_screen_sorted_by_solubility_descending():
    compounds = [
        {"drug_name": "LowSol", "logp": 5.0, "pka": 3.0, "ionization_type": "acid"},
        {"drug_name": "HighSol", "logp": -1.0, "pka": 3.0, "ionization_type": "acid"},
        {"drug_name": "MidSol", "logp": 2.0, "pka": 3.0, "ionization_type": "acid"},
    ]
    results = screen_solubility_profiles(compounds, target_ph=6.8)
    # Sorted descending: sol[0] >= sol[1] >= sol[2]
    s0 = results[0].solubility_at_intestinal_ph
    s1 = results[1].solubility_at_intestinal_ph
    s2 = results[2].solubility_at_intestinal_ph
    assert s0 >= s1 >= s2


def test_screen_first_result_is_type():
    compounds = [
        {"drug_name": "X", "logp": 2.0, "pka": 7.0, "ionization_type": "neutral"},
    ]
    results = screen_solubility_profiles(compounds)
    assert isinstance(results[0], SolubilityPhProfileResult)
