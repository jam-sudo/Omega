"""Tests for Phase 506 — Drug Solubility pH Profile Prediction."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.drug_solubility_ph_profile import (
    SolubilityPhProfileResult,
    find_optimal_gi_segment,
    predict_solubility_ph_profile,
)

_VALID_SEGMENTS = {"stomach", "duodenum", "jejunum", "ileum", "colon"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def acid_result():
    return predict_solubility_ph_profile(
        drug_name="AcidDrug", intrinsic_solubility_mg_mL=0.1, pKa=4.5, drug_type="acid"
    )


@pytest.fixture
def base_result():
    return predict_solubility_ph_profile(
        drug_name="BaseDrug", intrinsic_solubility_mg_mL=0.5, pKa=8.0, drug_type="base"
    )


@pytest.fixture
def neutral_result():
    return predict_solubility_ph_profile(
        drug_name="NeutralDrug", intrinsic_solubility_mg_mL=1.0, pKa=7.0, drug_type="neutral"
    )


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_returns_correct_type(acid_result):
    assert isinstance(acid_result, SolubilityPhProfileResult)


def test_has_required_fields(acid_result):
    assert hasattr(acid_result, "drug_name")
    assert hasattr(acid_result, "intrinsic_solubility_mg_mL")
    assert hasattr(acid_result, "pKa")
    assert hasattr(acid_result, "drug_type")
    assert hasattr(acid_result, "ph_values")
    assert hasattr(acid_result, "solubility_values_mg_mL")
    assert hasattr(acid_result, "stomach_solubility_mg_mL")
    assert hasattr(acid_result, "duodenum_solubility_mg_mL")
    assert hasattr(acid_result, "jejunum_solubility_mg_mL")
    assert hasattr(acid_result, "ileum_solubility_mg_mL")
    assert hasattr(acid_result, "colon_solubility_mg_mL")
    assert hasattr(acid_result, "max_solubility_mg_mL")
    assert hasattr(acid_result, "min_solubility_mg_mL")
    assert hasattr(acid_result, "max_solubility_ph")
    assert hasattr(acid_result, "dissolution_limited_segments")
    assert hasattr(acid_result, "notes")


# ---------------------------------------------------------------------------
# pH values range
# ---------------------------------------------------------------------------


def test_ph_values_range_from_1_to_10(acid_result):
    assert acid_result.ph_values[0] == pytest.approx(1.0)
    assert acid_result.ph_values[-1] == pytest.approx(10.0)


def test_ph_values_length(acid_result):
    """pH 1.0 to 10.0 in 0.5 steps = 19 points."""
    assert len(acid_result.ph_values) == 19


def test_ph_and_solubility_same_length(acid_result):
    assert len(acid_result.ph_values) == len(acid_result.solubility_values_mg_mL)


# ---------------------------------------------------------------------------
# Acid: higher solubility at high pH
# ---------------------------------------------------------------------------


def test_acid_ileum_greater_than_stomach(acid_result):
    """Acids are more soluble at high pH (ileum pH=7.2 > stomach pH=1.5)."""
    assert acid_result.ileum_solubility_mg_mL > acid_result.stomach_solubility_mg_mL


def test_acid_solubility_increases_with_ph(acid_result):
    """For an acid, solubility should increase monotonically with pH."""
    sol = acid_result.solubility_values_mg_mL
    # Allow for floating point; check general trend
    low_ph_avg = sum(sol[:4]) / 4  # pH ~1-2.5
    high_ph_avg = sum(sol[-4:]) / 4  # pH ~8.5-10
    assert high_ph_avg > low_ph_avg


# ---------------------------------------------------------------------------
# Base: higher solubility at low pH
# ---------------------------------------------------------------------------


def test_base_stomach_greater_than_ileum(base_result):
    """Bases are more soluble at low pH (stomach pH=1.5 < ileum pH=7.2)."""
    assert base_result.stomach_solubility_mg_mL > base_result.ileum_solubility_mg_mL


def test_base_solubility_decreases_with_ph(base_result):
    """For a base, solubility should decrease monotonically with pH."""
    sol = base_result.solubility_values_mg_mL
    low_ph_avg = sum(sol[:4]) / 4
    high_ph_avg = sum(sol[-4:]) / 4
    assert low_ph_avg > high_ph_avg


# ---------------------------------------------------------------------------
# Neutral: constant solubility across all compartments
# ---------------------------------------------------------------------------


def test_neutral_stomach_equals_intrinsic(neutral_result):
    assert neutral_result.stomach_solubility_mg_mL == pytest.approx(1.0, rel=1e-6)


def test_neutral_duodenum_equals_intrinsic(neutral_result):
    assert neutral_result.duodenum_solubility_mg_mL == pytest.approx(1.0, rel=1e-6)


def test_neutral_jejunum_equals_intrinsic(neutral_result):
    assert neutral_result.jejunum_solubility_mg_mL == pytest.approx(1.0, rel=1e-6)


def test_neutral_ileum_equals_intrinsic(neutral_result):
    assert neutral_result.ileum_solubility_mg_mL == pytest.approx(1.0, rel=1e-6)


def test_neutral_colon_equals_intrinsic(neutral_result):
    assert neutral_result.colon_solubility_mg_mL == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Max/min invariants
# ---------------------------------------------------------------------------


def test_max_solubility_gte_min_solubility(acid_result):
    assert acid_result.max_solubility_mg_mL >= acid_result.min_solubility_mg_mL


def test_max_solubility_gte_min_base(base_result):
    assert base_result.max_solubility_mg_mL >= base_result.min_solubility_mg_mL


def test_max_solubility_gte_min_neutral(neutral_result):
    assert neutral_result.max_solubility_mg_mL >= neutral_result.min_solubility_mg_mL


def test_max_solubility_ph_in_range(acid_result):
    assert 1.0 <= acid_result.max_solubility_ph <= 10.0


# ---------------------------------------------------------------------------
# Field value correctness
# ---------------------------------------------------------------------------


def test_drug_name_stored(acid_result):
    assert acid_result.drug_name == "AcidDrug"


def test_intrinsic_solubility_stored(acid_result):
    assert acid_result.intrinsic_solubility_mg_mL == pytest.approx(0.1)


def test_pKa_stored(acid_result):
    assert acid_result.pKa == pytest.approx(4.5)


def test_drug_type_stored_acid(acid_result):
    assert acid_result.drug_type == "acid"


def test_drug_type_stored_base(base_result):
    assert base_result.drug_type == "base"


def test_notes_non_empty(acid_result):
    assert isinstance(acid_result.notes, str)
    assert len(acid_result.notes) > 0


def test_dissolution_limited_is_list(acid_result):
    assert isinstance(acid_result.dissolution_limited_segments, list)


# ---------------------------------------------------------------------------
# find_optimal_gi_segment
# ---------------------------------------------------------------------------


def test_find_optimal_returns_valid_segment_name():
    seg = find_optimal_gi_segment(
        drug_name="X", intrinsic_solubility_mg_mL=0.1, pKa=4.5, drug_type="acid"
    )
    assert seg in _VALID_SEGMENTS


def test_find_optimal_acid_is_high_ph_segment():
    """For an acid, optimal solubility is at high pH (ileum or colon)."""
    seg = find_optimal_gi_segment(
        drug_name="X", intrinsic_solubility_mg_mL=0.01, pKa=3.0, drug_type="acid"
    )
    assert seg in {"ileum", "colon"}


def test_find_optimal_base_is_stomach():
    """For a base, optimal solubility is at low pH (stomach)."""
    seg = find_optimal_gi_segment(
        drug_name="X", intrinsic_solubility_mg_mL=0.5, pKa=9.0, drug_type="base"
    )
    assert seg == "stomach"


def test_find_optimal_neutral_returns_valid():
    seg = find_optimal_gi_segment(
        drug_name="X", intrinsic_solubility_mg_mL=1.0, pKa=7.0, drug_type="neutral"
    )
    assert seg in _VALID_SEGMENTS


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_intrinsic_solubility():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        predict_solubility_ph_profile(
            drug_name="X", intrinsic_solubility_mg_mL=0.0, pKa=5.0, drug_type="acid"
        )


def test_raises_on_negative_intrinsic_solubility():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        predict_solubility_ph_profile(
            drug_name="X", intrinsic_solubility_mg_mL=-1.0, pKa=5.0, drug_type="acid"
        )


def test_raises_on_invalid_drug_type():
    with pytest.raises(ValueError, match="drug_type"):
        predict_solubility_ph_profile(
            drug_name="X", intrinsic_solubility_mg_mL=1.0, pKa=5.0, drug_type="zwitterion"
        )
