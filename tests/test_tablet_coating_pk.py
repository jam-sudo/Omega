"""Tests for Phase 217 — tablet_coating_pk.py"""

import pytest

from omega_pbpk.prediction.tablet_coating_pk import (
    VALID_COATINGS,
    CoatingReleaseResult,
    compare_coatings,
    simulate_coating_release,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def immediate_result():
    return simulate_coating_release("DrugA", 500.0, "immediate", pH_dissolution=6.8)


@pytest.fixture
def enteric_high_ph():
    return simulate_coating_release("DrugA", 500.0, "enteric", pH_dissolution=6.8)


@pytest.fixture
def enteric_low_ph():
    return simulate_coating_release("DrugA", 500.0, "enteric", pH_dissolution=3.0)


@pytest.fixture
def sustained_result():
    return simulate_coating_release("DrugA", 500.0, "sustained_release", pH_dissolution=6.8)


@pytest.fixture
def film_result():
    return simulate_coating_release("DrugA", 500.0, "film", pH_dissolution=6.8)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type(immediate_result):
    assert isinstance(immediate_result, CoatingReleaseResult)


def test_field_preservation(immediate_result):
    assert immediate_result.drug_name == "DrugA"
    assert immediate_result.dose_mg == 500.0
    assert immediate_result.coating_type == "immediate"
    assert immediate_result.ph_dissolution == 6.8


def test_times_are_list(immediate_result):
    assert isinstance(immediate_result.times_h, list)
    assert len(immediate_result.times_h) > 0


def test_pct_released_is_list(immediate_result):
    assert isinstance(immediate_result.pct_released, list)
    assert len(immediate_result.pct_released) == len(immediate_result.times_h)


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------


def test_pct_released_starts_at_zero(immediate_result):
    assert immediate_result.pct_released[0] == pytest.approx(0.0, abs=1e-9)


def test_pct_released_monotonically_increases(immediate_result):
    prev = -1.0
    for v in immediate_result.pct_released:
        assert v >= prev - 1e-9
        prev = v


def test_pct_released_at_most_100(immediate_result):
    for v in immediate_result.pct_released:
        assert v <= 100.0 + 1e-9


def test_sustained_release_starts_at_zero(sustained_result):
    assert sustained_result.pct_released[0] == pytest.approx(0.0, abs=1e-9)


def test_sustained_release_monotonically_increases(sustained_result):
    prev = -1.0
    for v in sustained_result.pct_released:
        assert v >= prev - 1e-9
        prev = v


def test_film_starts_at_zero(film_result):
    assert film_result.pct_released[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Coating-specific behaviour
# ---------------------------------------------------------------------------


def test_immediate_faster_than_enteric_at_high_ph(immediate_result, enteric_high_ph):
    """Immediate release t50 should be lower than enteric t50 at pH 6.8."""
    assert immediate_result.t50_h is not None
    assert enteric_high_ph.t50_h is not None
    assert immediate_result.t50_h < enteric_high_ph.t50_h


def test_enteric_low_ph_no_release(enteric_low_ph):
    """At pH < 5.5 enteric tablet should show no or very minimal release in 24 h."""
    # t50 should be None (never crosses 50%) or very large
    if enteric_low_ph.t50_h is not None:
        assert enteric_low_ph.t50_h >= 24.0
    # All pct_released should be effectively 0
    for v in enteric_low_ph.pct_released:
        assert v < 1e-6


def test_enteric_high_ph_releases(enteric_high_ph):
    """At pH 6.8 enteric tablet should eventually release drug."""
    final_pct = enteric_high_ph.pct_released[-1]
    assert final_pct > 50.0


def test_sustained_release_t50_around_6h(sustained_result):
    """Zero-order 12h formulation: 50% at 6h."""
    assert sustained_result.t50_h is not None
    assert 5.5 <= sustained_result.t50_h <= 6.5


def test_film_faster_than_immediate(film_result, immediate_result):
    """Film coating (k=0.8) should be faster than immediate (k=0.5)."""
    assert film_result.t50_h is not None
    assert immediate_result.t50_h is not None
    assert film_result.t50_h < immediate_result.t50_h


def test_dissolution_efficiency_positive(immediate_result):
    assert immediate_result.dissolution_efficiency > 0.0


def test_dissolution_efficiency_between_0_and_1(
    immediate_result, enteric_high_ph, sustained_result
):
    for r in [immediate_result, enteric_high_ph, sustained_result]:
        assert 0.0 <= r.dissolution_efficiency <= 1.0


def test_coating_quality_field_valid(immediate_result, enteric_high_ph, sustained_result):
    valid = {"fast", "intermediate", "slow"}
    for r in [immediate_result, enteric_high_ph, sustained_result]:
        assert r.coating_quality in valid


def test_notes_nonempty(immediate_result):
    assert len(immediate_result.notes) > 0


# ---------------------------------------------------------------------------
# compare_coatings
# ---------------------------------------------------------------------------


def test_compare_coatings_returns_four_results():
    results = compare_coatings("DrugB", 100.0, pH_dissolution=6.8)
    assert len(results) == 4


def test_compare_coatings_sorted_by_t50():
    results = compare_coatings("DrugB", 100.0, pH_dissolution=6.8)
    t50s = [r.t50_h if r.t50_h is not None else float("inf") for r in results]
    assert t50s == sorted(t50s)


def test_compare_coatings_all_coating_types():
    results = compare_coatings("DrugB", 100.0, pH_dissolution=6.8)
    types = {r.coating_type for r in results}
    assert types == VALID_COATINGS


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_coating_release("Drug", 0.0, "immediate")


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_coating_release("Drug", -100.0, "immediate")


def test_invalid_coating_type():
    with pytest.raises(ValueError, match="coating_type"):
        simulate_coating_release("Drug", 100.0, "magic_coating")


def test_ph_too_low():
    with pytest.raises(ValueError, match="pH_dissolution"):
        simulate_coating_release("Drug", 100.0, "immediate", pH_dissolution=0.5)


def test_ph_too_high():
    with pytest.raises(ValueError, match="pH_dissolution"):
        simulate_coating_release("Drug", 100.0, "immediate", pH_dissolution=10.0)
