"""Tests for Phase 1064 — Bioequivalence Statistical Simulator."""

import math

import pytest

from omega_pbpk.clinical.bioequivalence_simulator import (
    BioequivalenceResult,
    simulate_bioequivalence,
)

# ---------------------------------------------------------------------------
# Basic return type and field sanity
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_bioequivalence("RefDrug", "TestDrug", n_subjects=24)
    assert isinstance(result, BioequivalenceResult)


def test_result_is_frozen():
    result = simulate_bioequivalence("RefDrug", "TestDrug", n_subjects=24)
    with pytest.raises((AttributeError, TypeError)):
        result.gm_ratio_auc = 0.9  # type: ignore[misc]


def test_gm_ratio_auc_positive():
    result = simulate_bioequivalence("R", "T", n_subjects=24, gm_ratio_auc=1.05)
    assert result.gm_ratio_auc > 0.0


def test_gm_ratio_cmax_positive():
    result = simulate_bioequivalence("R", "T", n_subjects=24, gm_ratio_cmax=0.95)
    assert result.gm_ratio_cmax > 0.0


def test_ci_auc_ordering():
    result = simulate_bioequivalence("R", "T", n_subjects=24)
    assert result.ci_lower_auc < result.ci_upper_auc


def test_ci_cmax_ordering():
    result = simulate_bioequivalence("R", "T", n_subjects=24)
    assert result.ci_lower_cmax < result.ci_upper_cmax


def test_intra_cv_auc_preserved():
    result = simulate_bioequivalence("R", "T", n_subjects=24, intra_cv_auc_pct=18.5)
    assert math.isclose(result.intra_subject_cv_auc, 18.5)


def test_notes_nonempty():
    result = simulate_bioequivalence("R", "T", n_subjects=24)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# BE conclusions
# ---------------------------------------------------------------------------


def test_be_pass_when_gm_ratio_1_low_cv_large_n():
    """GMR=1.0, low CV, large n → should pass BE."""
    result = simulate_bioequivalence(
        "R",
        "T",
        n_subjects=40,
        gm_ratio_auc=1.0,
        gm_ratio_cmax=1.0,
        intra_cv_auc_pct=10.0,
        intra_cv_cmax_pct=10.0,
    )
    assert result.be_conclusion_auc is True
    assert result.be_conclusion_cmax is True
    assert result.overall_be is True


def test_be_fail_when_gm_ratio_far_from_1():
    """GMR=0.70 (far outside limits) → AUC BE should fail."""
    result = simulate_bioequivalence(
        "R",
        "T",
        n_subjects=24,
        gm_ratio_auc=0.70,
        gm_ratio_cmax=0.70,
        intra_cv_auc_pct=20.0,
        intra_cv_cmax_pct=20.0,
    )
    assert result.be_conclusion_auc is False
    assert result.overall_be is False


def test_overall_be_requires_both():
    """overall_be = be_auc AND be_cmax."""
    result = simulate_bioequivalence(
        "R",
        "T",
        n_subjects=24,
        gm_ratio_auc=1.0,
        gm_ratio_cmax=0.70,
        intra_cv_auc_pct=10.0,
        intra_cv_cmax_pct=10.0,
    )
    # AUC should pass, Cmax should fail
    assert result.overall_be == (result.be_conclusion_auc and result.be_conclusion_cmax)


def test_overall_be_consistency():
    """overall_be must always equal be_auc AND be_cmax."""
    for gmr_auc, gmr_cmax, cv in [(1.0, 1.0, 15.0), (0.85, 1.1, 20.0), (0.70, 0.70, 25.0)]:
        result = simulate_bioequivalence(
            "R",
            "T",
            n_subjects=20,
            gm_ratio_auc=gmr_auc,
            gm_ratio_cmax=gmr_cmax,
            intra_cv_auc_pct=cv,
            intra_cv_cmax_pct=cv,
        )
        assert result.overall_be == (result.be_conclusion_auc and result.be_conclusion_cmax)


# ---------------------------------------------------------------------------
# Power estimation
# ---------------------------------------------------------------------------


def test_power_in_valid_range():
    result = simulate_bioequivalence("R", "T", n_subjects=24)
    assert 0.01 <= result.power_estimate <= 0.99


def test_power_reasonable_high_for_good_study():
    """Large n, GMR=1.0, low CV → power should be reasonably high."""
    result = simulate_bioequivalence(
        "R",
        "T",
        n_subjects=60,
        gm_ratio_auc=1.0,
        gm_ratio_cmax=1.0,
        intra_cv_auc_pct=10.0,
        intra_cv_cmax_pct=10.0,
    )
    assert result.power_estimate > 0.4


def test_power_low_for_small_n():
    """Very small n → power should be low."""
    result = simulate_bioequivalence(
        "R",
        "T",
        n_subjects=4,
        gm_ratio_auc=1.0,
        gm_ratio_cmax=1.0,
        intra_cv_auc_pct=30.0,
        intra_cv_cmax_pct=30.0,
    )
    assert result.power_estimate < 0.5


# ---------------------------------------------------------------------------
# Comparative behaviour
# ---------------------------------------------------------------------------


def test_larger_n_gives_narrower_ci_auc():
    result_small = simulate_bioequivalence("R", "T", n_subjects=12, intra_cv_auc_pct=20.0)
    result_large = simulate_bioequivalence("R", "T", n_subjects=60, intra_cv_auc_pct=20.0)
    width_small = result_small.ci_upper_auc - result_small.ci_lower_auc
    width_large = result_large.ci_upper_auc - result_large.ci_lower_auc
    assert width_large < width_small


def test_larger_n_gives_narrower_ci_cmax():
    result_small = simulate_bioequivalence("R", "T", n_subjects=12, intra_cv_cmax_pct=25.0)
    result_large = simulate_bioequivalence("R", "T", n_subjects=60, intra_cv_cmax_pct=25.0)
    width_small = result_small.ci_upper_cmax - result_small.ci_lower_cmax
    width_large = result_large.ci_upper_cmax - result_large.ci_lower_cmax
    assert width_large < width_small


def test_ci_contains_gm_ratio():
    """The CI should bracket the true GMR."""
    result = simulate_bioequivalence("R", "T", n_subjects=24, gm_ratio_auc=0.95)
    assert result.ci_lower_auc <= 0.95 <= result.ci_upper_auc


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_n_subjects_1_raises():
    with pytest.raises(ValueError, match="n_subjects"):
        simulate_bioequivalence("R", "T", n_subjects=1)


def test_gm_ratio_auc_zero_raises():
    with pytest.raises(ValueError, match="gm_ratio_auc"):
        simulate_bioequivalence("R", "T", n_subjects=24, gm_ratio_auc=0.0)


def test_gm_ratio_cmax_zero_raises():
    with pytest.raises(ValueError, match="gm_ratio_cmax"):
        simulate_bioequivalence("R", "T", n_subjects=24, gm_ratio_cmax=0.0)


def test_cv_auc_zero_raises():
    with pytest.raises(ValueError, match="intra_cv_auc_pct"):
        simulate_bioequivalence("R", "T", n_subjects=24, intra_cv_auc_pct=0.0)


def test_cv_cmax_zero_raises():
    with pytest.raises(ValueError, match="intra_cv_cmax_pct"):
        simulate_bioequivalence("R", "T", n_subjects=24, intra_cv_cmax_pct=0.0)


def test_be_limits_bad_raises():
    with pytest.raises(ValueError):
        simulate_bioequivalence("R", "T", n_subjects=24, be_limits=(1.25, 0.80))
