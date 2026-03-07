"""Tests for Phase 660 — Metabolic Liability Scorer."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.metabolic_liability_scorer import (
    MetabolicLiabilityResult,
    score_metabolic_liability,
    screen_metabolic_liability,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

LOW_LIABILITY_KWARGS = dict(
    compound_name="LowLiab",
    logP=0.5,
    mw=200.0,
    psa=80.0,
    n_hbd=2,
    n_hba=4,
    n_aromatic_rings=0,
    n_unshielded_positions=0,
    has_michael_acceptor=False,
    has_epoxide_precursor=False,
    has_aniline=False,
    has_nitro_group=False,
    has_hydrazine=False,
)

HIGH_LIABILITY_KWARGS = dict(
    compound_name="HighLiab",
    logP=5.0,
    mw=450.0,
    psa=20.0,
    n_hbd=1,
    n_hba=2,
    n_aromatic_rings=4,
    n_unshielded_positions=5,
    has_michael_acceptor=True,
    has_epoxide_precursor=True,
    has_aniline=True,
    has_nitro_group=True,
    has_hydrazine=True,
)


# ---------------------------------------------------------------------------
# Basic type and range tests
# ---------------------------------------------------------------------------


def test_returns_metabolic_liability_result():
    result = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert isinstance(result, MetabolicLiabilityResult)


def test_liability_score_in_range():
    for kwargs in [LOW_LIABILITY_KWARGS, HIGH_LIABILITY_KWARGS]:
        r = score_metabolic_liability(**kwargs)
        assert 0.0 <= r.liability_score <= 100.0


def test_liability_class_valid_values():
    valid = {"low", "moderate", "high", "very_high"}
    for kwargs in [LOW_LIABILITY_KWARGS, HIGH_LIABILITY_KWARGS]:
        r = score_metabolic_liability(**kwargs)
        assert r.liability_class in valid


def test_clint_predicted_positive():
    r = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert r.clint_predicted_uL_per_min > 0.0


def test_hepatic_er_in_range():
    for kwargs in [LOW_LIABILITY_KWARGS, HIGH_LIABILITY_KWARGS]:
        r = score_metabolic_liability(**kwargs)
        assert 0.0 <= r.hepatic_er_predicted <= 1.0


def test_t_half_positive():
    r = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert r.t_half_in_vitro_min > 0.0


def test_n_metabolic_soft_spots_non_negative():
    r = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert r.n_metabolic_soft_spots >= 0


def test_n_reactive_alerts_non_negative():
    r = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert r.n_reactive_alerts >= 0


def test_primary_liability_valid():
    valid = {"rapid_clearance", "reactive_metabolite", "both", "none"}
    for kwargs in [LOW_LIABILITY_KWARGS, HIGH_LIABILITY_KWARGS]:
        r = score_metabolic_liability(**kwargs)
        assert r.primary_liability in valid


def test_notes_nonempty():
    r = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Logic / science tests
# ---------------------------------------------------------------------------


def test_high_logP_many_soft_spots_gives_high_score():
    """High logP + many reactive features → high liability score."""
    r = score_metabolic_liability(**HIGH_LIABILITY_KWARGS)
    low_r = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert r.liability_score > low_r.liability_score


def test_low_logP_no_alerts_gives_low_score():
    """Low logP, no alerts → low liability score."""
    r = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert r.liability_class in {"low", "moderate"}


def test_aniline_increases_reactive_alerts():
    r = score_metabolic_liability(
        compound_name="AnilineCompound",
        logP=2.0,
        mw=250.0,
        psa=40.0,
        has_aniline=True,
    )
    assert r.n_reactive_alerts >= 1


def test_nitro_group_increases_reactive_alerts():
    r = score_metabolic_liability(
        compound_name="NitroCompound",
        logP=2.0,
        mw=250.0,
        psa=40.0,
        has_nitro_group=True,
    )
    assert r.n_reactive_alerts >= 1


def test_reactive_metabolite_risk_true_when_alerts_present():
    r = score_metabolic_liability(
        compound_name="Reactive",
        logP=2.0,
        mw=300.0,
        psa=50.0,
        has_aniline=True,
    )
    assert r.reactive_metabolite_risk is True


def test_rapid_clearance_risk_true_when_er_high():
    """Very high CLint (>1458 µL/min/mg) → ER > 0.7 → rapid_clearance_risk.

    Threshold: clint_body = clint * 0.06 * 40 = clint * 2.4
    ER > 0.7 requires clint_body > 3500 mL/min → clint > 1458 µL/min/mg
    """
    r = score_metabolic_liability(
        compound_name="HighCLint",
        logP=6.0,
        mw=400.0,
        psa=10.0,
        n_unshielded_positions=8,
        n_aromatic_rings=5,
        measured_clint=2000.0,
    )
    assert r.rapid_clearance_risk is True


def test_measured_clint_overrides_predicted():
    """When measured_clint is provided, it must be used as CLint."""
    measured = 99.0
    r = score_metabolic_liability(
        compound_name="MeasuredCL",
        logP=2.0,
        mw=300.0,
        psa=50.0,
        measured_clint=measured,
    )
    assert r.clint_predicted_uL_per_min == measured


def test_higher_measured_clint_shorter_t_half():
    """Higher CLint → shorter in vitro t½."""
    r_low = score_metabolic_liability(
        compound_name="LowCLint",
        logP=1.0,
        mw=300.0,
        psa=60.0,
        measured_clint=5.0,
    )
    r_high = score_metabolic_liability(
        compound_name="HighCLint",
        logP=1.0,
        mw=300.0,
        psa=60.0,
        measured_clint=200.0,
    )
    assert r_high.t_half_in_vitro_min < r_low.t_half_in_vitro_min


def test_optimization_suggestions_nonempty():
    r = score_metabolic_liability(**LOW_LIABILITY_KWARGS)
    assert isinstance(r.optimization_suggestions, str)
    assert len(r.optimization_suggestions) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_mw_zero_raises():
    with pytest.raises(ValueError):
        score_metabolic_liability(compound_name="X", logP=2.0, mw=0.0, psa=40.0)


def test_measured_clint_negative_raises():
    with pytest.raises(ValueError):
        score_metabolic_liability(
            compound_name="X", logP=2.0, mw=300.0, psa=40.0, measured_clint=-1.0
        )


# ---------------------------------------------------------------------------
# screen_metabolic_liability tests
# ---------------------------------------------------------------------------


def test_screen_returns_sorted_by_score():
    """screen_metabolic_liability must sort descending by liability_score."""
    compounds = [
        {**LOW_LIABILITY_KWARGS, "compound_name": "Low"},
        {**HIGH_LIABILITY_KWARGS, "compound_name": "High"},
    ]
    results = screen_metabolic_liability(compounds)
    assert len(results) == 2
    assert results[0].liability_score >= results[1].liability_score


def test_screen_empty_returns_empty():
    assert screen_metabolic_liability([]) == []
