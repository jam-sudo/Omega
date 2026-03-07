"""Tests for Phase 274: two-site receptor binding model."""

from __future__ import annotations

import math

import pytest
from omega_pbpk.core.two_site_binding import (
    TwoSiteBindingResult,
    evaluate_two_site_binding,
    two_site_occupancy,
)


# ---------------------------------------------------------------------------
# two_site_occupancy validation
# ---------------------------------------------------------------------------

def test_occupancy_c_negative():
    with pytest.raises(ValueError, match="c must be >= 0"):
        two_site_occupancy(-1.0, kd1=1.0, kd2=2.0)


def test_occupancy_kd1_zero():
    with pytest.raises(ValueError, match="kd1"):
        two_site_occupancy(1.0, kd1=0, kd2=2.0)


def test_occupancy_kd1_negative():
    with pytest.raises(ValueError, match="kd1"):
        two_site_occupancy(1.0, kd1=-1.0, kd2=2.0)


def test_occupancy_kd2_zero():
    with pytest.raises(ValueError, match="kd2"):
        two_site_occupancy(1.0, kd1=1.0, kd2=0)


def test_occupancy_kd2_negative():
    with pytest.raises(ValueError, match="kd2"):
        two_site_occupancy(1.0, kd1=1.0, kd2=-2.0)


def test_occupancy_alpha_zero():
    with pytest.raises(ValueError, match="alpha"):
        two_site_occupancy(1.0, kd1=1.0, kd2=2.0, alpha=0)


def test_occupancy_alpha_negative():
    with pytest.raises(ValueError, match="alpha"):
        two_site_occupancy(1.0, kd1=1.0, kd2=2.0, alpha=-0.5)


# ---------------------------------------------------------------------------
# two_site_occupancy correctness
# ---------------------------------------------------------------------------

def test_occupancy_zero_concentration():
    occ1, occ2 = two_site_occupancy(0.0, kd1=1.0, kd2=2.0)
    assert occ1 == 0.0
    assert occ2 == 0.0


def test_occupancy_in_range():
    occ1, occ2 = two_site_occupancy(1.0, kd1=1.0, kd2=2.0)
    assert 0.0 <= occ1 <= 1.0
    assert 0.0 <= occ2 <= 1.0


def test_occupancy_high_concentration_approaches_one():
    occ1, occ2 = two_site_occupancy(1e6, kd1=1.0, kd2=2.0)
    assert occ1 > 0.99
    assert occ2 > 0.99


def test_positive_cooperativity_increases_occ2():
    # With alpha > 1, effective kd2 decreases -> occ2 should be higher
    occ1_base, occ2_base = two_site_occupancy(1.0, kd1=1.0, kd2=2.0, alpha=1.0)
    occ1_coop, occ2_coop = two_site_occupancy(1.0, kd1=1.0, kd2=2.0, alpha=5.0)
    assert occ2_coop > occ2_base


def test_site1_occupancy_standard_hill():
    # Site 1 is simple Hill equation with n=1
    c, kd1 = 2.0, 1.0
    occ1, _ = two_site_occupancy(c, kd1=kd1, kd2=5.0)
    expected = c / (kd1 + c)
    assert occ1 == pytest.approx(expected)


# ---------------------------------------------------------------------------
# evaluate_two_site_binding validation
# ---------------------------------------------------------------------------

def test_eval_kd1_zero():
    with pytest.raises(ValueError, match="kd1"):
        evaluate_two_site_binding("Drug", kd1=0, kd2=1.0)


def test_eval_kd2_negative():
    with pytest.raises(ValueError, match="kd2"):
        evaluate_two_site_binding("Drug", kd1=1.0, kd2=-1.0)


def test_eval_emax1_zero():
    with pytest.raises(ValueError, match="emax1"):
        evaluate_two_site_binding("Drug", kd1=1.0, kd2=2.0, emax1=0)


def test_eval_emax1_too_large():
    with pytest.raises(ValueError, match="emax1"):
        evaluate_two_site_binding("Drug", kd1=1.0, kd2=2.0, emax1=1.5)


def test_eval_emax2_negative():
    with pytest.raises(ValueError, match="emax2"):
        evaluate_two_site_binding("Drug", kd1=1.0, kd2=2.0, emax2=-0.1)


def test_eval_alpha_zero():
    with pytest.raises(ValueError, match="alpha"):
        evaluate_two_site_binding("Drug", kd1=1.0, kd2=2.0, alpha=0)


def test_eval_c_min_zero():
    with pytest.raises(ValueError, match="c_min"):
        evaluate_two_site_binding("Drug", kd1=1.0, kd2=2.0, c_min=0)


def test_eval_c_max_le_c_min():
    with pytest.raises(ValueError, match="c_max"):
        evaluate_two_site_binding("Drug", kd1=1.0, kd2=2.0, c_min=10.0, c_max=5.0)


def test_eval_n_points_lt_2():
    with pytest.raises(ValueError, match="n_points"):
        evaluate_two_site_binding("Drug", kd1=1.0, kd2=2.0, n_points=1)


# ---------------------------------------------------------------------------
# evaluate_two_site_binding correctness
# ---------------------------------------------------------------------------

def test_concentrations_length_equals_n_points():
    result = evaluate_two_site_binding("Drug", kd1=1.0, kd2=5.0, n_points=30)
    assert len(result.concentrations) == 30
    assert len(result.site1_occupancy) == 30
    assert len(result.site2_occupancy) == 30
    assert len(result.total_effect) == 30


def test_site1_occupancy_in_range():
    result = evaluate_two_site_binding("Drug", kd1=1.0, kd2=5.0)
    assert all(0.0 <= o <= 1.0 for o in result.site1_occupancy)


def test_site2_occupancy_in_range():
    result = evaluate_two_site_binding("Drug", kd1=1.0, kd2=5.0)
    assert all(0.0 <= o <= 1.0 for o in result.site2_occupancy)


def test_total_effect_bounded():
    emax1, emax2 = 0.6, 0.4
    result = evaluate_two_site_binding("Drug", kd1=1.0, kd2=5.0, emax1=emax1, emax2=emax2)
    max_possible = emax1 + emax2
    assert all(0.0 <= e <= max_possible + 1e-9 for e in result.total_effect)


def test_ec50_apparent_positive():
    result = evaluate_two_site_binding("Drug", kd1=1.0, kd2=5.0)
    assert result.ec50_apparent > 0


def test_hill_apparent_positive():
    result = evaluate_two_site_binding("Drug", kd1=1.0, kd2=5.0)
    assert result.hill_apparent > 0


def test_drug_name_stored():
    result = evaluate_two_site_binding("Imatinib", kd1=0.5, kd2=10.0)
    assert result.drug_name == "Imatinib"


def test_notes_contain_drug_name():
    result = evaluate_two_site_binding("Gefitinib", kd1=0.5, kd2=10.0)
    assert "Gefitinib" in result.notes


def test_emax2_zero_allowed():
    # emax2=0 should not raise (emax2 in [0, 1])
    result = evaluate_two_site_binding("Drug", kd1=1.0, kd2=5.0, emax2=0.0)
    assert all(e >= 0 for e in result.total_effect)


def test_concentrations_log_spaced():
    result = evaluate_two_site_binding(
        "Drug", kd1=1.0, kd2=5.0, c_min=1e-3, c_max=1e3, n_points=7
    )
    concs = result.concentrations
    assert concs[0] == pytest.approx(1e-3, rel=1e-6)
    assert concs[-1] == pytest.approx(1e3, rel=1e-6)
