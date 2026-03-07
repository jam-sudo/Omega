"""Tests for Phase 262: plasma_protein_competition.py"""

import pytest

from omega_pbpk.core.plasma_protein_competition import (
    assess_ppb_competition,
    batch_ppb_screen,
)

# ---------------------------------------------------------------------------
# assess_ppb_competition — input validation
# ---------------------------------------------------------------------------


def test_fu_a_zero_raises():
    with pytest.raises(ValueError, match="fu_a must be in"):
        assess_ppb_competition("DrugA", 0.0, "DrugB", 0.5, 10.0)


def test_fu_a_negative_raises():
    with pytest.raises(ValueError, match="fu_a must be in"):
        assess_ppb_competition("DrugA", -0.1, "DrugB", 0.5, 10.0)


def test_fu_a_above_one_raises():
    with pytest.raises(ValueError, match="fu_a must be in"):
        assess_ppb_competition("DrugA", 1.1, "DrugB", 0.5, 10.0)


def test_fu_b_zero_raises():
    with pytest.raises(ValueError, match="fu_b must be in"):
        assess_ppb_competition("DrugA", 0.1, "DrugB", 0.0, 10.0)


def test_fu_b_negative_raises():
    with pytest.raises(ValueError, match="fu_b must be in"):
        assess_ppb_competition("DrugA", 0.1, "DrugB", -0.1, 10.0)


def test_c_b_zero_raises():
    with pytest.raises(ValueError, match="c_b_total_mg_L must be > 0"):
        assess_ppb_competition("DrugA", 0.1, "DrugB", 0.5, 0.0)


def test_c_b_negative_raises():
    with pytest.raises(ValueError, match="c_b_total_mg_L must be > 0"):
        assess_ppb_competition("DrugA", 0.1, "DrugB", 0.5, -5.0)


def test_c_protein_zero_raises():
    with pytest.raises(ValueError, match="c_protein_g_dL must be > 0"):
        assess_ppb_competition("DrugA", 0.1, "DrugB", 0.5, 10.0, c_protein_g_dL=0.0)


def test_invalid_protein_type_raises():
    with pytest.raises(ValueError, match="protein_type must be"):
        assess_ppb_competition("DrugA", 0.1, "DrugB", 0.5, 10.0, protein_type="globulin")


# ---------------------------------------------------------------------------
# assess_ppb_competition — no competition (fu_b = 1.0: drug B fully unbound)
# ---------------------------------------------------------------------------


def test_no_competition_when_b_fully_unbound():
    """If drug B is fully unbound (fu_b=1.0), it cannot displace drug A."""
    result = assess_ppb_competition("DrugA", 0.05, "DrugB", 1.0, 50.0)
    # fu_b=1 → fb_b=0 → occupancy_b=0 → fu_a_displaced = fu_a
    assert abs(result.fu_a_displaced - result.fu_a_baseline) < 1e-9
    assert result.displacement_pct < 1.0


# ---------------------------------------------------------------------------
# assess_ppb_competition — displacement occurs with highly bound drug B
# ---------------------------------------------------------------------------


def test_displacement_increases_fu_a():
    """Highly protein-bound drug B (fu_b=0.01) should displace drug A."""
    result = assess_ppb_competition("Warfarin", 0.01, "Salicylate", 0.01, 200.0)
    assert result.fu_a_displaced > result.fu_a_baseline


def test_fu_ratio_ge_one():
    result = assess_ppb_competition("DrugA", 0.05, "DrugB", 0.02, 100.0)
    assert result.fu_ratio >= 1.0


def test_fu_a_displaced_in_range():
    result = assess_ppb_competition("DrugA", 0.05, "DrugB", 0.02, 100.0)
    assert 0 < result.fu_a_displaced <= 1.0


def test_displacement_pct_non_negative():
    result = assess_ppb_competition("DrugA", 0.1, "DrugB", 0.1, 50.0)
    assert result.displacement_pct >= 0.0


# ---------------------------------------------------------------------------
# assess_ppb_competition — clinical significance categories
# ---------------------------------------------------------------------------


def test_clinical_significance_negligible():
    # Minimal displacement
    result = assess_ppb_competition("DrugA", 0.5, "DrugB", 1.0, 10.0)
    assert result.clinical_significance == "negligible"


def test_clinical_significance_valid_value():
    result = assess_ppb_competition("DrugA", 0.05, "DrugB", 0.05, 50.0)
    assert result.clinical_significance in {"negligible", "minor", "moderate", "major"}


def test_high_displacement_significant():
    """Extreme concentrations of a highly bound drug should produce significant displacement."""
    result = assess_ppb_competition("DrugA", 0.01, "DrugB", 0.01, 5000.0, c_protein_g_dL=4.0)
    assert result.clinical_significance in {"moderate", "major"}


# ---------------------------------------------------------------------------
# assess_ppb_competition — protein type
# ---------------------------------------------------------------------------


def test_albumin_protein_type():
    result = assess_ppb_competition("DrugA", 0.05, "DrugB", 0.05, 50.0, protein_type="albumin")
    assert result.protein_type == "albumin"


def test_aag_protein_type():
    result = assess_ppb_competition("DrugA", 0.05, "DrugB", 0.05, 50.0, protein_type="aag")
    assert result.protein_type == "aag"


# ---------------------------------------------------------------------------
# batch_ppb_screen
# ---------------------------------------------------------------------------


def test_batch_sorted_descending():
    perpetrators = [
        {"name": "DrugB_weak", "fu_b": 0.9, "c_b_total_mg_L": 10.0},
        {"name": "DrugB_strong", "fu_b": 0.01, "c_b_total_mg_L": 200.0},
        {"name": "DrugB_medium", "fu_b": 0.2, "c_b_total_mg_L": 50.0},
    ]
    results = batch_ppb_screen("DrugA", 0.05, perpetrators)
    pcts = [r.displacement_pct for r in results]
    assert pcts == sorted(pcts, reverse=True)


def test_batch_empty_returns_empty():
    results = batch_ppb_screen("DrugA", 0.05, [])
    assert results == []


def test_batch_returns_correct_count():
    perpetrators = [
        {"name": f"Drug{i}", "fu_b": 0.1, "c_b_total_mg_L": 10.0 * (i + 1)} for i in range(5)
    ]
    results = batch_ppb_screen("DrugA", 0.05, perpetrators)
    assert len(results) == 5


def test_batch_aag_protein_type():
    perpetrators = [{"name": "DrugB", "fu_b": 0.1, "c_b_total_mg_L": 20.0}]
    results = batch_ppb_screen("DrugA", 0.1, perpetrators, protein_type="aag")
    assert results[0].protein_type == "aag"
