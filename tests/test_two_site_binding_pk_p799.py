"""Tests for Phase 799: two-site receptor binding pharmacodynamics."""

from __future__ import annotations

import pytest

from omega_pbpk.core.two_site_binding_pk_p799 import (
    TwoSiteBindingResult,
    calculate_two_site_binding,
    scan_two_site_binding,
)

# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


def test_returns_two_site_binding_result():
    result = calculate_two_site_binding("DrugA", 1.0, 1.0, 2.0)
    assert isinstance(result, TwoSiteBindingResult)


def test_result_is_frozen():
    result = calculate_two_site_binding("DrugA", 1.0, 1.0, 2.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


def test_drug_name_stored():
    result = calculate_two_site_binding("Agonist-X", 1.0, 1.0, 2.0)
    assert result.drug_name == "Agonist-X"


def test_conc_mg_L_stored():
    result = calculate_two_site_binding("Drug", 3.5, 1.0, 2.0)
    assert result.conc_mg_L == pytest.approx(3.5)


def test_interaction_type_stored():
    result = calculate_two_site_binding("Drug", 1.0, 1.0, 2.0, interaction_type="competitive")
    assert result.interaction_type == "competitive"


# ---------------------------------------------------------------------------
# Site occupancy at EC50
# ---------------------------------------------------------------------------


def test_site1_occupancy_50pct_at_ec50():
    """At conc == EC50_site1 with Hill=1, site1_occupancy should be ~50%."""
    ec50 = 2.0
    result = calculate_two_site_binding("Drug", ec50, ec50, 10.0, hill1=1.0)
    assert result.site1_occupancy_pct == pytest.approx(50.0, rel=0.01)


def test_site2_occupancy_50pct_at_ec50_independent():
    """At conc == EC50_site2 with Hill=1 (independent), site2_occupancy ~50%."""
    ec50_2 = 3.0
    result = calculate_two_site_binding(
        "Drug", ec50_2, 100.0, ec50_2, hill1=1.0, hill2=1.0, interaction_type="independent"
    )
    assert result.site2_occupancy_pct == pytest.approx(50.0, rel=0.01)


def test_site1_occupancy_near_100_at_high_conc():
    result = calculate_two_site_binding("Drug", 1000.0, 1.0, 2.0)
    assert result.site1_occupancy_pct > 99.0


def test_site1_occupancy_zero_at_zero_conc():
    result = calculate_two_site_binding("Drug", 0.0, 1.0, 2.0)
    assert result.site1_occupancy_pct == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Combined effect
# ---------------------------------------------------------------------------


def test_combined_effect_bliss_independence():
    """Test Bliss independence: E = E1 + E2 - E1*E2/100."""
    e1 = 60.0
    e2 = 40.0
    expected = e1 + e2 - e1 * e2 / 100.0  # 76.0
    # Use saturating concentrations so occ ~ 1
    result = calculate_two_site_binding(
        "Drug", 1000.0, 1.0, 1.0, emax1_pct=e1, emax2_pct=e2, interaction_type="independent"
    )
    # At saturation, combined ~ emax1 + emax2 - emax1*emax2/100
    assert result.combined_effect_pct == pytest.approx(expected, rel=0.02)


def test_emax_combined_bliss():
    """emax_combined = emax1 + emax2 - emax1*emax2/100."""
    emax1 = 80.0
    emax2 = 60.0
    result = calculate_two_site_binding("Drug", 1.0, 1.0, 2.0, emax1_pct=emax1, emax2_pct=emax2)
    expected = emax1 + emax2 - emax1 * emax2 / 100.0
    assert result.emax_combined == pytest.approx(expected, rel=1e-6)


def test_combined_effect_increases_with_conc():
    low = calculate_two_site_binding("Drug", 0.1, 1.0, 2.0)
    high = calculate_two_site_binding("Drug", 10.0, 1.0, 2.0)
    assert high.combined_effect_pct > low.combined_effect_pct


# ---------------------------------------------------------------------------
# Allosteric positive: combined_effect > independent at same conc
# ---------------------------------------------------------------------------


def test_allosteric_positive_increases_site2_effect():
    """Positive allosteric (alpha>1) increases site2 affinity → higher combined effect."""
    c = 1.0
    ec50_1 = 1.0
    ec50_2 = 5.0
    independent = calculate_two_site_binding(
        "Drug", c, ec50_1, ec50_2, interaction_type="independent"
    )
    allosteric = calculate_two_site_binding(
        "Drug", c, ec50_1, ec50_2, interaction_type="allosteric_positive", alpha=5.0
    )
    # Positive allosteric lowers EC50_site2_eff → higher site2 occupancy → higher combined
    assert allosteric.site2_occupancy_pct > independent.site2_occupancy_pct
    assert allosteric.combined_effect_pct > independent.combined_effect_pct


def test_allosteric_negative_decreases_site2_effect():
    """Negative allosteric (alpha>1) decreases site2 affinity → lower combined effect."""
    c = 1.0
    ec50_1 = 1.0
    ec50_2 = 1.0
    independent = calculate_two_site_binding(
        "Drug", c, ec50_1, ec50_2, interaction_type="independent"
    )
    allosteric = calculate_two_site_binding(
        "Drug", c, ec50_1, ec50_2, interaction_type="allosteric_negative", alpha=5.0
    )
    assert allosteric.site2_occupancy_pct < independent.site2_occupancy_pct


# ---------------------------------------------------------------------------
# Cooperativity index
# ---------------------------------------------------------------------------


def test_cooperativity_index_positive_allosteric():
    """Positive allosteric: cooperativity_index == alpha."""
    result = calculate_two_site_binding(
        "Drug", 1.0, 1.0, 2.0, interaction_type="allosteric_positive", alpha=3.0
    )
    assert result.cooperativity_index == pytest.approx(3.0, rel=1e-6)


def test_cooperativity_index_negative_allosteric():
    """Negative allosteric: cooperativity_index == 1/alpha."""
    result = calculate_two_site_binding(
        "Drug", 1.0, 1.0, 2.0, interaction_type="allosteric_negative", alpha=4.0
    )
    assert result.cooperativity_index == pytest.approx(0.25, rel=1e-6)


def test_cooperativity_index_independent_is_1():
    result = calculate_two_site_binding("Drug", 1.0, 1.0, 2.0, interaction_type="independent")
    assert result.cooperativity_index == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Interaction type in expected set
# ---------------------------------------------------------------------------


def test_interaction_type_in_valid_set():
    valid_types = {"competitive", "allosteric_positive", "allosteric_negative", "independent"}
    for t in valid_types:
        result = calculate_two_site_binding("Drug", 1.0, 1.0, 2.0, interaction_type=t)
        assert result.interaction_type in valid_types


# ---------------------------------------------------------------------------
# scan_two_site_binding
# ---------------------------------------------------------------------------


def test_scan_returns_list():
    concs = [0.1, 1.0, 10.0]
    results = scan_two_site_binding("Drug", concs, 1.0, 2.0)
    assert isinstance(results, list)


def test_scan_returns_correct_count():
    concs = [0.1, 0.5, 1.0, 5.0, 10.0]
    results = scan_two_site_binding("Drug", concs, 1.0, 2.0)
    assert len(results) == 5


def test_scan_sorted_by_conc_ascending():
    concs = [10.0, 0.1, 5.0, 1.0]
    results = scan_two_site_binding("Drug", concs, 1.0, 2.0)
    conc_vals = [r.conc_mg_L for r in results]
    assert conc_vals == sorted(conc_vals)


def test_scan_all_two_site_binding_results():
    concs = [1.0, 2.0, 3.0]
    results = scan_two_site_binding("Drug", concs, 1.0, 2.0)
    for r in results:
        assert isinstance(r, TwoSiteBindingResult)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_conc_negative_raises():
    with pytest.raises(ValueError, match="conc_mg_L"):
        calculate_two_site_binding("Drug", -1.0, 1.0, 2.0)


def test_validation_ec50_site1_zero_raises():
    with pytest.raises(ValueError, match="ec50_site1_mg_L"):
        calculate_two_site_binding("Drug", 1.0, 0.0, 2.0)


def test_validation_ec50_site2_negative_raises():
    with pytest.raises(ValueError, match="ec50_site2_mg_L"):
        calculate_two_site_binding("Drug", 1.0, 1.0, -2.0)


def test_notes_nonempty():
    result = calculate_two_site_binding("Drug", 1.0, 1.0, 2.0)
    assert len(result.notes) > 0
