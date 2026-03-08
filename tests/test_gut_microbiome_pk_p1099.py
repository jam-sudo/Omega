"""Tests for gut microbiome PK model — Phase 1099."""

from __future__ import annotations

import pytest

from omega_pbpk.core.gut_microbiome_pk_p1099 import (
    GutMicrobiomePKResult,
    compare_colonization_levels,
    simulate_microbiome_pk,
)

# ── basic return type and structure ──────────────────────────────────────────


def test_returns_correct_type():
    result = simulate_microbiome_pk("TestDrug", 100.0, "azoreductase")
    assert isinstance(result, GutMicrobiomePKResult)


def test_fields_present():
    result = simulate_microbiome_pk("TestDrug", 100.0, "azoreductase")
    assert result.drug_name == "TestDrug"
    assert result.enzymatic_reaction == "azoreductase"
    assert result.colonization_level == "normal"
    assert result.transit_h == 24.0


def test_times_list_length():
    result = simulate_microbiome_pk("X", 50.0, "none", t_end_h=10.0, dt_h=0.5)
    expected = int(10.0 / 0.5) + 1
    assert len(result.times_h) == expected


def test_plasma_starts_at_zero():
    result = simulate_microbiome_pk("X", 100.0, "azoreductase")
    assert result.c_plasma_mg_L[0] == 0.0


def test_prodrug_starts_at_dose():
    dose = 200.0
    result = simulate_microbiome_pk("X", dose, "azoreductase")
    assert abs(result.a_prodrug_mg[0] - dose) < 1e-6


def test_active_starts_at_zero():
    result = simulate_microbiome_pk("X", 100.0, "azoreductase")
    assert result.a_active_mg[0] == 0.0


# ── fraction_activated ────────────────────────────────────────────────────────


def test_fraction_activated_in_range():
    result = simulate_microbiome_pk("X", 100.0, "azoreductase")
    assert 0.0 <= result.fraction_activated <= 1.0


def test_none_enzyme_fraction_activated_near_zero():
    result = simulate_microbiome_pk("X", 100.0, "none")
    assert result.fraction_activated < 1e-9


def test_higher_colonization_higher_fraction():
    dysbiotic = simulate_microbiome_pk("X", 100.0, "azoreductase", colonization_level="dysbiotic")
    normal = simulate_microbiome_pk("X", 100.0, "azoreductase", colonization_level="normal")
    enriched = simulate_microbiome_pk("X", 100.0, "azoreductase", colonization_level="enriched")
    assert dysbiotic.fraction_activated < normal.fraction_activated < enriched.fraction_activated


# ── AUC and Cmax ──────────────────────────────────────────────────────────────


def test_auc_positive_with_enzyme():
    result = simulate_microbiome_pk("X", 100.0, "azoreductase")
    assert result.auc_plasma > 0.0


def test_auc_near_zero_no_enzyme():
    result = simulate_microbiome_pk("X", 100.0, "none")
    assert result.auc_plasma < 1e-9


def test_cmax_positive_with_enzyme():
    result = simulate_microbiome_pk("X", 100.0, "beta_glucuronidase")
    assert result.cmax_plasma > 0.0


# ── microbiome_contribution_pct ───────────────────────────────────────────────


def test_microbiome_contribution_in_range():
    result = simulate_microbiome_pk("X", 100.0, "sulfatase")
    assert 0.0 <= result.microbiome_contribution_pct <= 100.0


def test_microbiome_contribution_zero_for_none():
    result = simulate_microbiome_pk("X", 100.0, "none")
    assert result.microbiome_contribution_pct == 0.0


# ── dysbiosis_risk ────────────────────────────────────────────────────────────


def test_dysbiosis_risk_valid_values():
    for reaction in ["none", "sulfatase", "beta_glucuronidase"]:
        result = simulate_microbiome_pk("X", 100.0, reaction)
        assert result.dysbiosis_risk in {"low", "moderate", "high"}


def test_dysbiosis_risk_low_for_none():
    result = simulate_microbiome_pk("X", 100.0, "none")
    assert result.dysbiosis_risk == "low"


def test_dysbiosis_risk_high_enriched_potent():
    result = simulate_microbiome_pk("X", 100.0, "azoreductase", colonization_level="enriched")
    assert result.dysbiosis_risk == "high"


# ── compare_colonization_levels ───────────────────────────────────────────────


def test_compare_returns_three_results():
    results = compare_colonization_levels("X", 100.0, "azoreductase")
    assert len(results) == 3


def test_compare_sorted_descending():
    results = compare_colonization_levels("X", 100.0, "nitroreductase")
    for i in range(len(results) - 1):
        assert results[i].fraction_activated >= results[i + 1].fraction_activated


def test_compare_enriched_highest():
    results = compare_colonization_levels("X", 100.0, "azoreductase")
    # enriched should be first (highest fraction_activated)
    assert results[0].colonization_level == "enriched"


def test_compare_dysbiotic_lowest():
    results = compare_colonization_levels("X", 100.0, "azoreductase")
    assert results[-1].colonization_level == "dysbiotic"


def test_compare_all_levels_present():
    results = compare_colonization_levels("X", 100.0, "beta_glucuronidase")
    levels = {r.colonization_level for r in results}
    assert levels == {"dysbiotic", "normal", "enriched"}


# ── validation errors ─────────────────────────────────────────────────────────


def test_invalid_enzymatic_reaction():
    with pytest.raises(ValueError, match="enzymatic_reaction"):
        simulate_microbiome_pk("X", 100.0, "unknown_enzyme")


def test_invalid_colonization_level():
    with pytest.raises(ValueError, match="colonization_level"):
        simulate_microbiome_pk("X", 100.0, "azoreductase", colonization_level="very_high")


def test_invalid_transit_too_short():
    with pytest.raises(ValueError, match="colon_transit_h"):
        simulate_microbiome_pk("X", 100.0, "azoreductase", colon_transit_h=1.0)


def test_invalid_transit_too_long():
    with pytest.raises(ValueError, match="colon_transit_h"):
        simulate_microbiome_pk("X", 100.0, "azoreductase", colon_transit_h=100.0)


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_microbiome_pk("X", 0.0, "azoreductase")


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_microbiome_pk("X", -50.0, "azoreductase")


# ── transit time effect ───────────────────────────────────────────────────────


def test_longer_transit_more_activation():
    short = simulate_microbiome_pk("X", 100.0, "azoreductase", colon_transit_h=8.0)
    long = simulate_microbiome_pk("X", 100.0, "azoreductase", colon_transit_h=48.0)
    assert long.fraction_activated > short.fraction_activated


# ── all enzyme types work ─────────────────────────────────────────────────────


def test_all_enzyme_types_run():
    for reaction in ["beta_glucuronidase", "azoreductase", "sulfatase", "nitroreductase", "none"]:
        result = simulate_microbiome_pk("X", 100.0, reaction)
        assert isinstance(result, GutMicrobiomePKResult)
