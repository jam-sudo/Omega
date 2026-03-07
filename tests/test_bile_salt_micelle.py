"""Tests for bile salt micelle solubilization model (Phase 333)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.bile_salt_micelle import (
    MicelleResult,
    calculate_micelle_solubility,
    compare_fed_fasted,
)

# ── Input validation ──────────────────────────────────────────────────────────


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw must be > 0"):
        calculate_micelle_solubility("Drug", logP=2.0, mw=0.0, solubility_aqueous_mg_mL=0.1)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw must be > 0"):
        calculate_micelle_solubility("Drug", logP=2.0, mw=-300.0, solubility_aqueous_mg_mL=0.1)


def test_solubility_zero_raises():
    with pytest.raises(ValueError, match="solubility_aqueous_mg_mL must be > 0"):
        calculate_micelle_solubility("Drug", logP=2.0, mw=300.0, solubility_aqueous_mg_mL=0.0)


def test_solubility_negative_raises():
    with pytest.raises(ValueError, match="solubility_aqueous_mg_mL must be > 0"):
        calculate_micelle_solubility("Drug", logP=2.0, mw=300.0, solubility_aqueous_mg_mL=-1.0)


def test_bile_salt_conc_negative_raises():
    with pytest.raises(ValueError, match="bile_salt_conc_mM must be >= 0"):
        calculate_micelle_solubility(
            "Drug", logP=2.0, mw=300.0, solubility_aqueous_mg_mL=0.1, bile_salt_conc_mM=-1.0
        )


# ── Zero bile salt ────────────────────────────────────────────────────────────


def test_zero_bile_salt_enhancement_ratio_is_one():
    result = calculate_micelle_solubility(
        "ZeroBile", logP=3.0, mw=300.0, solubility_aqueous_mg_mL=0.5, bile_salt_conc_mM=0.0
    )
    assert result.enhancement_ratio == pytest.approx(1.0, abs=1e-9)


def test_zero_bile_salt_s_eff_equals_s_aq():
    result = calculate_micelle_solubility(
        "ZeroBile", logP=3.0, mw=300.0, solubility_aqueous_mg_mL=0.5, bile_salt_conc_mM=0.0
    )
    assert result.solubility_micellar_mg_mL == pytest.approx(
        result.solubility_aqueous_mg_mL, rel=1e-9
    )


def test_zero_bile_salt_fa_increase_is_zero():
    result = calculate_micelle_solubility(
        "ZeroBile", logP=1.0, mw=300.0, solubility_aqueous_mg_mL=1.0, bile_salt_conc_mM=0.0
    )
    assert result.absorption_fraction_increase_pct == pytest.approx(0.0, abs=1e-9)


# ── High logP: large enhancement ─────────────────────────────────────────────


def test_high_logP_large_enhancement():
    """logP=6 should produce enhancement_ratio > 5."""
    result = calculate_micelle_solubility(
        "HighLogP", logP=6.0, mw=400.0, solubility_aqueous_mg_mL=0.01
    )
    assert result.enhancement_ratio > 5.0


def test_high_logP_s_eff_greater_than_s_aq():
    result = calculate_micelle_solubility(
        "HighLogP", logP=4.0, mw=400.0, solubility_aqueous_mg_mL=0.01
    )
    assert result.solubility_micellar_mg_mL > result.solubility_aqueous_mg_mL


def test_high_logP_fa_increase_capped_at_50():
    """For very high logP the fa increase should be capped at 50%."""
    result = calculate_micelle_solubility(
        "VeryHighLogP", logP=8.0, mw=500.0, solubility_aqueous_mg_mL=0.001
    )
    assert result.absorption_fraction_increase_pct <= 50.0


# ── Low logP: small enhancement ───────────────────────────────────────────────


def test_low_logP_small_enhancement():
    """logP=0 should produce enhancement_ratio < 2."""
    result = calculate_micelle_solubility(
        "LowLogP", logP=0.0, mw=200.0, solubility_aqueous_mg_mL=10.0
    )
    assert result.enhancement_ratio < 2.0


def test_negative_logP_enhancement_near_one():
    """Very negative logP → Km very small → enhancement ≈ 1."""
    result = calculate_micelle_solubility(
        "NegLogP", logP=-3.0, mw=150.0, solubility_aqueous_mg_mL=5.0
    )
    assert result.enhancement_ratio < 1.5


# ── Fed vs fasted ────────────────────────────────────────────────────────────


def test_compare_fed_fasted_returns_dict_with_keys():
    result = compare_fed_fasted("Drug", logP=3.0, mw=350.0, solubility_aqueous_mg_mL=0.1)
    assert "fasted" in result
    assert "fed" in result
    assert "enhancement_ratio_fed_vs_fasted" in result


def test_fed_has_higher_enhancement_than_fasted():
    result = compare_fed_fasted("Drug", logP=3.0, mw=350.0, solubility_aqueous_mg_mL=0.1)
    assert result["fed"].enhancement_ratio > result["fasted"].enhancement_ratio


def test_fed_higher_s_eff_than_fasted():
    result = compare_fed_fasted("Drug", logP=2.5, mw=300.0, solubility_aqueous_mg_mL=0.2)
    assert result["fed"].solubility_micellar_mg_mL > result["fasted"].solubility_micellar_mg_mL


def test_fed_vs_fasted_ratio_positive():
    result = compare_fed_fasted("Drug", logP=3.0, mw=350.0, solubility_aqueous_mg_mL=0.1)
    assert result["enhancement_ratio_fed_vs_fasted"] > 1.0


# ── Linear scaling with bile salt concentration ───────────────────────────────


def test_linear_scaling_with_bile_salt():
    """Doubling bile salt concentration → proportionally higher S_eff."""
    s_aq = 0.5
    logP = 3.0
    mw = 300.0
    r1 = calculate_micelle_solubility(
        "Drug", logP=logP, mw=mw, solubility_aqueous_mg_mL=s_aq, bile_salt_conc_mM=3.0
    )
    r2 = calculate_micelle_solubility(
        "Drug", logP=logP, mw=mw, solubility_aqueous_mg_mL=s_aq, bile_salt_conc_mM=6.0
    )
    # S_eff = S_aq*(1 + Km*phi), so (S2-S_aq)/(S1-S_aq) should ≈ 2
    ratio = (r2.solubility_micellar_mg_mL - s_aq) / (r1.solubility_micellar_mg_mL - s_aq)
    assert ratio == pytest.approx(2.0, rel=1e-6)


# ── Dataclass properties ──────────────────────────────────────────────────────


def test_result_is_frozen_dataclass():
    result = calculate_micelle_solubility(
        "Aspirin", logP=1.19, mw=180.16, solubility_aqueous_mg_mL=3.0
    )
    assert isinstance(result, MicelleResult)
    with pytest.raises((AttributeError, TypeError)):
        result.logP = 5.0  # type: ignore[misc]


def test_result_fields_accessible():
    result = calculate_micelle_solubility(
        "Aspirin", logP=1.19, mw=180.16, solubility_aqueous_mg_mL=3.0
    )
    assert result.drug_name == "Aspirin"
    assert result.logP == pytest.approx(1.19)
    assert result.solubility_aqueous_mg_mL == pytest.approx(3.0)
    assert result.bile_salt_conc_mM == pytest.approx(3.0)


def test_notes_contains_drug_name():
    result = calculate_micelle_solubility(
        "TestDrugXYZ", logP=2.0, mw=300.0, solubility_aqueous_mg_mL=0.5
    )
    assert "TestDrugXYZ" in result.notes


def test_micellar_solubility_greater_than_aqueous_for_positive_logP():
    result = calculate_micelle_solubility(
        "Ibuprofen", logP=3.5, mw=206.28, solubility_aqueous_mg_mL=0.021
    )
    assert result.solubility_micellar_mg_mL > result.solubility_aqueous_mg_mL


def test_enhancement_ratio_computed_correctly():
    """Manually verify: log(Km)=0.69*2-0.32=1.06, Km=11.48, phi=3*0.001*0.6=0.0018."""
    logP = 2.0
    km = 10 ** (0.69 * logP - 0.32)
    phi = 3.0 * 0.001 * 0.6
    expected_ratio = 1.0 + km * phi
    result = calculate_micelle_solubility(
        "Check", logP=logP, mw=300.0, solubility_aqueous_mg_mL=1.0, bile_salt_conc_mM=3.0
    )
    assert result.enhancement_ratio == pytest.approx(expected_ratio, rel=1e-6)


def test_temperature_parameter_accepted():
    """temperature_C is accepted without affecting results."""
    r1 = calculate_micelle_solubility(
        "Drug", logP=2.0, mw=300.0, solubility_aqueous_mg_mL=1.0, temperature_C=25.0
    )
    r2 = calculate_micelle_solubility(
        "Drug", logP=2.0, mw=300.0, solubility_aqueous_mg_mL=1.0, temperature_C=37.0
    )
    assert r1.enhancement_ratio == pytest.approx(r2.enhancement_ratio, rel=1e-9)


def test_fa_increase_positive_for_high_logP():
    result = calculate_micelle_solubility("Drug", logP=4.0, mw=400.0, solubility_aqueous_mg_mL=0.01)
    assert result.absorption_fraction_increase_pct > 0.0
