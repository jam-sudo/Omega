"""Tests for Phase 339 — Ocular Drug Bioavailability Predictor."""

import math

import pytest

from omega_pbpk.prediction.ocular_bioavailability import (
    OcularBioavailabilityResult,
    compare_formulations,
    predict_ocular_bioavailability,
)

VALID_CONC_CLASSES = {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        logP=1.5,
        mw_Da=300.0,
        pKa=7.4,
        drug_type="neutral",
    )
    defaults.update(kwargs)
    return predict_ocular_bioavailability(**defaults)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type_is_frozen_dataclass():
    result = make_result()
    assert isinstance(result, OcularBioavailabilityResult)


def test_result_is_frozen():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.logP = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Ocular bioavailability range
# ---------------------------------------------------------------------------


def test_ocular_bioavailability_pct_max_30():
    """Ocular BA should never exceed 30%."""
    result = make_result(logP=5.0, mw_Da=100.0)
    assert result.ocular_bioavailability_pct <= 30.0


def test_ocular_bioavailability_pct_positive():
    result = make_result()
    assert result.ocular_bioavailability_pct > 0.0


def test_ocular_bioavailability_pct_in_range():
    result = make_result()
    assert 0.0 < result.ocular_bioavailability_pct <= 30.0


# ---------------------------------------------------------------------------
# Physicochemical effects
# ---------------------------------------------------------------------------


def test_higher_logp_higher_permeability():
    """Higher logP → higher corneal permeability (Huang model)."""
    r_low = make_result(logP=0.0, mw_Da=300.0)
    r_high = make_result(logP=3.0, mw_Da=300.0)
    assert r_high.corneal_permeability_cm_s > r_low.corneal_permeability_cm_s


def test_higher_logp_higher_bioavailability_same_mw():
    """Higher logP → higher corneal absorption fraction (same MW, no protein binding)."""
    r_low = make_result(logP=0.0, mw_Da=300.0, protein_binding_pct=0.0)
    r_high = make_result(logP=3.0, mw_Da=300.0, protein_binding_pct=0.0)
    assert r_high.ocular_bioavailability_pct >= r_low.ocular_bioavailability_pct


def test_higher_protein_binding_lower_free_fraction():
    """Higher protein binding → lower ocular bioavailability."""
    r_low_pb = make_result(logP=2.0, mw_Da=250.0, protein_binding_pct=20.0)
    r_high_pb = make_result(logP=2.0, mw_Da=250.0, protein_binding_pct=80.0)
    assert r_high_pb.ocular_bioavailability_pct < r_low_pb.ocular_bioavailability_pct


def test_zero_protein_binding_max_free_fraction():
    """Zero protein binding → fa_cornea * 100 == ocular_bioavailability_pct."""
    result = make_result(protein_binding_pct=0.0)
    assert math.isclose(result.fa_cornea * 100.0, result.ocular_bioavailability_pct, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Precorneal retention
# ---------------------------------------------------------------------------


def test_precorneal_retention_fraction_between_0_and_1():
    result = make_result()
    assert 0.0 < result.precorneal_retention_fraction < 1.0


def test_precorneal_retention_consistent():
    """Precorneal retention = 1 - exp(-0.46 * 2) ≈ 0.600."""
    result = make_result()
    expected = 1.0 - math.exp(-0.46 * 2.0)
    assert math.isclose(result.precorneal_retention_fraction, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Systemic absorption
# ---------------------------------------------------------------------------


def test_systemic_absorption_pct_positive():
    result = make_result()
    assert result.systemic_absorption_pct > 0.0


def test_systemic_absorption_pct_fixed_35():
    """35% of instilled dose goes nasolacrimal → systemic."""
    result = make_result()
    assert math.isclose(result.systemic_absorption_pct, 35.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Aqueous humor concentration class
# ---------------------------------------------------------------------------


def test_aqueous_conc_class_valid():
    result = make_result()
    assert result.aqueous_humor_expected_conc_class in VALID_CONC_CLASSES


def test_aqueous_conc_class_low_for_very_hydrophilic():
    """Very hydrophilic + high MW drug → low aqueous concentration class."""
    result = make_result(logP=-2.0, mw_Da=800.0)
    assert result.aqueous_humor_expected_conc_class == "low"


def test_aqueous_conc_class_logic():
    """Verify classification thresholds."""
    result = make_result()
    fa = result.fa_cornea
    if fa < 0.05:
        assert result.aqueous_humor_expected_conc_class == "low"
    elif fa <= 0.15:
        assert result.aqueous_humor_expected_conc_class == "moderate"
    else:
        assert result.aqueous_humor_expected_conc_class == "high"


# ---------------------------------------------------------------------------
# Instilled volume effect
# ---------------------------------------------------------------------------


def test_larger_instilled_volume_higher_absorption():
    """Larger volume → more drug available → higher or equal absorption (up to clamp)."""
    r_small = make_result(logP=1.0, mw_Da=300.0, instilled_volume_uL=10.0)
    r_large = make_result(logP=1.0, mw_Da=300.0, instilled_volume_uL=50.0)
    assert r_large.fa_cornea >= r_small.fa_cornea


# ---------------------------------------------------------------------------
# Provided permeability
# ---------------------------------------------------------------------------


def test_provided_corneal_permeability_used():
    """When corneal_permeability_cm_s is provided, it should be used directly."""
    kp_provided = 1e-4
    result = predict_ocular_bioavailability(
        drug_name="Drug",
        logP=1.0,
        mw_Da=300.0,
        pKa=7.0,
        drug_type="neutral",
        corneal_permeability_cm_s=kp_provided,
    )
    assert math.isclose(result.corneal_permeability_cm_s, kp_provided, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_returns_list():
    results = compare_formulations(
        drug_name="Drug",
        logP=1.5,
        mw_Da=300.0,
        pKa=7.0,
        drug_type="neutral",
        instilled_volumes=[10.0, 20.0, 30.0, 50.0],
    )
    assert isinstance(results, list)
    assert len(results) == 4


def test_compare_formulations_sorted_descending():
    results = compare_formulations(
        drug_name="Drug",
        logP=1.5,
        mw_Da=300.0,
        pKa=7.0,
        drug_type="neutral",
        instilled_volumes=[10.0, 20.0, 30.0, 50.0],
    )
    pcts = [r.ocular_bioavailability_pct for r in results]
    assert pcts == sorted(pcts, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        make_result(mw_Da=0.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        make_result(mw_Da=-50.0)


def test_validation_instilled_volume_zero():
    with pytest.raises(ValueError, match="instilled_volume_uL"):
        make_result(instilled_volume_uL=0.0)


def test_validation_instilled_volume_negative():
    with pytest.raises(ValueError, match="instilled_volume_uL"):
        make_result(instilled_volume_uL=-5.0)


def test_validation_protein_binding_negative():
    with pytest.raises(ValueError, match="protein_binding_pct"):
        make_result(protein_binding_pct=-10.0)


def test_validation_protein_binding_over_100():
    with pytest.raises(ValueError, match="protein_binding_pct"):
        make_result(protein_binding_pct=110.0)


def test_validation_invalid_drug_type():
    with pytest.raises(ValueError, match="drug_type"):
        make_result(drug_type="zwitterion")
