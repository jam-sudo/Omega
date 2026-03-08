"""Tests for Phase 316: Drug Solubility in Gastrointestinal Fluids."""

import pytest

from omega_pbpk.prediction.gi_fluid_solubility import (
    GIFluidSolubilityResult,
    compare_fluids,
    predict_gi_solubility,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _lipophilic_acid(fluid_type: str = "fessif") -> GIFluidSolubilityResult:
    """Lipophilic acid drug (high logP, low pKa)."""
    return predict_gi_solubility(
        drug_name="LipophilicAcid",
        logp=3.5,
        pka=4.0,
        ionization_type="acid",
        mw=350.0,
        intrinsic_solubility_mg_mL=0.01,
        fluid_type=fluid_type,
    )


def _hydrophilic_base(fluid_type: str = "sgf") -> GIFluidSolubilityResult:
    """Hydrophilic base drug (low logP, high pKa)."""
    return predict_gi_solubility(
        drug_name="HydrophilicBase",
        logp=0.5,
        pka=9.0,
        ionization_type="base",
        mw=200.0,
        intrinsic_solubility_mg_mL=1.0,
        fluid_type=fluid_type,
    )


def _neutral_drug(fluid_type: str = "sif") -> GIFluidSolubilityResult:
    return predict_gi_solubility(
        drug_name="NeutralDrug",
        logp=2.0,
        pka=7.0,
        ionization_type="neutral",
        mw=300.0,
        intrinsic_solubility_mg_mL=0.5,
        fluid_type=fluid_type,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _lipophilic_acid()
    assert isinstance(result, GIFluidSolubilityResult)


def test_frozen_dataclass():
    result = _lipophilic_acid()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Metadata preserved
# ---------------------------------------------------------------------------


def test_fluid_type_preserved():
    result = predict_gi_solubility(
        drug_name="Drug",
        logp=2.0,
        pka=5.0,
        ionization_type="acid",
        mw=300.0,
        intrinsic_solubility_mg_mL=0.1,
        fluid_type="fassif",
    )
    assert result.fluid_type == "fassif"


def test_drug_name_preserved():
    result = _lipophilic_acid()
    assert result.drug_name == "LipophilicAcid"


def test_ionization_type_preserved():
    result = _hydrophilic_base()
    assert result.ionization_type == "base"


# ---------------------------------------------------------------------------
# Predicted solubility
# ---------------------------------------------------------------------------


def test_predicted_solubility_positive():
    result = _lipophilic_acid()
    assert result.predicted_solubility_mg_mL > 0.0


def test_predicted_solubility_for_all_fluids():
    for ft in ["sgf", "sif", "fassif", "fessif", "fassgf"]:
        result = predict_gi_solubility(
            drug_name="Drug",
            logp=2.5,
            pka=5.0,
            ionization_type="acid",
            mw=300.0,
            intrinsic_solubility_mg_mL=0.1,
            fluid_type=ft,
        )
        assert result.predicted_solubility_mg_mL > 0.0


# ---------------------------------------------------------------------------
# FeSSIF > FaSSIF for lipophilic drug
# ---------------------------------------------------------------------------


def test_fessif_greater_than_fassif_for_lipophilic():
    """FeSSIF has more bile salts and lecithin, so neutral lipophilic drug
    should have higher predicted solubility than FaSSIF."""
    # Use neutral drug so pH differences don't dominate
    fessif = predict_gi_solubility(
        drug_name="NeutralLipophilic",
        logp=4.0,
        pka=7.0,
        ionization_type="neutral",
        mw=350.0,
        intrinsic_solubility_mg_mL=0.01,
        fluid_type="fessif",
    )
    fassif = predict_gi_solubility(
        drug_name="NeutralLipophilic",
        logp=4.0,
        pka=7.0,
        ionization_type="neutral",
        mw=350.0,
        intrinsic_solubility_mg_mL=0.01,
        fluid_type="fassif",
    )
    assert fessif.predicted_solubility_mg_mL > fassif.predicted_solubility_mg_mL


# ---------------------------------------------------------------------------
# Acid drug pH effect
# ---------------------------------------------------------------------------


def test_acid_drug_higher_solubility_at_basic_vs_acidic_ph():
    """Acid drug is more soluble at higher pH (more ionized)."""
    # SIF pH 6.8 vs SGF pH 1.6; acid drug pKa=4 → ionized at pH 6.8
    sif = predict_gi_solubility(
        drug_name="AcidDrug",
        logp=0.5,  # hydrophilic to isolate pH effect
        pka=4.0,
        ionization_type="acid",
        mw=200.0,
        intrinsic_solubility_mg_mL=1.0,
        fluid_type="sif",
    )
    sgf = predict_gi_solubility(
        drug_name="AcidDrug",
        logp=0.5,
        pka=4.0,
        ionization_type="acid",
        mw=200.0,
        intrinsic_solubility_mg_mL=1.0,
        fluid_type="sgf",
    )
    assert sif.predicted_solubility_mg_mL > sgf.predicted_solubility_mg_mL


# ---------------------------------------------------------------------------
# biorelevant_enhancement_fold
# ---------------------------------------------------------------------------


def test_biorelevant_enhancement_fold_fessif_lipophilic():
    """FeSSIF should enhance solubility of lipophilic drug (fold >= 1)."""
    result = _lipophilic_acid(fluid_type="fessif")
    assert result.biorelevant_enhancement_fold >= 1.0


def test_biorelevant_enhancement_fold_positive():
    result = _lipophilic_acid()
    assert result.biorelevant_enhancement_fold > 0.0


def test_biorelevant_enhancement_fold_formula():
    result = _lipophilic_acid()
    expected = result.predicted_solubility_mg_mL / result.intrinsic_solubility_mg_mL
    assert result.biorelevant_enhancement_fold == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# BCS solubility class
# ---------------------------------------------------------------------------


def test_bcs_solubility_class_valid_values():
    for ft in ["sgf", "sif", "fassif", "fessif", "fassgf"]:
        result = predict_gi_solubility(
            drug_name="Drug",
            logp=2.0,
            pka=7.0,
            ionization_type="neutral",
            mw=300.0,
            intrinsic_solubility_mg_mL=0.5,
            fluid_type=ft,
        )
        assert result.bcs_solubility_class in {"high", "low"}


def test_highly_soluble_drug_bcs_high():
    """Very soluble drug (1 mg/mL intrinsic) with 1 mg dose should be BCS high."""
    result = predict_gi_solubility(
        drug_name="Soluble",
        logp=0.5,
        pka=7.0,
        ionization_type="neutral",
        mw=200.0,
        intrinsic_solubility_mg_mL=1.0,
        fluid_type="sif",
        dose_mg_reference=1.0,
    )
    # dose_number = 1 / (1.0 * 250) << 1 → high
    assert result.bcs_solubility_class == "high"


# ---------------------------------------------------------------------------
# compare_fluids
# ---------------------------------------------------------------------------


def test_compare_fluids_returns_list():
    results = compare_fluids(
        drug_name="Drug",
        logp=3.0,
        pka=5.0,
        ionization_type="acid",
        mw=300.0,
        intrinsic_solubility_mg_mL=0.05,
    )
    assert isinstance(results, list)


def test_compare_fluids_returns_all_fluid_types():
    results = compare_fluids(
        drug_name="Drug",
        logp=3.0,
        pka=5.0,
        ionization_type="acid",
        mw=300.0,
        intrinsic_solubility_mg_mL=0.05,
    )
    assert len(results) == 5


def test_compare_fluids_sorted_descending():
    results = compare_fluids(
        drug_name="Drug",
        logp=3.0,
        pka=5.0,
        ionization_type="acid",
        mw=300.0,
        intrinsic_solubility_mg_mL=0.05,
    )
    sols = [r.predicted_solubility_mg_mL for r in results]
    assert sols == sorted(sols, reverse=True)


def test_compare_fluids_all_gi_results():
    results = compare_fluids(
        drug_name="Drug",
        logp=3.0,
        pka=5.0,
        ionization_type="acid",
        mw=300.0,
        intrinsic_solubility_mg_mL=0.05,
    )
    for r in results:
        assert isinstance(r, GIFluidSolubilityResult)


# ---------------------------------------------------------------------------
# notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = _lipophilic_acid()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_logp_too_low():
    with pytest.raises(ValueError, match="logp"):
        predict_gi_solubility(
            drug_name="Drug",
            logp=-6.0,
            pka=7.0,
            ionization_type="neutral",
            mw=300.0,
            intrinsic_solubility_mg_mL=0.1,
            fluid_type="sif",
        )


def test_invalid_logp_too_high():
    with pytest.raises(ValueError, match="logp"):
        predict_gi_solubility(
            drug_name="Drug",
            logp=11.0,
            pka=7.0,
            ionization_type="neutral",
            mw=300.0,
            intrinsic_solubility_mg_mL=0.1,
            fluid_type="sif",
        )


def test_invalid_pka_zero():
    with pytest.raises(ValueError, match="pka"):
        predict_gi_solubility(
            drug_name="Drug",
            logp=2.0,
            pka=0.0,
            ionization_type="acid",
            mw=300.0,
            intrinsic_solubility_mg_mL=0.1,
            fluid_type="sif",
        )


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_gi_solubility(
            drug_name="Drug",
            logp=2.0,
            pka=5.0,
            ionization_type="zwitterion",
            mw=300.0,
            intrinsic_solubility_mg_mL=0.1,
            fluid_type="sif",
        )


def test_invalid_intrinsic_solubility_zero():
    with pytest.raises(ValueError, match="intrinsic"):
        predict_gi_solubility(
            drug_name="Drug",
            logp=2.0,
            pka=5.0,
            ionization_type="acid",
            mw=300.0,
            intrinsic_solubility_mg_mL=0.0,
            fluid_type="sif",
        )


def test_invalid_fluid_type():
    with pytest.raises(ValueError, match="fluid_type"):
        predict_gi_solubility(
            drug_name="Drug",
            logp=2.0,
            pka=5.0,
            ionization_type="acid",
            mw=300.0,
            intrinsic_solubility_mg_mL=0.1,
            fluid_type="unknown_fluid",
        )
