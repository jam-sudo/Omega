"""Tests for Phase 1089 — Pharmacogenomics-Driven Dose Personalization."""

import pytest

from omega_pbpk.clinical.pgx_dose_personalization import (
    PGxDosePersonalizationResult,
    personalize_dose,
    screen_drug_pgx,
)

VALID_EFFICACY = {"normal", "reduced", "absent", "enhanced"}
VALID_TOXICITY = {"low", "moderate", "high", "very_high"}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = personalize_dose("codeine", "CYP2D6", "NM", 30.0)
    assert isinstance(result, PGxDosePersonalizationResult)


# ---------------------------------------------------------------------------
# Codeine / CYP2D6
# ---------------------------------------------------------------------------


def test_codeine_pm_dose_factor_zero():
    result = personalize_dose("codeine", "CYP2D6", "PM", 30.0)
    assert result.dose_adjustment_factor == 0.0


def test_codeine_pm_recommended_dose_zero():
    result = personalize_dose("codeine", "CYP2D6", "PM", 30.0)
    assert result.recommended_dose_mg == 0.0


def test_codeine_um_toxicity_high():
    result = personalize_dose("codeine", "CYP2D6", "UM", 30.0)
    assert result.toxicity_risk == "high"


def test_codeine_nm_standard_factor():
    result = personalize_dose("codeine", "CYP2D6", "NM", 30.0)
    assert result.dose_adjustment_factor == 1.0


def test_codeine_im_factor_reduced():
    result = personalize_dose("codeine", "CYP2D6", "IM", 30.0)
    assert result.dose_adjustment_factor == 0.75


# ---------------------------------------------------------------------------
# Clopidogrel / CYP2C19
# ---------------------------------------------------------------------------


def test_clopidogrel_pm_alternative_nonempty():
    result = personalize_dose("clopidogrel", "CYP2C19", "PM", 75.0)
    assert result.alternative_drug != ""


def test_clopidogrel_pm_factor_zero():
    result = personalize_dose("clopidogrel", "CYP2C19", "PM", 75.0)
    assert result.dose_adjustment_factor == 0.0


def test_clopidogrel_nm_factor_one():
    result = personalize_dose("clopidogrel", "CYP2C19", "NM", 75.0)
    assert result.dose_adjustment_factor == 1.0


# ---------------------------------------------------------------------------
# Mercaptopurine / TPMT
# ---------------------------------------------------------------------------


def test_mercaptopurine_pm_toxicity_very_high():
    result = personalize_dose("mercaptopurine", "TPMT", "PM", 75.0)
    assert result.toxicity_risk == "very_high"


def test_mercaptopurine_pm_factor_small():
    result = personalize_dose("mercaptopurine", "TPMT", "PM", 75.0)
    assert result.dose_adjustment_factor == 0.1


def test_mercaptopurine_im_factor_half():
    result = personalize_dose("mercaptopurine", "TPMT", "IM", 75.0)
    assert result.dose_adjustment_factor == 0.5


# ---------------------------------------------------------------------------
# Warfarin / CYP2C9
# ---------------------------------------------------------------------------


def test_warfarin_pm_factor_less_than_one():
    result = personalize_dose("warfarin", "CYP2C9", "PM", 5.0)
    assert result.dose_adjustment_factor < 1.0


def test_warfarin_pm_factor_value():
    result = personalize_dose("warfarin", "CYP2C9", "PM", 5.0)
    assert result.dose_adjustment_factor == 0.3


# ---------------------------------------------------------------------------
# Atomoxetine / CYP2D6
# ---------------------------------------------------------------------------


def test_atomoxetine_pm_toxicity_high():
    result = personalize_dose("atomoxetine", "CYP2D6", "PM", 40.0)
    assert result.toxicity_risk == "high"


def test_atomoxetine_pm_factor_reduced():
    result = personalize_dose("atomoxetine", "CYP2D6", "PM", 40.0)
    assert result.dose_adjustment_factor == 0.5


# ---------------------------------------------------------------------------
# Tamoxifen / CYP2D6
# ---------------------------------------------------------------------------


def test_tamoxifen_pm_alternative_nonempty():
    result = personalize_dose("tamoxifen", "CYP2D6", "PM", 20.0)
    assert result.alternative_drug != ""


def test_tamoxifen_pm_efficacy_absent():
    result = personalize_dose("tamoxifen", "CYP2D6", "PM", 20.0)
    assert result.efficacy_prediction == "absent"


# ---------------------------------------------------------------------------
# Validity checks across all drugs
# ---------------------------------------------------------------------------


def test_all_drugs_dose_factor_in_range():
    drug_configs = [
        ("codeine", "CYP2D6", ["PM", "IM", "NM", "UM"]),
        ("clopidogrel", "CYP2C19", ["PM", "IM", "NM", "RM", "UM"]),
        ("atomoxetine", "CYP2D6", ["PM", "IM", "NM", "UM"]),
        ("warfarin", "CYP2C9", ["PM", "IM", "NM"]),
        ("mercaptopurine", "TPMT", ["PM", "IM", "NM", "UM"]),
        ("tamoxifen", "CYP2D6", ["PM", "IM", "NM", "UM"]),
    ]
    for drug, gene, phenotypes in drug_configs:
        for ph in phenotypes:
            result = personalize_dose(drug, gene, ph, 100.0)
            assert 0.0 <= result.dose_adjustment_factor <= 3.0, (
                f"{drug}/{ph}: factor={result.dose_adjustment_factor}"
            )


def test_all_drugs_valid_efficacy():
    drug_configs = [
        ("codeine", "CYP2D6", ["PM", "IM", "NM", "UM"]),
        ("warfarin", "CYP2C9", ["PM", "IM", "NM"]),
        ("mercaptopurine", "TPMT", ["PM", "IM", "NM", "UM"]),
    ]
    for drug, gene, phenotypes in drug_configs:
        for ph in phenotypes:
            result = personalize_dose(drug, gene, ph, 100.0)
            assert result.efficacy_prediction in VALID_EFFICACY


def test_all_drugs_valid_toxicity():
    drug_configs = [
        ("atomoxetine", "CYP2D6", ["PM", "IM", "NM", "UM"]),
        ("warfarin", "CYP2C9", ["PM", "IM", "NM"]),
        ("mercaptopurine", "TPMT", ["PM", "IM", "NM", "UM"]),
    ]
    for drug, gene, phenotypes in drug_configs:
        for ph in phenotypes:
            result = personalize_dose(drug, gene, ph, 100.0)
            assert result.toxicity_risk in VALID_TOXICITY


def test_recommendation_nonempty():
    result = personalize_dose("warfarin", "CYP2C9", "NM", 5.0)
    assert result.recommendation != ""
    assert len(result.recommendation) > 5


# ---------------------------------------------------------------------------
# screen_drug_pgx
# ---------------------------------------------------------------------------


def test_screen_drug_pgx_returns_correct_count():
    phenotypes = ["PM", "IM", "NM"]
    results = screen_drug_pgx("warfarin", 5.0, phenotypes, "CYP2C9")
    assert len(results) == 3


def test_screen_drug_pgx_sorted_ascending():
    phenotypes = ["NM", "IM", "PM"]
    results = screen_drug_pgx("warfarin", 5.0, phenotypes, "CYP2C9")
    factors = [r.dose_adjustment_factor for r in results]
    assert factors == sorted(factors)


def test_screen_drug_pgx_all_phenotypes():
    phenotypes = ["PM", "IM", "NM", "UM"]
    results = screen_drug_pgx("codeine", 30.0, phenotypes, "CYP2D6")
    assert len(results) == 4
    factors = [r.dose_adjustment_factor for r in results]
    assert factors == sorted(factors)


def test_screen_drug_pgx_mercaptopurine():
    phenotypes = ["PM", "IM", "NM", "UM"]
    results = screen_drug_pgx("mercaptopurine", 75.0, phenotypes, "TPMT")
    assert results[0].dose_adjustment_factor <= results[-1].dose_adjustment_factor


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------


def test_invalid_drug_raises_value_error():
    with pytest.raises(ValueError, match="drug_name"):
        personalize_dose("aspirin", "CYP2D6", "NM", 100.0)


def test_invalid_gene_raises_value_error():
    with pytest.raises(ValueError, match="gene"):
        personalize_dose("codeine", "CYP3A4", "NM", 30.0)


def test_invalid_phenotype_raises_value_error():
    with pytest.raises(ValueError, match="phenotype"):
        personalize_dose("codeine", "CYP2D6", "XM", 30.0)


def test_invalid_phenotype_for_gene():
    # CYP2C9 only has PM/IM/NM, not RM
    with pytest.raises(ValueError, match="phenotype"):
        personalize_dose("warfarin", "CYP2C9", "RM", 5.0)


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_recommended_dose_calculation():
    result = personalize_dose("atomoxetine", "CYP2D6", "IM", 40.0)
    expected = 40.0 * 0.75
    assert abs(result.recommended_dose_mg - expected) < 1e-6


def test_cpic_level_set():
    result = personalize_dose("clopidogrel", "CYP2C19", "IM", 75.0)
    assert result.cpic_level in ("A", "B", "C")


def test_result_fields_populated():
    result = personalize_dose("tamoxifen", "CYP2D6", "NM", 20.0)
    assert result.drug_name == "tamoxifen"
    assert result.gene == "CYP2D6"
    assert result.phenotype == "NM"
    assert result.standard_dose_mg == 20.0
