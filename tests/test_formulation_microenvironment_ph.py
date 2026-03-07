"""Tests for formulation microenvironment pH prediction (Phase 851)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.formulation_microenvironment_ph import (
    FormulationMicroenvironmentResult,
    predict_microenvironment_ph,
    screen_excipients,
)

# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = predict_microenvironment_ph(
        "DrugA", pka=5.0, logp=1.5, ionization_type="acid", formulation_type="tablet"
    )
    assert isinstance(result, FormulationMicroenvironmentResult)


def test_drug_name_preserved():
    result = predict_microenvironment_ph(
        "Ibuprofen", pka=4.4, logp=3.5, ionization_type="acid", formulation_type="tablet"
    )
    assert result.drug_name == "Ibuprofen"


def test_formulation_type_preserved():
    result = predict_microenvironment_ph(
        "X", pka=7.0, logp=1.0, ionization_type="base", formulation_type="capsule"
    )
    assert result.formulation_type == "capsule"


def test_buffer_species_preserved():
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="acid",
        formulation_type="tablet",
        buffer_species="citrate",
    )
    assert result.buffer_species == "citrate"


def test_buffer_conc_preserved():
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="acid",
        formulation_type="tablet",
        buffer_conc_mM=50.0,
    )
    assert result.buffer_conc_mM == 50.0


# ---------------------------------------------------------------------------
# pH values
# ---------------------------------------------------------------------------


def test_non_tablet_microenv_equals_bulk():
    result = predict_microenvironment_ph(
        "X", pka=5.0, logp=1.0, ionization_type="neutral", formulation_type="capsule"
    )
    assert result.microenv_ph == result.bulk_ph


def test_tablet_acidic_excipient_lowers_ph():
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="tablet",
        tablet_core_excipient="citric_acid",
    )
    assert result.microenv_ph < result.bulk_ph


def test_tablet_basic_excipient_raises_ph():
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="tablet",
        tablet_core_excipient="Na_bicarbonate",
    )
    assert result.microenv_ph > result.bulk_ph


def test_tablet_neutral_excipient_no_gradient():
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="tablet",
        tablet_core_excipient="MCC",
    )
    assert result.ph_gradient == 0.0


def test_ph_gradient_equals_abs_diff():
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="tablet",
        tablet_core_excipient="tartaric_acid",
    )
    assert abs(result.ph_gradient - abs(result.microenv_ph - result.bulk_ph)) < 1e-9


def test_microenv_ph_clamped_min():
    # citrate bulk pH 4.5 + acid offset -1.5 = 3.0, which is above min=2.0
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="tablet",
        buffer_species="citrate",
        tablet_core_excipient="citric_acid",
    )
    assert result.microenv_ph >= 2.0


def test_microenv_ph_clamped_max():
    # phosphate bulk pH 7.0 + basic offset +2.0 = 9.0, well within max=10.0
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="tablet",
        buffer_species="phosphate",
        tablet_core_excipient="Na_bicarbonate",
    )
    assert result.microenv_ph <= 10.0


# ---------------------------------------------------------------------------
# Buffer species effect on bulk pH
# ---------------------------------------------------------------------------


def test_citrate_buffer_bulk_ph():
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="capsule",
        buffer_species="citrate",
    )
    assert result.bulk_ph == pytest.approx(4.5)


def test_phosphate_buffer_bulk_ph():
    result = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="capsule",
        buffer_species="phosphate",
    )
    assert result.bulk_ph == pytest.approx(7.0)


def test_solution_no_buffer_bulk_ph():
    result = predict_microenvironment_ph(
        "X", pka=5.0, logp=1.0, ionization_type="neutral", formulation_type="solution"
    )
    assert result.bulk_ph == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Solubility
# ---------------------------------------------------------------------------


def test_solubility_positive():
    result = predict_microenvironment_ph(
        "X", pka=5.0, logp=2.0, ionization_type="acid", formulation_type="tablet"
    )
    assert result.solubility_at_microenv_mg_mL > 0.0


def test_acid_higher_solubility_at_higher_ph():
    # acid drug: higher pH → more ionised → higher solubility
    r_low = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=2.0,
        ionization_type="acid",
        formulation_type="tablet",
        buffer_species="citrate",
    )
    r_high = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=2.0,
        ionization_type="acid",
        formulation_type="capsule",
        buffer_species="phosphate",
    )
    assert r_high.solubility_at_microenv_mg_mL > r_low.solubility_at_microenv_mg_mL


def test_neutral_solubility_independent_of_ph():
    r1 = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="tablet",
        buffer_species="citrate",
    )
    r2 = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="capsule",
        buffer_species="phosphate",
    )
    assert abs(r1.solubility_at_microenv_mg_mL - r2.solubility_at_microenv_mg_mL) < 1e-6


# ---------------------------------------------------------------------------
# Dissolution rate factor
# ---------------------------------------------------------------------------


def test_dissolution_rate_factor_positive():
    result = predict_microenvironment_ph(
        "X", pka=5.0, logp=1.0, ionization_type="acid", formulation_type="tablet"
    )
    assert result.dissolution_rate_factor > 0.0


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_stability_between_0_and_100():
    result = predict_microenvironment_ph(
        "X", pka=5.0, logp=1.0, ionization_type="acid", formulation_type="tablet"
    )
    assert 0.0 <= result.drug_stability_pct <= 100.0


def test_neutral_ph_highest_stability():
    r_neutral = predict_microenvironment_ph(
        "X", pka=7.0, logp=1.0, ionization_type="neutral", formulation_type="solution"
    )
    r_acidic = predict_microenvironment_ph(
        "X",
        pka=5.0,
        logp=1.0,
        ionization_type="neutral",
        formulation_type="tablet",
        buffer_species="citrate",
        tablet_core_excipient="citric_acid",
    )
    assert r_neutral.drug_stability_pct >= r_acidic.drug_stability_pct


# ---------------------------------------------------------------------------
# Release profile
# ---------------------------------------------------------------------------


def test_release_profile_is_valid_string():
    result = predict_microenvironment_ph(
        "X", pka=5.0, logp=1.0, ionization_type="acid", formulation_type="tablet"
    )
    assert result.release_profile in {"immediate", "pH_dependent", "sustained"}


# ---------------------------------------------------------------------------
# Compatibility score
# ---------------------------------------------------------------------------


def test_compatibility_score_in_range():
    result = predict_microenvironment_ph(
        "X", pka=5.0, logp=1.0, ionization_type="acid", formulation_type="tablet"
    )
    assert 0.0 <= result.compatibility_score <= 100.0


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = predict_microenvironment_ph(
        "X", pka=5.0, logp=1.0, ionization_type="acid", formulation_type="tablet"
    )
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_pka_below_raises():
    with pytest.raises(ValueError, match="pka must be in"):
        predict_microenvironment_ph(
            "X", pka=-1.0, logp=1.0, ionization_type="acid", formulation_type="tablet"
        )


def test_invalid_pka_above_raises():
    with pytest.raises(ValueError, match="pka must be in"):
        predict_microenvironment_ph(
            "X", pka=15.0, logp=1.0, ionization_type="acid", formulation_type="tablet"
        )


def test_negative_buffer_conc_raises():
    with pytest.raises(ValueError, match="buffer_conc_mM"):
        predict_microenvironment_ph(
            "X",
            pka=5.0,
            logp=1.0,
            ionization_type="acid",
            formulation_type="tablet",
            buffer_conc_mM=-1.0,
        )


def test_invalid_formulation_type_raises():
    with pytest.raises(ValueError, match="formulation_type"):
        predict_microenvironment_ph(
            "X", pka=5.0, logp=1.0, ionization_type="acid", formulation_type="injection"
        )


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_microenvironment_ph(
            "X", pka=5.0, logp=1.0, ionization_type="zwitterion", formulation_type="tablet"
        )


def test_invalid_buffer_species_raises():
    with pytest.raises(ValueError, match="buffer_species"):
        predict_microenvironment_ph(
            "X",
            pka=5.0,
            logp=1.0,
            ionization_type="acid",
            formulation_type="tablet",
            buffer_species="tris",
        )


# ---------------------------------------------------------------------------
# screen_excipients
# ---------------------------------------------------------------------------


def test_screen_excipients_returns_list():
    results = screen_excipients(
        "DrugA",
        pka=5.0,
        logp=2.0,
        ionization_type="acid",
        tablet_core_excipients=["MCC", "citric_acid", "Na_bicarbonate"],
    )
    assert isinstance(results, list)


def test_screen_excipients_length():
    excipients = ["MCC", "citric_acid", "tartaric_acid", "Na_bicarbonate"]
    results = screen_excipients(
        "DrugA", pka=5.0, logp=2.0, ionization_type="acid", tablet_core_excipients=excipients
    )
    assert len(results) == 4


def test_screen_excipients_sorted_by_compatibility():
    excipients = ["citric_acid", "MCC", "Na_bicarbonate"]
    results = screen_excipients(
        "DrugA", pka=5.0, logp=2.0, ionization_type="acid", tablet_core_excipients=excipients
    )
    scores = [r.compatibility_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_excipients_all_tablets():
    results = screen_excipients(
        "DrugA",
        pka=5.0,
        logp=2.0,
        ionization_type="acid",
        tablet_core_excipients=["MCC", "citric_acid"],
    )
    assert all(r.formulation_type == "tablet" for r in results)
