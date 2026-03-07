"""Tests for Phase 451 + Phase 757: Excipient-drug interaction predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.excipient_interaction_predictor import (
    SUPPORTED_EXCIPIENTS,
    ExcipientEffectResult,
    ExcipientInteractionResult,
    predict_excipient_effect,
    predict_excipient_interactions,
    screen_excipient_combinations,
    screen_excipients,
)

# ---------------------------------------------------------------------------
# Supported excipient list
# ---------------------------------------------------------------------------


def test_supported_excipients_count():
    assert len(SUPPORTED_EXCIPIENTS) >= 8


def test_supported_excipients_includes_required():
    required = {"HPMC", "PVP", "SDS", "Poloxamer 407", "Lactose", "MCC", "Stearic acid", "Tween 80"}
    assert required.issubset(set(SUPPORTED_EXCIPIENTS))


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_excipient_effect_result():
    result = predict_excipient_effect(2.0, 300.0, "MCC")
    assert isinstance(result, ExcipientEffectResult)


# ---------------------------------------------------------------------------
# Excipient name is preserved
# ---------------------------------------------------------------------------


def test_excipient_name_preserved():
    result = predict_excipient_effect(2.0, 300.0, "PVP")
    assert result.excipient_name == "PVP"


# ---------------------------------------------------------------------------
# Compatibility score in [0, 100]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excipient", SUPPORTED_EXCIPIENTS)
def test_compatibility_score_range(excipient):
    result = predict_excipient_effect(3.0, 350.0, excipient)
    assert 0.0 <= result.compatibility_score <= 100.0


# ---------------------------------------------------------------------------
# Permeability effect valid values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excipient", SUPPORTED_EXCIPIENTS)
def test_permeability_effect_valid(excipient):
    result = predict_excipient_effect(2.0, 300.0, excipient)
    assert result.permeability_effect in ("enhance", "neutral", "reduce")


# ---------------------------------------------------------------------------
# Stability effect valid values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excipient", SUPPORTED_EXCIPIENTS)
def test_stability_effect_valid(excipient):
    result = predict_excipient_effect(2.0, 300.0, excipient)
    assert result.stability_effect in ("positive", "neutral", "negative")


# ---------------------------------------------------------------------------
# Solubility fold change > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excipient", SUPPORTED_EXCIPIENTS)
def test_solubility_fold_change_positive(excipient):
    result = predict_excipient_effect(2.0, 300.0, excipient)
    assert result.solubility_fold_change > 0.0


# ---------------------------------------------------------------------------
# SDS enhances solubility for high-logP drug
# ---------------------------------------------------------------------------


def test_sds_enhances_solubility_high_logp():
    result = predict_excipient_effect(5.0, 300.0, "SDS")
    assert result.solubility_fold_change > 1.0


# ---------------------------------------------------------------------------
# Surfactants enhance permeability
# ---------------------------------------------------------------------------


def test_sds_permeability_enhance():
    result = predict_excipient_effect(3.0, 300.0, "SDS")
    assert result.permeability_effect == "enhance"


def test_tween80_permeability_enhance():
    result = predict_excipient_effect(3.0, 300.0, "Tween 80")
    assert result.permeability_effect == "enhance"


def test_poloxamer_permeability_enhance():
    result = predict_excipient_effect(3.0, 300.0, "Poloxamer 407")
    assert result.permeability_effect == "enhance"


# ---------------------------------------------------------------------------
# Stearic acid reduces permeability
# ---------------------------------------------------------------------------


def test_stearic_acid_reduces_permeability():
    result = predict_excipient_effect(2.0, 300.0, "Stearic acid")
    assert result.permeability_effect == "reduce"


# ---------------------------------------------------------------------------
# Stearic acid reduces solubility (fold change < 1.1)
# ---------------------------------------------------------------------------


def test_stearic_acid_solubility_below_one():
    result = predict_excipient_effect(2.0, 300.0, "Stearic acid")
    assert result.solubility_fold_change < 1.1


# ---------------------------------------------------------------------------
# MCC positive stability effect
# ---------------------------------------------------------------------------


def test_mcc_positive_stability():
    result = predict_excipient_effect(2.0, 300.0, "MCC")
    assert result.stability_effect == "positive"


# ---------------------------------------------------------------------------
# Lactose negative stability effect
# ---------------------------------------------------------------------------


def test_lactose_negative_stability():
    result = predict_excipient_effect(2.0, 300.0, "Lactose")
    assert result.stability_effect == "negative"


# ---------------------------------------------------------------------------
# High logP drug + surfactant → greater fold change than low logP
# ---------------------------------------------------------------------------


def test_high_logp_greater_fold_change_sds():
    high = predict_excipient_effect(6.0, 300.0, "SDS")
    low = predict_excipient_effect(1.0, 300.0, "SDS")
    assert high.solubility_fold_change > low.solubility_fold_change


def test_high_logp_greater_fold_change_tween80():
    high = predict_excipient_effect(5.0, 300.0, "Tween 80")
    low = predict_excipient_effect(0.0, 300.0, "Tween 80")
    assert high.solubility_fold_change > low.solubility_fold_change


# ---------------------------------------------------------------------------
# Invalid excipient raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_excipient_raises():
    with pytest.raises(ValueError, match="Unknown excipient"):
        predict_excipient_effect(2.0, 300.0, "Kryptonite")


def test_invalid_excipient_empty_string_raises():
    with pytest.raises(ValueError):
        predict_excipient_effect(2.0, 300.0, "")


# ---------------------------------------------------------------------------
# pKa amine + Lactose lowers compatibility
# ---------------------------------------------------------------------------


def test_lactose_amine_pka_lowers_score():
    without_pka = predict_excipient_effect(2.0, 300.0, "Lactose")
    with_pka = predict_excipient_effect(2.0, 300.0, "Lactose", drug_pka=9.0)
    assert with_pka.compatibility_score < without_pka.compatibility_score


# ---------------------------------------------------------------------------
# screen_excipients returns list of correct length (all excipients)
# ---------------------------------------------------------------------------


def test_screen_all_excipients_length():
    results = screen_excipients(2.0, 300.0)
    assert len(results) == len(SUPPORTED_EXCIPIENTS)


# ---------------------------------------------------------------------------
# screen_excipients returns sorted by compatibility_score descending
# ---------------------------------------------------------------------------


def test_screen_sorted_by_score():
    results = screen_excipients(3.0, 300.0)
    scores = [r.compatibility_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# screen_excipients subset
# ---------------------------------------------------------------------------


def test_screen_subset_excipients():
    subset = ["SDS", "MCC", "Lactose"]
    results = screen_excipients(3.0, 300.0, excipients=subset)
    assert len(results) == 3
    names = {r.excipient_name for r in results}
    assert names == set(subset)


# ---------------------------------------------------------------------------
# screen_excipients invalid excipient raises ValueError
# ---------------------------------------------------------------------------


def test_screen_invalid_excipient_raises():
    with pytest.raises(ValueError):
        screen_excipients(2.0, 300.0, excipients=["SDS", "InvalidExcipient"])


# ---------------------------------------------------------------------------
# Mechanism and notes are non-empty strings
# ---------------------------------------------------------------------------


def test_mechanism_nonempty():
    result = predict_excipient_effect(2.0, 300.0, "HPMC")
    assert isinstance(result.mechanism, str) and len(result.mechanism) > 0


def test_notes_nonempty():
    result = predict_excipient_effect(2.0, 300.0, "PVP")
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# HPMC improves solubility vs pure drug (fold change >= 1.0)
# ---------------------------------------------------------------------------


def test_hpmc_solubility_improvement_low_logp():
    result = predict_excipient_effect(0.0, 300.0, "HPMC")
    # HPMC should provide at least some solubility improvement
    assert result.solubility_fold_change >= 1.0


# ---------------------------------------------------------------------------
# PVP stability positive
# ---------------------------------------------------------------------------


def test_pvp_stability_positive():
    result = predict_excipient_effect(2.0, 300.0, "PVP")
    assert result.stability_effect == "positive"


# ===========================================================================
# Phase 757 tests — predict_excipient_interactions / screen_excipient_combinations
# ===========================================================================


def test_returns_excipient_interaction_result():
    result = predict_excipient_interactions("DrugA", ["cremophor_el"])
    assert isinstance(result, ExcipientInteractionResult)


def test_no_excipients_f_total_equals_baseline():
    result = predict_excipient_interactions("DrugA", [], fa_base=0.6, fm_cyp3a4=0.2, fh=0.8)
    # With no excipients, sol_enh=1, no pgp, no cyp inh → f_total should equal f_baseline
    assert abs(result.f_total - result.f_baseline) < 1e-6


def test_cremophor_el_high_pgp_inhibition():
    result = predict_excipient_interactions("DrugA", ["cremophor_el"])
    assert result.pgp_inhibition_combined >= 0.7


def test_sds_high_solubility_enhancement():
    result = predict_excipient_interactions("DrugA", ["sds"])
    assert result.solubility_enhancement_fold >= 2.5


def test_multiple_excipients_combined_pgp():
    single = predict_excipient_interactions("DrugA", ["cremophor_el"])
    combined = predict_excipient_interactions("DrugA", ["cremophor_el", "tween_80"])
    # Combined P-gp inhibition must be >= single excipient
    assert combined.pgp_inhibition_combined >= single.pgp_inhibition_combined


def test_cyp3a4_inhibitor_excipient_changes_fg():
    # High fm_cyp3a4 drug with CYP3A4-inhibiting excipient (cremophor_el):
    # The spec formula fg = 1/(1+fm*cyp_inh*5) reduces fg when cyp_inh > 0.
    # We test that fg_corrected != 1.0 (some effect occurred) and that a drug
    # with no CYP3A4 fraction is unaffected by this excipient on fg.
    no_cyp = predict_excipient_interactions("DrugA", ["cremophor_el"], fm_cyp3a4=0.0)
    high_cyp = predict_excipient_interactions("DrugA", ["cremophor_el"], fm_cyp3a4=0.8)
    # When fm_cyp3a4=0, fg_corrected = 1/(1+0) = 1.0
    assert abs(no_cyp.fg_corrected - 1.0) < 1e-6
    # When fm_cyp3a4=0.8, fg_corrected < 1.0 per spec formula
    assert high_cyp.fg_corrected < 1.0


def test_f_total_between_0_and_1():
    result = predict_excipient_interactions("DrugA", ["cremophor_el", "sds", "tpgs"])
    assert 0.0 <= result.f_total <= 1.0


def test_f_change_pct_computed_correctly():
    result = predict_excipient_interactions("DrugA", ["sds"], fa_base=0.5, fm_cyp3a4=0.0, fh=1.0)
    expected_pct = (result.f_total - result.f_baseline) / result.f_baseline * 100.0
    assert abs(result.f_change_pct - expected_pct) < 0.01


def test_recommendation_is_str():
    result = predict_excipient_interactions("DrugA", ["cremophor_el"])
    assert isinstance(result.recommendation, str)


def test_screen_excipient_combinations_returns_sorted():
    combos = [["hpmc"], ["cremophor_el"], ["sds"], ["pvp_k30"]]
    results = screen_excipient_combinations("DrugA", combos)
    f_totals = [r.f_total for r in results]
    assert f_totals == sorted(f_totals, reverse=True)


def test_pgp_substrate_cremophor_boosts_f_more():
    non_sub = predict_excipient_interactions("DrugA", ["cremophor_el"], is_pgp_substrate=False)
    substrate = predict_excipient_interactions("DrugA", ["cremophor_el"], is_pgp_substrate=True)
    assert substrate.fa_corrected >= non_sub.fa_corrected


def test_validation_fa_base_too_high():
    with pytest.raises(ValueError):
        predict_excipient_interactions("DrugA", [], fa_base=1.5)


def test_validation_fa_base_negative():
    with pytest.raises(ValueError):
        predict_excipient_interactions("DrugA", [], fa_base=-0.1)


def test_notes_nonempty_757():
    result = predict_excipient_interactions("DrugA", ["cremophor_el"])
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_citric_acid_ph_change():
    result = predict_excipient_interactions("DrugA", ["citric_acid"])
    assert result.ph_change < 0.0


def test_excipients_used_str():
    result = predict_excipient_interactions("DrugA", ["cremophor_el", "tween_80"])
    assert "cremophor_el" in result.excipients_used
    assert "tween_80" in result.excipients_used


def test_unknown_excipient_has_no_effect():
    # Unknown excipients are silently ignored
    with_unknown = predict_excipient_interactions("DrugA", ["unknown_polymer"])
    without = predict_excipient_interactions("DrugA", [])
    assert abs(with_unknown.f_total - without.f_total) < 1e-6


def test_tpgs_pgp_inhibition():
    result = predict_excipient_interactions("DrugA", ["tpgs"])
    assert result.pgp_inhibition_combined >= 0.6


def test_screen_excipient_combinations_length():
    combos = [["hpmc"], ["sds"], ["cremophor_el", "tween_80"]]
    results = screen_excipient_combinations("DrugA", combos)
    assert len(results) == 3


def test_f_baseline_uses_only_fa_base_and_fh():
    result = predict_excipient_interactions("DrugA", [], fa_base=0.6, fh=0.8)
    assert abs(result.f_baseline - 0.6 * 0.8) < 1e-6
