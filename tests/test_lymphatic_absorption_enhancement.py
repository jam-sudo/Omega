"""Tests for Phase 1035 — Lymphatic Absorption Enhancement."""

import pytest

from omega_pbpk.prediction.lymphatic_absorption_enhancement import (
    LymphaticEnhancementResult,
    compare_strategies,
    predict_lymphatic_enhancement,
)

# ---------- Helper -----------------------------------------------------------


def make_result(
    drug_name="TestDrug",
    logp=3.0,
    mw_Da=400.0,
    strategy="lipid_formulation",
    long_chain_triglyceride_pct=0.0,
    f_hepatic=0.7,
    **kwargs,
):
    return predict_lymphatic_enhancement(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        strategy=strategy,
        long_chain_triglyceride_pct=long_chain_triglyceride_pct,
        f_hepatic=f_hepatic,
        **kwargs,
    )


# ---------- Return-type tests ------------------------------------------------


def test_returns_frozen_dataclass():
    r = make_result()
    assert isinstance(r, LymphaticEnhancementResult)


def test_frozen_immutable():
    r = make_result()
    with pytest.raises((AttributeError, TypeError)):
        r.drug_name = "other"  # type: ignore[misc]


# ---------- Range tests ------------------------------------------------------


def test_f_lymph_baseline_in_range():
    for logp in [-2.0, 0.0, 3.0, 5.0, 8.0]:
        r = make_result(logp=logp)
        assert 0.0 <= r.f_lymph_baseline <= 0.5, f"Out of range at logP={logp}"


def test_f_lymph_enhanced_in_range():
    for strategy in ["lipid_formulation", "self_emulsifying", "nanoparticle", "prodrug", "none"]:
        r = make_result(strategy=strategy, logp=5.0)
        assert 0.0 <= r.f_lymph_enhanced <= 0.80, f"Out of range for strategy={strategy}"


def test_enhancement_ratio_gte_one():
    for strategy in ["lipid_formulation", "self_emulsifying", "nanoparticle", "prodrug", "none"]:
        r = make_result(strategy=strategy)
        assert r.enhancement_ratio >= 1.0


def test_portal_bypass_in_range():
    r = make_result()
    assert 0.0 <= r.portal_bypass_fraction <= 1.0


def test_bioavailability_gain_nonnegative():
    r = make_result()
    assert r.bioavailability_gain >= 0.0


# ---------- Biological logic -------------------------------------------------


def test_high_logp_higher_baseline_than_low():
    r_high = make_result(logp=5.0)
    r_low = make_result(logp=0.0)
    assert r_high.f_lymph_baseline > r_low.f_lymph_baseline


def test_self_emulsifying_higher_than_nanoparticle():
    r_se = make_result(strategy="self_emulsifying", logp=4.0)
    r_np = make_result(strategy="nanoparticle", logp=4.0)
    assert r_se.f_lymph_enhanced >= r_np.f_lymph_enhanced


def test_lct_increases_lipid_formulation():
    r_no_lct = make_result(strategy="lipid_formulation", long_chain_triglyceride_pct=0.0, logp=4.0)
    r_lct = make_result(strategy="lipid_formulation", long_chain_triglyceride_pct=50.0, logp=4.0)
    assert r_lct.f_lymph_enhanced >= r_no_lct.f_lymph_enhanced


def test_recommended_flag_matches_ratio():
    for strategy in ["lipid_formulation", "self_emulsifying", "nanoparticle", "prodrug", "none"]:
        r = make_result(strategy=strategy)
        assert r.recommended == (r.enhancement_ratio > 2.0)


def test_first_pass_avoidance_matches_threshold():
    r = make_result(logp=6.0, strategy="self_emulsifying")
    assert r.first_pass_avoidance == (r.portal_bypass_fraction > 0.3)


def test_none_strategy_ratio_equals_one():
    r = make_result(strategy="none")
    assert r.enhancement_ratio == pytest.approx(1.0, abs=1e-9)


# ---------- compare_strategies -----------------------------------------------


def test_compare_strategies_returns_five():
    results = compare_strategies("Drug", logp=4.0, mw_Da=400.0)
    assert len(results) == 5


def test_compare_strategies_sorted_desc():
    results = compare_strategies("Drug", logp=4.0, mw_Da=400.0)
    for i in range(len(results) - 1):
        assert results[i].f_lymph_enhanced >= results[i + 1].f_lymph_enhanced


def test_compare_strategies_all_strategies_present():
    results = compare_strategies("Drug", logp=3.0, mw_Da=350.0)
    strategies = {r.strategy for r in results}
    expected = {"lipid_formulation", "self_emulsifying", "nanoparticle", "prodrug", "none"}
    assert strategies == expected


# ---------- notes field ------------------------------------------------------


def test_notes_nonempty():
    r = make_result()
    assert len(r.notes) > 0


# ---------- Validation tests -------------------------------------------------


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_lymphatic_enhancement("Drug", logp=3.0, mw_Da=0.0)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError):
        predict_lymphatic_enhancement("Drug", logp=3.0, mw_Da=-100.0)


def test_validation_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        predict_lymphatic_enhancement("Drug", logp=3.0, mw_Da=400.0, strategy="magic")


def test_validation_f_hepatic_negative_raises():
    with pytest.raises(ValueError, match="f_hepatic"):
        predict_lymphatic_enhancement("Drug", logp=3.0, mw_Da=400.0, f_hepatic=-0.1)


def test_validation_lct_over_100_raises():
    with pytest.raises(ValueError, match="long_chain_triglyceride_pct"):
        predict_lymphatic_enhancement(
            "Drug", logp=3.0, mw_Da=400.0, long_chain_triglyceride_pct=101.0
        )


def test_validation_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_lymphatic_enhancement("Drug", logp=3.0, mw_Da=400.0, ionization_type="charged")
