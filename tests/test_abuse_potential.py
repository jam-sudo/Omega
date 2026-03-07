"""Tests for Phase 1103: Drug Abuse Potential Assessment."""

import pytest

from omega_pbpk.risk.abuse_potential import (
    AbuseResult,
    assess_abuse_potential,
    compare_formulations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG_KWARGS_BASE = dict(
    drug_name="TestDrug",
    logP=2.5,
    mw_Da=300.0,
    tmax_h=1.0,
    t_half_h=4.0,
)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------

def test_return_type():
    result = assess_abuse_potential(**DRUG_KWARGS_BASE)
    assert isinstance(result, AbuseResult)


def test_abuse_score_range():
    result = assess_abuse_potential(**DRUG_KWARGS_BASE)
    assert 0.0 <= result.abuse_score <= 100.0


def test_abuse_score_range_high_logp():
    result = assess_abuse_potential(
        drug_name="HighLogP", logP=5.0, mw_Da=200.0, tmax_h=0.5, t_half_h=1.0
    )
    assert 0.0 <= result.abuse_score <= 100.0


def test_result_fields_present():
    result = assess_abuse_potential(**DRUG_KWARGS_BASE)
    for field in [
        "drug_name", "logP", "mw_Da", "tmax_h", "t_half_h", "formulation",
        "kp_brain", "abuse_score", "abuse_category", "cns_penetration_class",
        "physical_dependence_risk", "scheduling_recommendation", "notes",
    ]:
        assert hasattr(result, field)


# ---------------------------------------------------------------------------
# Formulation comparison
# ---------------------------------------------------------------------------

def test_ir_higher_than_er():
    ir = assess_abuse_potential(**DRUG_KWARGS_BASE, formulation="IR")
    er = assess_abuse_potential(**DRUG_KWARGS_BASE, formulation="ER")
    assert ir.abuse_score > er.abuse_score


def test_ir_higher_than_abuse_deterrent():
    ir = assess_abuse_potential(**DRUG_KWARGS_BASE, formulation="IR")
    ad = assess_abuse_potential(**DRUG_KWARGS_BASE, formulation="abuse_deterrent")
    assert ir.abuse_score > ad.abuse_score


def test_abuse_deterrent_lowest():
    ir = assess_abuse_potential(**DRUG_KWARGS_BASE, formulation="IR")
    er = assess_abuse_potential(**DRUG_KWARGS_BASE, formulation="ER")
    ad = assess_abuse_potential(**DRUG_KWARGS_BASE, formulation="abuse_deterrent")
    assert ad.abuse_score <= er.abuse_score <= ir.abuse_score


# ---------------------------------------------------------------------------
# Effect of PK parameters on score
# ---------------------------------------------------------------------------

def test_high_logp_higher_score():
    low = assess_abuse_potential(drug_name="D", logP=0.5, mw_Da=300.0)
    high = assess_abuse_potential(drug_name="D", logP=4.5, mw_Da=300.0)
    assert high.abuse_score > low.abuse_score


def test_short_tmax_higher_score():
    slow = assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=300.0, tmax_h=4.0)
    fast = assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=300.0, tmax_h=0.25)
    assert fast.abuse_score > slow.abuse_score


def test_short_thalf_higher_dependence():
    short = assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=300.0, t_half_h=0.5)
    assert short.physical_dependence_risk == "high"


def test_long_thalf_low_dependence():
    long_ = assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=300.0, t_half_h=12.0)
    assert long_.physical_dependence_risk == "low"


def test_moderate_thalf_moderate_dependence():
    mid = assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=300.0, t_half_h=4.0)
    assert mid.physical_dependence_risk == "moderate"


# ---------------------------------------------------------------------------
# Categorical fields
# ---------------------------------------------------------------------------

def test_abuse_category_valid_values():
    valid = {"low", "moderate", "high", "very_high"}
    result = assess_abuse_potential(**DRUG_KWARGS_BASE)
    assert result.abuse_category in valid


def test_abuse_category_low_for_low_score():
    # abuse_deterrent + high MW + low logP + slow onset + long t½ → low score
    result = assess_abuse_potential(
        drug_name="D", logP=0.1, mw_Da=500.0, tmax_h=6.0, t_half_h=24.0,
        formulation="abuse_deterrent"
    )
    assert result.abuse_category == "low"
    assert result.abuse_score < 25


def test_abuse_category_very_high_for_high_score():
    result = assess_abuse_potential(
        drug_name="D", logP=5.0, mw_Da=150.0, tmax_h=0.1, t_half_h=0.5,
        formulation="IR"
    )
    assert result.abuse_category in {"very_high", "high"}


def test_cns_penetration_class_valid():
    valid = {"poor", "moderate", "good"}
    result = assess_abuse_potential(**DRUG_KWARGS_BASE)
    assert result.cns_penetration_class in valid


def test_cns_penetration_class_poor_for_high_mw_low_logp():
    result = assess_abuse_potential(drug_name="D", logP=-1.0, mw_Da=600.0)
    assert result.cns_penetration_class == "poor"


def test_scheduling_recommendation_valid():
    valid = {"unscheduled", "schedule_IV", "schedule_II", "schedule_I"}
    result = assess_abuse_potential(**DRUG_KWARGS_BASE)
    assert result.scheduling_recommendation in valid


def test_high_score_high_scheduling():
    result = assess_abuse_potential(
        drug_name="D", logP=5.0, mw_Da=150.0, tmax_h=0.1, t_half_h=0.5,
        formulation="IR"
    )
    assert result.scheduling_recommendation in {"schedule_I", "schedule_II"}


def test_low_score_unscheduled():
    result = assess_abuse_potential(
        drug_name="D", logP=0.1, mw_Da=500.0, tmax_h=6.0, t_half_h=24.0,
        formulation="abuse_deterrent"
    )
    assert result.scheduling_recommendation == "unscheduled"


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------

def test_compare_formulations_returns_three():
    results = compare_formulations("Drug", logP=2.5, mw_Da=300.0)
    assert len(results) == 3


def test_compare_formulations_all_are_abuse_result():
    results = compare_formulations("Drug", logP=2.5, mw_Da=300.0)
    for r in results:
        assert isinstance(r, AbuseResult)


def test_compare_formulations_sorted_ascending():
    results = compare_formulations("Drug", logP=2.5, mw_Da=300.0)
    scores = [r.abuse_score for r in results]
    assert scores == sorted(scores)


def test_compare_formulations_safest_first_is_abuse_deterrent():
    results = compare_formulations("Drug", logP=2.5, mw_Da=300.0)
    assert results[0].formulation == "abuse_deterrent"


def test_compare_formulations_riskiest_last_is_ir():
    results = compare_formulations("Drug", logP=2.5, mw_Da=300.0)
    assert results[-1].formulation == "IR"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_formulation_raises():
    with pytest.raises(ValueError, match="formulation"):
        assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=300.0, formulation="patch")


def test_logp_out_of_range_raises_low():
    with pytest.raises(ValueError, match="logP"):
        assess_abuse_potential(drug_name="D", logP=-10.0, mw_Da=300.0)


def test_logp_out_of_range_raises_high():
    with pytest.raises(ValueError, match="logP"):
        assess_abuse_potential(drug_name="D", logP=15.0, mw_Da=300.0)


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=0.0)


def test_negative_tmax_raises():
    with pytest.raises(ValueError, match="tmax_h"):
        assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=300.0, tmax_h=-1.0)


def test_zero_thalf_raises():
    with pytest.raises(ValueError, match="t_half_h"):
        assess_abuse_potential(drug_name="D", logP=2.0, mw_Da=300.0, t_half_h=0.0)


# ---------------------------------------------------------------------------
# kp_brain sanity
# ---------------------------------------------------------------------------

def test_kp_brain_positive():
    result = assess_abuse_potential(**DRUG_KWARGS_BASE)
    assert result.kp_brain > 0.0


def test_kp_brain_clamped_to_max():
    # Very lipophilic, very small MW
    result = assess_abuse_potential(drug_name="D", logP=10.0, mw_Da=50.0)
    assert result.kp_brain <= 2.0


def test_kp_brain_clamped_to_min():
    result = assess_abuse_potential(drug_name="D", logP=-5.0, mw_Da=600.0)
    assert result.kp_brain >= 0.01
