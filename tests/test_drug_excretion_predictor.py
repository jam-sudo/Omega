"""
Tests for Phase 448 — Drug Excretion Predictor (mass balance model).
"""

import pytest

from omega_pbpk.prediction.drug_excretion_predictor import (
    ExcretionPredictionResult,
    compare_excretion_profiles,
    predict_drug_excretion,
)

# ─── Fixture ────────────────────────────────────────────────────────────────


@pytest.fixture
def base_result():
    return predict_drug_excretion(
        drug_name="Metformin",
        dose_mg=500.0,
        fu_plasma=0.8,
        logp=-1.5,
        mw_Da=129.0,
        cl_hepatic_L_per_h=0.5,
        gfr_L_per_h=7.5,
        t_half_h=6.0,
    )


# ─── Return type ─────────────────────────────────────────────────────────────


def test_return_type(base_result):
    assert isinstance(base_result, ExcretionPredictionResult)


def test_drug_name_preserved(base_result):
    assert base_result.drug_name == "Metformin"


def test_dose_preserved(base_result):
    assert base_result.dose_mg == 500.0


# ─── Mass balance ─────────────────────────────────────────────────────────────


def test_fe_total_near_100(base_result):
    """Total accounted fraction should be close to 100% (within 1%)."""
    assert abs(base_result.fe_total_accounted_pct - 100.0) < 1.0


def test_all_fractions_between_0_and_100(base_result):
    assert 0.0 <= base_result.fe_urine_unchanged_pct <= 100.0
    assert 0.0 <= base_result.fe_urine_metabolites_pct <= 100.0
    assert 0.0 <= base_result.fe_bile_unchanged_pct <= 100.0
    assert 0.0 <= base_result.fe_bile_metabolites_pct <= 100.0
    assert 0.0 <= base_result.fe_expired_air_pct <= 100.0


# ─── logP effects ─────────────────────────────────────────────────────────────


def test_low_logp_higher_renal_excretion():
    """Low logP → less reabsorption → more unchanged renal excretion."""
    r_low = predict_drug_excretion(
        drug_name="HydrophilicDrug",
        dose_mg=100.0,
        fu_plasma=0.9,
        logp=-2.0,
        mw_Da=200.0,
        cl_hepatic_L_per_h=1.0,
    )
    r_high = predict_drug_excretion(
        drug_name="LipophilicDrug",
        dose_mg=100.0,
        fu_plasma=0.9,
        logp=3.0,
        mw_Da=200.0,
        cl_hepatic_L_per_h=1.0,
    )
    assert r_low.fe_urine_unchanged_pct > r_high.fe_urine_unchanged_pct


def test_high_logp_lower_renal_unchanged():
    """High logP → more reabsorption → less unchanged in urine."""
    r = predict_drug_excretion(
        drug_name="LipophilicDrug",
        dose_mg=100.0,
        fu_plasma=0.5,
        logp=4.0,
        mw_Da=300.0,
        cl_hepatic_L_per_h=5.0,
    )
    assert r.fe_urine_unchanged_pct < 30.0


# ─── MW effect on biliary ────────────────────────────────────────────────────


def test_high_mw_higher_biliary():
    """High MW (>500 Da) and high logP → significant biliary fraction."""
    r_high_mw = predict_drug_excretion(
        drug_name="LargeDrug",
        dose_mg=100.0,
        fu_plasma=0.3,
        logp=3.0,
        mw_Da=600.0,
        cl_hepatic_L_per_h=2.0,
    )
    r_low_mw = predict_drug_excretion(
        drug_name="SmallDrug",
        dose_mg=100.0,
        fu_plasma=0.3,
        logp=1.5,
        mw_Da=200.0,
        cl_hepatic_L_per_h=2.0,
    )
    fe_bile_high = r_high_mw.fe_bile_unchanged_pct + r_high_mw.fe_bile_metabolites_pct
    fe_bile_low = r_low_mw.fe_bile_unchanged_pct + r_low_mw.fe_bile_metabolites_pct
    assert fe_bile_high > fe_bile_low


# ─── Primary excretion route ─────────────────────────────────────────────────


def test_primary_route_renal_for_hydrophilic(base_result):
    """Low logP drug with good fu should be primarily renally excreted."""
    assert base_result.primary_excretion_route == "renal"


def test_primary_route_valid_strings():
    """primary_excretion_route must be one of the expected values."""
    valid_routes = {"renal", "biliary", "mixed", "metabolic_then_renal"}
    for logp in [-2.0, 0.0, 2.5, 4.0]:
        r = predict_drug_excretion(
            drug_name="Drug",
            dose_mg=100.0,
            fu_plasma=0.5,
            logp=logp,
            mw_Da=300.0,
            cl_hepatic_L_per_h=2.0,
        )
        assert r.primary_excretion_route in valid_routes


# ─── Timing metrics ──────────────────────────────────────────────────────────


def test_urinary_t50_positive(base_result):
    assert base_result.urinary_t50_h > 0.0


def test_urinary_t50_related_to_thalf(base_result):
    """urinary_t50 = 0.8 * t_half = 0.8 * 6.0 = 4.8 h."""
    assert abs(base_result.urinary_t50_h - 4.8) < 0.01


def test_cumulative_urine_at_24h_positive(base_result):
    assert base_result.cumulative_urine_mg_at_24h > 0.0


def test_cumulative_urine_at_24h_less_than_dose(base_result):
    assert base_result.cumulative_urine_mg_at_24h < base_result.dose_mg


# ─── Sensitivity index ───────────────────────────────────────────────────────


def test_renal_sensitivity_index_range(base_result):
    assert 0.0 <= base_result.renal_sensitivity_index <= 1.0


# ─── Biliary recycling potential ─────────────────────────────────────────────


def test_biliary_recycling_valid_values():
    valid = {"high", "moderate", "low"}
    for mw in [200.0, 450.0, 600.0]:
        r = predict_drug_excretion(
            drug_name="Drug",
            dose_mg=100.0,
            fu_plasma=0.5,
            logp=2.5,
            mw_Da=mw,
            cl_hepatic_L_per_h=2.0,
        )
        assert r.biliary_recycling_potential in valid


def test_high_mw_logp_gives_high_biliary_recycling():
    """MW > 500 and logP > 2 → high biliary fraction → high recycling potential."""
    r = predict_drug_excretion(
        drug_name="Drug",
        dose_mg=100.0,
        fu_plasma=0.5,
        logp=3.0,
        mw_Da=600.0,
        cl_hepatic_L_per_h=0.5,
    )
    assert r.biliary_recycling_potential in {"high", "moderate"}


# ─── compare_excretion_profiles ──────────────────────────────────────────────


def test_compare_excretion_profiles_length():
    compounds = [
        {
            "drug_name": "A",
            "dose_mg": 100.0,
            "fu_plasma": 0.9,
            "logp": -1.0,
            "mw_Da": 200.0,
            "cl_hepatic_L_per_h": 0.5,
        },
        {
            "drug_name": "B",
            "dose_mg": 100.0,
            "fu_plasma": 0.3,
            "logp": 3.0,
            "mw_Da": 400.0,
            "cl_hepatic_L_per_h": 3.0,
        },
        {
            "drug_name": "C",
            "dose_mg": 100.0,
            "fu_plasma": 0.6,
            "logp": 1.5,
            "mw_Da": 300.0,
            "cl_hepatic_L_per_h": 2.0,
        },
    ]
    results = compare_excretion_profiles(compounds)
    assert len(results) == 3


def test_compare_excretion_profiles_sorted_descending():
    compounds = [
        {
            "drug_name": "A",
            "dose_mg": 100.0,
            "fu_plasma": 0.9,
            "logp": -1.0,
            "mw_Da": 200.0,
            "cl_hepatic_L_per_h": 0.5,
        },
        {
            "drug_name": "B",
            "dose_mg": 100.0,
            "fu_plasma": 0.3,
            "logp": 3.0,
            "mw_Da": 400.0,
            "cl_hepatic_L_per_h": 3.0,
        },
        {
            "drug_name": "C",
            "dose_mg": 100.0,
            "fu_plasma": 0.6,
            "logp": 1.5,
            "mw_Da": 300.0,
            "cl_hepatic_L_per_h": 2.0,
        },
    ]
    results = compare_excretion_profiles(compounds)
    for i in range(len(results) - 1):
        assert results[i].fe_urine_unchanged_pct >= results[i + 1].fe_urine_unchanged_pct


# ─── Validation errors ───────────────────────────────────────────────────────


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_drug_excretion(
            drug_name="D",
            dose_mg=0.0,
            fu_plasma=0.5,
            logp=1.0,
            mw_Da=300.0,
            cl_hepatic_L_per_h=2.0,
        )


def test_validation_fu_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_drug_excretion(
            drug_name="D",
            dose_mg=100.0,
            fu_plasma=0.0,
            logp=1.0,
            mw_Da=300.0,
            cl_hepatic_L_per_h=2.0,
        )


def test_validation_fu_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_drug_excretion(
            drug_name="D",
            dose_mg=100.0,
            fu_plasma=1.5,
            logp=1.0,
            mw_Da=300.0,
            cl_hepatic_L_per_h=2.0,
        )


def test_notes_not_empty(base_result):
    assert len(base_result.notes) > 0
