"""Tests for Phase 287 neonatal PK scaling additions."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.neonatal_pk import (
    NeonatalDoseResult287,
    NeonatalPKResult,
    NeonatalScaling,
    neonatal_scaling_factors,
    scale_neonatal_dose,
    simulate_neonatal_pk,
)

# ---------------------------------------------------------------------------
# neonatal_scaling_factors tests
# ---------------------------------------------------------------------------


def test_scaling_factors_returns_dataclass():
    sf = neonatal_scaling_factors(gestational_age_weeks=40.0, postnatal_age_days=0.0)
    assert isinstance(sf, NeonatalScaling)


def test_scaling_factors_frozen():
    sf = neonatal_scaling_factors(gestational_age_weeks=40.0, postnatal_age_days=7.0)
    with pytest.raises((AttributeError, TypeError)):
        sf.cyp3a4_factor = 0.99  # type: ignore[misc]


def test_cyp3a4_at_day_0_near_lower():
    # At pna=0: 0.1 + 0.9 * sigmoid((0-7)/30) = 0.1 + 0.9*sigmoid(-0.233...)
    sf = neonatal_scaling_factors(40.0, 0.0)
    assert 0.1 <= sf.cyp3a4_factor < 0.5


def test_cyp2d6_at_day_0_near_lower():
    sf = neonatal_scaling_factors(40.0, 0.0)
    # 0.05 + 0.95 * sigmoid((0-14)/45) ≈ 0.05 + 0.95 * sigmoid(-0.311)
    assert 0.05 <= sf.cyp2d6_factor < 0.5


def test_cyp3a4_increases_with_age():
    sf0 = neonatal_scaling_factors(40.0, 0.0)
    sf30 = neonatal_scaling_factors(40.0, 30.0)
    sf180 = neonatal_scaling_factors(40.0, 180.0)
    assert sf30.cyp3a4_factor > sf0.cyp3a4_factor
    assert sf180.cyp3a4_factor > sf30.cyp3a4_factor


def test_gfr_factor_at_day_0_near_lower():
    sf = neonatal_scaling_factors(40.0, 0.0)
    # At pna=0: 0.1 + 0.9 * sigmoid((0-3)/10) ≈ 0.1 + 0.9 * sigmoid(-0.3)
    assert 0.1 <= sf.gfr_factor < 0.5


def test_gfr_factor_increases_with_age():
    sf0 = neonatal_scaling_factors(40.0, 0.0)
    sf14 = neonatal_scaling_factors(40.0, 14.0)
    assert sf14.gfr_factor > sf0.gfr_factor


def test_protein_binding_factor_range():
    sf = neonatal_scaling_factors(40.0, 0.0)
    assert 0.7 <= sf.protein_binding_factor <= 1.0


def test_protein_binding_increases_with_age():
    sf0 = neonatal_scaling_factors(40.0, 0.0)
    sf90 = neonatal_scaling_factors(40.0, 90.0)
    assert sf90.protein_binding_factor > sf0.protein_binding_factor


def test_body_water_higher_at_birth():
    sf0 = neonatal_scaling_factors(40.0, 0.0)
    sf180 = neonatal_scaling_factors(40.0, 180.0)
    assert sf0.body_water_factor > sf180.body_water_factor
    assert sf0.body_water_factor > 1.0  # neonates have higher TBW than adult


def test_body_water_asymptotes_to_1():
    sf = neonatal_scaling_factors(40.0, 365.0)
    # exp(-365/30) ≈ 0 → body_water ≈ 1.0
    assert abs(sf.body_water_factor - 1.0) < 0.01


def test_maturation_stage_term_neonate():
    sf = neonatal_scaling_factors(40.0, 0.0)
    assert sf.maturation_stage == "term_neonate"


def test_maturation_stage_young_infant():
    sf = neonatal_scaling_factors(40.0, 90.0)
    assert sf.maturation_stage == "young_infant"


def test_maturation_stage_infant():
    sf = neonatal_scaling_factors(40.0, 200.0)
    assert sf.maturation_stage == "infant"


def test_maturation_stage_preterm():
    # GA=35, PNA=0 → PMA=35 weeks → preterm
    sf = neonatal_scaling_factors(35.0, 0.0)
    assert sf.maturation_stage == "preterm"


def test_ga_out_of_range_raises():
    with pytest.raises(ValueError, match="gestational_age_weeks"):
        neonatal_scaling_factors(gestational_age_weeks=23.0, postnatal_age_days=0.0)


def test_ga_too_high_raises():
    with pytest.raises(ValueError, match="gestational_age_weeks"):
        neonatal_scaling_factors(gestational_age_weeks=43.0, postnatal_age_days=0.0)


def test_pna_negative_raises():
    with pytest.raises(ValueError, match="postnatal_age_days"):
        neonatal_scaling_factors(gestational_age_weeks=40.0, postnatal_age_days=-1.0)


def test_pna_too_high_raises():
    with pytest.raises(ValueError, match="postnatal_age_days"):
        neonatal_scaling_factors(gestational_age_weeks=40.0, postnatal_age_days=366.0)


# ---------------------------------------------------------------------------
# scale_neonatal_dose tests
# ---------------------------------------------------------------------------


def test_scale_neonatal_dose_returns_result():
    res = scale_neonatal_dose(
        drug_name="TestDrug",
        adult_dose_mg_per_kg=10.0,
        weight_kg=3.0,
        gestational_age_weeks=40.0,
        postnatal_age_days=7.0,
    )
    assert isinstance(res, NeonatalDoseResult287)


def test_scale_dose_frozen():
    res = scale_neonatal_dose("D", 10.0, 3.0, 40.0, 7.0)
    with pytest.raises((AttributeError, TypeError)):
        res.recommended_dose_mg = 999.0  # type: ignore[misc]


def test_scale_dose_positive():
    res = scale_neonatal_dose("D", 10.0, 3.0, 40.0, 7.0)
    assert res.recommended_dose_mg > 0


def test_scale_dose_less_than_adult_for_newborn():
    # At PNA=7, maturation factors are all < 1, so dose should be < adult dose*weight
    res = scale_neonatal_dose("D", 10.0, 3.0, 40.0, 7.0)
    adult_dose_mg = 10.0 * 3.0
    assert res.recommended_dose_mg < adult_dose_mg


def test_scale_dose_invalid_weight_raises():
    with pytest.raises(ValueError, match="weight_kg"):
        scale_neonatal_dose("D", 10.0, 0.0, 40.0, 7.0)


def test_scale_dose_invalid_dose_raises():
    with pytest.raises(ValueError, match="adult_dose_mg_per_kg"):
        scale_neonatal_dose("D", 0.0, 3.0, 40.0, 7.0)


def test_scale_dose_negative_fm_raises():
    with pytest.raises(ValueError, match="fm"):
        scale_neonatal_dose("D", 10.0, 3.0, 40.0, 7.0, fm_cyp3a4=-0.1)


def test_scale_dose_fm_sum_exceeds_1_raises():
    with pytest.raises(ValueError, match="Sum"):
        scale_neonatal_dose(
            "D",
            10.0,
            3.0,
            40.0,
            7.0,
            fm_cyp3a4=0.5,
            fm_cyp2d6=0.3,
            fm_cyp2c9=0.3,
            fm_cyp2c19=0.1,
            f_renal=0.1,
        )


def test_scale_dose_stores_scaling():
    res = scale_neonatal_dose("D", 10.0, 3.0, 40.0, 7.0)
    assert isinstance(res.scaling, NeonatalScaling)


# ---------------------------------------------------------------------------
# simulate_neonatal_pk tests
# ---------------------------------------------------------------------------


def test_simulate_neonatal_pk_iv_returns_result():
    res = simulate_neonatal_pk(
        drug_name="Drug_IV",
        dose_mg=10.0,
        weight_kg=3.0,
        gestational_age_weeks=40.0,
        postnatal_age_days=7.0,
        cl_adult_per_kg_L_per_h=0.1,
        vd_adult_per_kg_L=1.0,
        route="iv",
    )
    assert isinstance(res, NeonatalPKResult)


def test_simulate_neonatal_pk_oral_returns_result():
    res = simulate_neonatal_pk(
        drug_name="Drug_Oral",
        dose_mg=10.0,
        weight_kg=3.0,
        gestational_age_weeks=40.0,
        postnatal_age_days=7.0,
        cl_adult_per_kg_L_per_h=0.1,
        vd_adult_per_kg_L=1.0,
        route="oral",
        ka_per_h=1.0,
    )
    assert isinstance(res, NeonatalPKResult)


def test_simulate_pk_frozen():
    res = simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    with pytest.raises((AttributeError, TypeError)):
        res.cmax = 999.0  # type: ignore[misc]


def test_simulate_pk_cmax_positive():
    res = simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    assert res.cmax > 0


def test_simulate_pk_auc_positive():
    res = simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    assert res.auc > 0


def test_simulate_pk_t_half_consistent():
    res = simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    # t_half should be positive and finite
    assert res.t_half_h > 0
    assert math.isfinite(res.t_half_h)


def test_simulate_pk_time_array_length():
    res = simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0, t_end_h=24.0, dt_h=0.1)
    assert len(res.times_h) == len(res.c_plasma)
    assert len(res.times_h) > 0


def test_simulate_pk_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0, route="im")


def test_simulate_pk_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_neonatal_pk("D", 0.0, 3.0, 40.0, 7.0, 0.1, 1.0)


def test_simulate_pk_cl_neonatal_positive():
    res = simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    assert res.cl_neonatal > 0


def test_simulate_pk_vd_neonatal_positive():
    res = simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    assert res.vd_neonatal > 0


def test_simulate_pk_iv_cmax_at_t0():
    # For IV bolus, Cmax should occur at t=0
    res = simulate_neonatal_pk(
        "D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0, route="iv", t_end_h=24.0, dt_h=0.5
    )
    assert res.tmax_h == pytest.approx(0.0, abs=0.6)


def test_simulate_pk_oral_tmax_positive():
    res = simulate_neonatal_pk(
        "D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0, route="oral", ka_per_h=1.0, t_end_h=24.0
    )
    assert res.tmax_h > 0


def test_simulate_pk_higher_dose_higher_cmax():
    res_low = simulate_neonatal_pk("D", 5.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    res_high = simulate_neonatal_pk("D", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    assert res_high.cmax > res_low.cmax


def test_simulate_pk_stores_drug_name():
    res = simulate_neonatal_pk("Vancomycin", 10.0, 3.0, 40.0, 7.0, 0.1, 1.0)
    assert res.drug_name == "Vancomycin"


def test_simulate_pk_fm_sum_exceeds_1_raises():
    with pytest.raises(ValueError, match="Sum"):
        simulate_neonatal_pk(
            "D",
            10.0,
            3.0,
            40.0,
            7.0,
            0.1,
            1.0,
            fm_cyp3a4=0.5,
            fm_cyp2d6=0.3,
            fm_cyp2c9=0.3,
            fm_cyp2c19=0.1,
            f_renal=0.1,
        )
