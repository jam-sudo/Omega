"""Tests for neonatal_pk module — Phase 197 and Phase 744."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.neonatal_pk import (
    NeonatalDoseResult,
    NeonatalPKResult744,
    NeonatalScalingFactors,
    adjust_dose_neonatal,
    compute_maturation_factors,
    neonatal_scaling,
    simulate_neonatal_pk_744,
)

# ---------------------------------------------------------------------------
# neonatal_scaling tests
# ---------------------------------------------------------------------------


def test_scaling_returns_dataclass():
    sf = neonatal_scaling(age_days=1.0, weight_kg=3.5)
    assert isinstance(sf, NeonatalScalingFactors)


def test_day_0_cyp3a4_near_0_1():
    sf = neonatal_scaling(age_days=0.0, weight_kg=3.0)
    # At day 0: cyp3a4 = 0.1 + 0.9 * (0 / (0 + 30)) = 0.1
    assert pytest.approx(sf.cyp3a4_maturation, abs=1e-6) == 0.1


def test_day_0_hepatic_near_0_1():
    sf = neonatal_scaling(age_days=0.0, weight_kg=3.0)
    assert pytest.approx(sf.hepatic_enzyme_maturation, abs=1e-6) == 0.1


def test_day_28_more_mature_than_day_0():
    sf0 = neonatal_scaling(age_days=0.0, weight_kg=3.0)
    sf28 = neonatal_scaling(age_days=28.0, weight_kg=3.5)
    assert sf28.cyp3a4_maturation > sf0.cyp3a4_maturation
    assert sf28.hepatic_enzyme_maturation > sf0.hepatic_enzyme_maturation
    assert sf28.renal_maturation > sf0.renal_maturation


def test_gfr_increases_with_age():
    sf0 = neonatal_scaling(age_days=0.0, weight_kg=3.0)
    sf14 = neonatal_scaling(age_days=14.0, weight_kg=3.5)
    sf28 = neonatal_scaling(age_days=28.0, weight_kg=3.8)
    # GFR/kg can change with both age and weight, but renal_maturation increases
    assert sf14.renal_maturation > sf0.renal_maturation
    assert sf28.renal_maturation > sf14.renal_maturation


def test_tbw_decreases_with_age():
    sf0 = neonatal_scaling(age_days=0.0, weight_kg=3.0)
    sf28 = neonatal_scaling(age_days=28.0, weight_kg=3.5)
    assert sf0.total_body_water_fraction > sf28.total_body_water_fraction


def test_tbw_minimum_0_6():
    # At day 100, tbw would be 0.80 - 0.002*100 = 0.60
    sf = neonatal_scaling(age_days=200.0, weight_kg=5.0)
    assert sf.total_body_water_fraction >= 0.6


def test_bsa_positive():
    sf = neonatal_scaling(age_days=7.0, weight_kg=3.2)
    assert sf.bsa_m2 > 0


def test_plasma_protein_capped_at_1():
    sf = neonatal_scaling(age_days=200.0, weight_kg=7.0)
    assert sf.plasma_protein_fraction <= 1.0


def test_negative_age_raises():
    with pytest.raises(ValueError, match="age_days"):
        neonatal_scaling(age_days=-1.0, weight_kg=3.0)


def test_zero_weight_raises():
    with pytest.raises(ValueError, match="weight_kg"):
        neonatal_scaling(age_days=5.0, weight_kg=0.0)


# ---------------------------------------------------------------------------
# adjust_dose_neonatal tests
# ---------------------------------------------------------------------------


def test_adjust_dose_returns_result():
    res = adjust_dose_neonatal(
        drug_name="Drug_A",
        adult_dose_mg_per_kg=10.0,
        cl_adult_mL_per_min_per_kg=5.0,
        vd_adult_L_per_kg=1.0,
        age_days=7.0,
        weight_kg=3.0,
        elimination="hepatic",
    )
    assert isinstance(res, NeonatalDoseResult)


def test_neonatal_dose_less_than_adult_hepatic():
    res = adjust_dose_neonatal(
        drug_name="Drug_A",
        adult_dose_mg_per_kg=10.0,
        cl_adult_mL_per_min_per_kg=5.0,
        vd_adult_L_per_kg=1.0,
        age_days=7.0,
        weight_kg=3.0,
        elimination="hepatic",
    )
    assert res.neonatal_dose_mg_per_kg < res.adult_dose_mg_per_kg


def test_neonatal_dose_less_than_adult_renal():
    res = adjust_dose_neonatal(
        drug_name="Drug_B",
        adult_dose_mg_per_kg=5.0,
        cl_adult_mL_per_min_per_kg=3.0,
        vd_adult_L_per_kg=0.5,
        age_days=1.0,
        weight_kg=3.0,
        elimination="renal",
    )
    assert res.neonatal_dose_mg_per_kg < res.adult_dose_mg_per_kg


def test_mixed_elimination():
    res = adjust_dose_neonatal(
        drug_name="Drug_C",
        adult_dose_mg_per_kg=8.0,
        cl_adult_mL_per_min_per_kg=4.0,
        vd_adult_L_per_kg=1.2,
        age_days=10.0,
        weight_kg=3.5,
        elimination="mixed",
    )
    assert res.neonatal_dose_mg_per_kg > 0


def test_absolute_dose_consistent():
    res = adjust_dose_neonatal(
        drug_name="Drug_D",
        adult_dose_mg_per_kg=10.0,
        cl_adult_mL_per_min_per_kg=5.0,
        vd_adult_L_per_kg=1.0,
        age_days=7.0,
        weight_kg=3.0,
    )
    assert (
        pytest.approx(res.neonatal_dose_mg, rel=1e-6) == res.neonatal_dose_mg_per_kg * res.weight_kg
    )


def test_caution_cyp3a4_immature():
    res = adjust_dose_neonatal(
        drug_name="Drug_E",
        adult_dose_mg_per_kg=10.0,
        cl_adult_mL_per_min_per_kg=5.0,
        vd_adult_L_per_kg=1.0,
        age_days=0.0,  # day 0 → CYP3A4 = 0.1 < 0.3
        weight_kg=3.0,
        elimination="hepatic",
    )
    assert any("CYP3A4" in c for c in res.cautions)


def test_caution_renal_immature():
    res = adjust_dose_neonatal(
        drug_name="Drug_F",
        adult_dose_mg_per_kg=5.0,
        cl_adult_mL_per_min_per_kg=3.0,
        vd_adult_L_per_kg=0.5,
        age_days=0.0,  # very immature renal
        weight_kg=3.0,
        elimination="renal",
    )
    # renal_maturation at day 0 = gfr/120 ≈ 1/120 << 0.2
    assert any("Renal" in c for c in res.cautions)


def test_invalid_elimination_raises():
    with pytest.raises(ValueError, match="elimination"):
        adjust_dose_neonatal(
            drug_name="Drug_G",
            adult_dose_mg_per_kg=10.0,
            cl_adult_mL_per_min_per_kg=5.0,
            vd_adult_L_per_kg=1.0,
            age_days=7.0,
            weight_kg=3.0,
            elimination="unknown",
        )


def test_neonatal_t_half_longer_than_adult():
    # Very immature → very low CL → much longer t_half
    res = adjust_dose_neonatal(
        drug_name="Drug_H",
        adult_dose_mg_per_kg=10.0,
        cl_adult_mL_per_min_per_kg=5.0,
        vd_adult_L_per_kg=1.0,
        age_days=0.0,
        weight_kg=3.0,
        elimination="hepatic",
        t_half_adult_h=6.0,
    )
    assert res.t_half_neonatal_h > res.t_half_adult_h


def test_dose_interval_long_for_long_thalf():
    res = adjust_dose_neonatal(
        drug_name="Drug_I",
        adult_dose_mg_per_kg=10.0,
        cl_adult_mL_per_min_per_kg=5.0,
        vd_adult_L_per_kg=1.0,
        age_days=0.0,
        weight_kg=3.0,
        elimination="hepatic",
    )
    # t_half_neo is very large at day 0 due to 0.1 maturation
    assert res.dose_interval_h == 24.0


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        adjust_dose_neonatal(
            drug_name="Drug_J",
            adult_dose_mg_per_kg=0.0,
            cl_adult_mL_per_min_per_kg=5.0,
            vd_adult_L_per_kg=1.0,
            age_days=7.0,
            weight_kg=3.0,
        )


# ===========================================================================
# Phase 744 — simulate_neonatal_pk_744 and compute_maturation_factors tests
# ===========================================================================


def test_744_returns_dataclass():
    res = simulate_neonatal_pk_744("TestDrug", dose_mg_per_kg=10.0)
    assert isinstance(res, NeonatalPKResult744)


def test_744_cmax_positive():
    res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=5.0, route="iv")
    assert res.cmax_mg_L > 0


def test_744_premature_lower_cl_than_term():
    """Premature neonate (28w) → lower CL than term neonate (40w)."""
    premature = simulate_neonatal_pk_744(
        "Drug", dose_mg_per_kg=10.0, pma_weeks=28.0, cl_adult_L_per_h=5.0
    )
    term = simulate_neonatal_pk_744(
        "Drug", dose_mg_per_kg=10.0, pma_weeks=40.0, cl_adult_L_per_h=5.0
    )
    assert premature.cl_neonate_L_per_h < term.cl_neonate_L_per_h


def test_744_term_neonate_lower_cl_than_adult():
    """Term neonate CL should be lower than adult CL."""
    res = simulate_neonatal_pk_744(
        "Drug", dose_mg_per_kg=10.0, pma_weeks=40.0, cl_adult_L_per_h=5.0
    )
    assert res.cl_neonate_L_per_h < res.cl_adult_L_per_h


def test_744_vd_neonate_greater_than_adult():
    """Neonatal Vd is scaled up due to higher body water fraction."""
    res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, vd_adult_L=50.0, pma_weeks=40.0)
    assert res.vd_neonate_L > res.vd_adult_L


def test_744_cyp_maturation_between_0_and_1():
    for pma in (25.0, 32.0, 40.0, 50.0, 60.0):
        res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=5.0, pma_weeks=pma)
        assert 0.0 <= res.cyp_maturation_fraction <= 1.0


def test_744_renal_maturation_between_0_and_1():
    for pma in (25.0, 32.0, 40.0, 52.0, 60.0):
        res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=5.0, pma_weeks=pma)
        assert 0.0 <= res.renal_maturation_fraction <= 1.0


def test_compute_maturation_factors_keys():
    factors = compute_maturation_factors(40.0)
    assert "cyp_maturation" in factors
    assert "renal_maturation" in factors
    assert "vd_scaling" in factors


def test_744_auc_positive():
    res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, route="iv")
    assert res.auc_mg_h_per_L > 0


def test_744_t_half_positive():
    res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, route="iv")
    assert res.t_half_h > 0


def test_744_higher_pma_higher_cl():
    """More mature neonate (higher PMA) → higher CL."""
    res28 = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, pma_weeks=28.0)
    res44 = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, pma_weeks=44.0)
    assert res44.cl_neonate_L_per_h > res28.cl_neonate_L_per_h


def test_744_fup_neonate_gte_fup_adult():
    """Reduced protein binding → fup_neonate >= fup_adult for neonates."""
    fup_adult = 0.5
    res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, pma_weeks=38.0, fup_adult=fup_adult)
    assert res.fup_neonate >= fup_adult


def test_744_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg_per_kg"):
        simulate_neonatal_pk_744("Drug", dose_mg_per_kg=0.0)


def test_744_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg_per_kg"):
        simulate_neonatal_pk_744("Drug", dose_mg_per_kg=-5.0)


def test_744_pma_too_low_raises():
    with pytest.raises(ValueError, match="pma_weeks"):
        simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, pma_weeks=20.0)


def test_744_notes_nonempty():
    res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0)
    assert len(res.notes) > 0


def test_744_dose_mg_consistent():
    """dose_mg should equal dose_mg_per_kg * weight_kg."""
    res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, weight_kg=3.5)
    assert abs(res.dose_mg - 10.0 * 3.5) < 1e-9


def test_744_oral_route_cmax_positive():
    res = simulate_neonatal_pk_744("Drug", dose_mg_per_kg=10.0, route="oral", ka_per_h=0.5)
    assert res.cmax_mg_L > 0
