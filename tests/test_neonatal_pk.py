"""Tests for neonatal_pk module — Phase 197."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.neonatal_pk import (
    NeonatalDoseResult,
    NeonatalScalingFactors,
    adjust_dose_neonatal,
    neonatal_scaling,
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
    assert pytest.approx(res.neonatal_dose_mg, rel=1e-6) == res.neonatal_dose_mg_per_kg * res.weight_kg


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
