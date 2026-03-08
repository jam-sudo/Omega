"""Tests for Phase 610 — Drug Distribution in Adipose Tissue."""

import pytest

from omega_pbpk.core.adipose_distribution_p610 import (
    AdiposeDrugDistributionResult,
    simulate_adipose_distribution,
)

# Default parameters for quick tests
DEFAULTS = dict(
    drug_name="LipophilicDrug",
    dose_mg=200.0,
    logP=3.0,
    cl_sys_L_per_h=5.0,
    vd_central_L=30.0,
    bmi=25.0,
    age_years=40.0,
    sex="male",
    t_end_h=48.0,
    dt_h=0.1,
)


def run_default(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_adipose_distribution(**params)


# --- Return type and structure ---


def test_returns_correct_type():
    result = run_default()
    assert isinstance(result, AdiposeDrugDistributionResult)


def test_drug_name_stored():
    result = run_default(drug_name="Cyclosporine")
    assert result.drug_name == "Cyclosporine"


def test_dose_stored():
    result = run_default(dose_mg=100.0)
    assert result.dose_mg == 100.0


def test_bmi_stored():
    result = run_default(bmi=30.0)
    assert result.bmi == pytest.approx(30.0)


def test_list_fields_are_lists():
    result = run_default()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.c_adipose_mg_kg, list)
    assert isinstance(result.mass_adipose_mg, list)


def test_list_lengths_consistent():
    result = run_default()
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_adipose_mg_kg) == n
    assert len(result.mass_adipose_mg) == n


def test_times_start_at_zero():
    result = run_default()
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-9)


def test_times_end_near_t_end():
    result = run_default(t_end_h=48.0)
    assert result.times_h[-1] == pytest.approx(48.0, abs=0.2)


def test_c_plasma_nonnegative():
    result = run_default()
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


def test_c_adipose_nonnegative():
    result = run_default()
    assert all(c >= 0.0 for c in result.c_adipose_mg_kg)


def test_mass_adipose_nonnegative():
    result = run_default()
    assert all(m >= 0.0 for m in result.mass_adipose_mg)


# --- PK metrics ---


def test_cmax_positive():
    result = run_default()
    assert result.cmax_plasma_mg_L > 0.0


def test_tmax_positive():
    result = run_default()
    assert result.tmax_plasma_h > 0.0


def test_auc_positive():
    result = run_default()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_cmax_matches_list():
    result = run_default()
    assert result.cmax_plasma_mg_L == pytest.approx(max(result.c_plasma_mg_L), rel=1e-6)


def test_t_half_effective_positive():
    result = run_default()
    assert result.t_half_effective_h > 0.0


# --- Adipose accumulation for lipophilic drugs ---


def test_lipophilic_accumulates_more_than_hydrophilic():
    """High logP drug should accumulate more in adipose tissue."""
    hydrophilic = run_default(logP=-1.0)
    lipophilic = run_default(logP=5.0)
    assert max(lipophilic.c_adipose_mg_kg) > max(hydrophilic.c_adipose_mg_kg)


def test_adipose_accumulation_ratio_positive():
    result = run_default(logP=4.0)
    assert result.adipose_accumulation_ratio > 0.0


def test_redistribution_flag_is_bool():
    result = run_default()
    assert isinstance(result.redistribution_phase, bool)


def test_redistribution_true_for_hydrophilic_with_large_adipose():
    """Very hydrophilic drug (logP=-2) has tiny Kp so vd_ad/Kp is huge, extending t_half_eff."""
    # kp = max(0.01, -2*2+1) = 0.01 -> vd_ad/kp >> vd_central -> long effective t½
    result = run_default(logP=-2.0)
    assert result.redistribution_phase is True


def test_redistribution_can_be_false_for_very_lipophilic():
    """Highly lipophilic drug (logP=5): Kp=11 is large so vd_ad/Kp is small, t_half_eff ~ 1-cpt."""
    # kp=11 -> vd_ad/kp is small -> effective t½ close to 1-cpt t½ -> redistribution False
    result = run_default(logP=5.0)
    assert result.redistribution_phase is False


# --- BMI effect on fat mass ---


def test_higher_bmi_gives_more_adipose_drug():
    """Higher BMI leads to larger fat mass and more total drug in adipose."""
    low_bmi = run_default(bmi=20.0)
    high_bmi = run_default(bmi=40.0)
    assert max(high_bmi.mass_adipose_mg) > max(low_bmi.mass_adipose_mg)


# --- Sex differences ---


def test_female_lower_concentration_per_kg_than_male_at_same_bmi():
    """Females have more fat at same BMI, so drug concentration per kg adipose is lower."""
    male = run_default(sex="male", bmi=25.0, age_years=40.0)
    female = run_default(sex="female", bmi=25.0, age_years=40.0)
    # Female fat_mass (7.05 kg) >> male fat_mass (2.0 kg, clamped).
    # Both accumulate similar total drug mass, but females dilute it over more kg.
    assert max(female.c_adipose_mg_kg) < max(male.c_adipose_mg_kg)


# --- Notes ---


def test_notes_is_string():
    result = run_default()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# --- Validation ---


def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        run_default(dose_mg=0.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError):
        run_default(cl_sys_L_per_h=0.0)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError):
        run_default(vd_central_L=0.0)


def test_raises_on_zero_bmi():
    with pytest.raises(ValueError):
        run_default(bmi=0.0)


def test_raises_on_invalid_sex():
    with pytest.raises(ValueError):
        run_default(sex="other")
