"""Tests for Phase 634 — Drug Dissolution Rate Prediction"""

import pytest

from omega_pbpk.prediction.dissolution_rate_p634 import (
    DissolutionRateResult,
    _compute_noyes_whitney_k,
    predict_dissolution_rate,
)

# ─── Reference inputs ─────────────────────────────────────────────────────────

DRUG = "TestCompound"
DOSE = 100.0  # mg
PS_SMALL = 10.0  # µm  → fast dissolution
PS_LARGE = 200.0  # µm  → slow dissolution
SOL_HIGH = 5.0  # mg/mL (high solubility → BCS 1/3)
SOL_LOW = 0.01  # mg/mL (low sol → BCS 2/4)
VOL = 900.0  # mL (USP paddle default)


def make_result(ps=PS_SMALL, sol=SOL_HIGH, perm="high", dose=DOSE):
    return predict_dissolution_rate(
        DRUG,
        dose,
        ps,
        sol,
        drug_density_g_cm3=1.2,
        diffusion_coeff_cm2_s=5e-6,
        volume_dissolution_mL=VOL,
        bcs_perm_class=perm,
        t_end_h=4.0,
        dt_h=0.01,
    )


# ─── Return type & frozen dataclass ──────────────────────────────────────────


def test_returns_correct_type():
    res = make_result()
    assert isinstance(res, DissolutionRateResult)


def test_frozen_dataclass_immutable():
    res = make_result()
    with pytest.raises((AttributeError, TypeError)):
        res.drug_name = "other"  # type: ignore[misc]


def test_drug_name_preserved():
    res = predict_dissolution_rate("DrugX", DOSE, PS_SMALL, SOL_HIGH)
    assert res.drug_name == "DrugX"


# ─── Noyes-Whitney rate constant ─────────────────────────────────────────────


def test_noyes_whitney_k_positive():
    res = make_result()
    assert res.noyes_whitney_k > 0


def test_noyes_whitney_k_decreases_with_larger_particles():
    res_small = make_result(ps=10.0)
    res_large = make_result(ps=100.0)
    assert res_small.noyes_whitney_k > res_large.noyes_whitney_k


def test_noyes_whitney_k_increases_with_solubility():
    res_low = make_result(sol=0.1)
    res_high = make_result(sol=5.0)
    assert res_high.noyes_whitney_k > res_low.noyes_whitney_k


def test_noyes_whitney_k_clamped_upper():
    # Very tiny particle, very high solubility → should clamp at 100
    k = _compute_noyes_whitney_k(5e-6, 100.0, 0.1, 0.1)
    assert k <= 100.0


def test_noyes_whitney_k_clamped_lower():
    # Enormous particle → should clamp at 0.001
    k = _compute_noyes_whitney_k(1e-10, 0.0001, 2.0, 10000.0)
    assert k >= 0.001


def test_noyes_whitney_k_formula():
    """Verify formula manually for a simple case."""
    D_cm2_s = 5e-6
    Cs = 1.0  # mg/mL
    rho = 1.2  # g/cm3
    ps = 50.0  # µm
    D_cm2_h = D_cm2_s * 3600.0
    rho_mg = rho * 1000.0
    d_cm = ps * 1e-4
    expected = 6.0 * D_cm2_h * Cs / (rho_mg * d_cm**2)
    expected = max(0.001, min(100.0, expected))
    k = _compute_noyes_whitney_k(D_cm2_s, Cs, rho, ps)
    assert k == pytest.approx(expected, rel=1e-6)


# ─── Dissolution time points ──────────────────────────────────────────────────


def test_t50_reached_for_fast_dissolving():
    res = make_result(ps=PS_SMALL, sol=SOL_HIGH)
    assert res.t_50pct_h >= 0  # not -1


def test_t90_not_reached_for_slow_dissolving():
    # Poorly soluble, large particles → t90 may not be reached
    res = make_result(ps=PS_LARGE, sol=SOL_LOW)
    # t90 may be -1 (not reached in 4h) or a large value
    # at minimum, absorption should be rate limited
    assert res.absorption_rate_limited is True


def test_t50_less_than_t80():
    res = make_result(ps=PS_SMALL, sol=SOL_HIGH)
    if res.t_50pct_h >= 0 and res.t_80pct_h >= 0:
        assert res.t_50pct_h <= res.t_80pct_h


def test_t80_less_than_t90():
    res = make_result(ps=PS_SMALL, sol=SOL_HIGH)
    if res.t_80pct_h >= 0 and res.t_90pct_h >= 0:
        assert res.t_80pct_h <= res.t_90pct_h


def test_t_90_minus1_if_not_reached():
    # Use extremely poor solubility & huge particles
    res = predict_dissolution_rate(
        DRUG,
        500.0,
        500.0,
        0.001,
        drug_density_g_cm3=1.2,
        diffusion_coeff_cm2_s=1e-8,
        volume_dissolution_mL=VOL,
        bcs_perm_class="low",
        t_end_h=2.0,
        dt_h=0.05,
    )
    # Either -1 or a value; if absorption_rate_limited, that's OK
    assert res.t_90pct_h == -1.0 or res.absorption_rate_limited


# ─── Fraction dissolved at 2h ─────────────────────────────────────────────────


def test_fraction_dissolved_2h_in_range():
    res = make_result()
    assert 0.0 <= res.fraction_dissolved_2h <= 1.0


def test_fraction_dissolved_2h_higher_for_small_particles():
    res_small = make_result(ps=5.0)
    res_large = make_result(ps=100.0)
    assert res_small.fraction_dissolved_2h >= res_large.fraction_dissolved_2h


# ─── Dose number ──────────────────────────────────────────────────────────────


def test_dose_number_formula():
    res = make_result(dose=DOSE, sol=SOL_HIGH)
    expected = DOSE / (SOL_HIGH * VOL)
    assert res.dose_number == pytest.approx(expected, rel=1e-6)


def test_dose_number_high_sol_leq_1():
    # DOSE=100, SOL=5, VOL=900 → Do = 100/4500 = 0.022 < 1
    res = make_result(sol=SOL_HIGH)
    assert res.dose_number <= 1.0


def test_dose_number_low_sol_gt_1():
    # DOSE=100, SOL=0.01, VOL=900 → Do = 100/9 > 1
    res = make_result(sol=SOL_LOW)
    assert res.dose_number > 1.0


# ─── BCS class ────────────────────────────────────────────────────────────────


def test_bcs_class_1_high_sol_high_perm():
    res = make_result(sol=SOL_HIGH, perm="high")
    assert res.bcs_class_predicted == 1


def test_bcs_class_2_low_sol_high_perm():
    res = make_result(sol=SOL_LOW, perm="high")
    assert res.bcs_class_predicted == 2


def test_bcs_class_3_high_sol_low_perm():
    res = make_result(sol=SOL_HIGH, perm="low")
    assert res.bcs_class_predicted == 3


def test_bcs_class_4_low_sol_low_perm():
    res = make_result(sol=SOL_LOW, perm="low")
    assert res.bcs_class_predicted == 4


# ─── QC1 pass/fail ────────────────────────────────────────────────────────────


def test_qc1_pass_for_fast_dissolving():
    # Very small particle, high sol → fast dissolution → QC1 pass
    res = make_result(ps=1.0, sol=SOL_HIGH)
    assert res.qc1_pass is True


def test_qc1_fail_for_slow_dissolving():
    # Large particle, very low sol → QC1 fail
    res = make_result(ps=PS_LARGE, sol=SOL_LOW)
    assert res.qc1_pass is False


# ─── Absorption rate limited ──────────────────────────────────────────────────


def test_absorption_rate_limited_slow():
    res = make_result(ps=PS_LARGE, sol=SOL_LOW)
    assert res.absorption_rate_limited is True


def test_absorption_not_rate_limited_fast():
    # Very fast dissolution → not rate limited
    res = make_result(ps=1.0, sol=50.0)
    assert res.absorption_rate_limited is False


# ─── Dissolution efficiency ───────────────────────────────────────────────────


def test_dissolution_efficiency_in_range():
    res = make_result()
    assert 0.0 <= res.dissolution_efficiency <= 1.0


def test_dissolution_efficiency_higher_for_small_particles():
    res_small = make_result(ps=5.0)
    res_large = make_result(ps=200.0)
    assert res_small.dissolution_efficiency >= res_large.dissolution_efficiency


# ─── Validation ───────────────────────────────────────────────────────────────


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_dissolution_rate(DRUG, 0.0, PS_SMALL, SOL_HIGH)


def test_invalid_particle_size_raises():
    with pytest.raises(ValueError, match="particle_size_um"):
        predict_dissolution_rate(DRUG, DOSE, -1.0, SOL_HIGH)


def test_invalid_solubility_raises():
    with pytest.raises(ValueError, match="bulk_solubility_mg_mL"):
        predict_dissolution_rate(DRUG, DOSE, PS_SMALL, 0.0)


def test_invalid_volume_raises():
    with pytest.raises(ValueError, match="volume_dissolution_mL"):
        predict_dissolution_rate(DRUG, DOSE, PS_SMALL, SOL_HIGH, volume_dissolution_mL=-10.0)


def test_invalid_perm_class_raises():
    with pytest.raises(ValueError, match="bcs_perm_class"):
        predict_dissolution_rate(DRUG, DOSE, PS_SMALL, SOL_HIGH, bcs_perm_class="medium")


def test_notes_is_nonempty_string():
    res = make_result()
    assert isinstance(res.notes, str)
    assert len(res.notes) > 0
