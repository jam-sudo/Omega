"""Tests for core/rectal_absorption_pk.py (Phase 663)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.rectal_absorption_pk import (
    RectalAbsorptionResult,
    compare_rectal_vs_oral,
    simulate_rectal_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_result() -> RectalAbsorptionResult:
    return simulate_rectal_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        ka_rectal_per_h=1.5,
        f_bypass=0.5,
        fh=0.7,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=12.0,
        dt_h=0.05,
    )


@pytest.fixture()
def high_extraction_result() -> RectalAbsorptionResult:
    """Drug with high hepatic extraction (fh=0.2) — large rectal advantage."""
    return simulate_rectal_pk(
        drug_name="HighExtraction",
        dose_mg=100.0,
        ka_rectal_per_h=1.5,
        f_bypass=0.5,
        fh=0.2,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )


# ---------------------------------------------------------------------------
# Type and basic presence tests
# ---------------------------------------------------------------------------


def test_returns_rectal_absorption_result(default_result):
    assert isinstance(default_result, RectalAbsorptionResult)


def test_cmax_positive(default_result):
    assert default_result.cmax_mg_L > 0


def test_auc_positive(default_result):
    assert default_result.auc_mg_h_per_L > 0


def test_t_half_positive(default_result):
    assert default_result.t_half_h > 0


def test_f_systemic_formula(default_result):
    """f_systemic should equal f_bypass + (1-f_bypass)*fh."""
    expected = 0.5 + 0.5 * 0.7
    assert abs(default_result.f_systemic - expected) < 1e-6


def test_f_hepatic_bypass_in_range(default_result):
    assert 0 < default_result.f_hepatic_bypass < 1


def test_tmax_positive(default_result):
    assert default_result.tmax_h > 0


def test_times_and_plasma_same_length(default_result):
    assert len(default_result.times_h) == len(default_result.c_plasma_mg_L)


def test_a_rectal_starts_at_dose(default_result):
    assert abs(default_result.a_rectal_mg[0] - 100.0) < 1e-6


def test_a_rectal_decreases(default_result):
    """Drug in rectal compartment should monotonically decrease."""
    a = default_result.a_rectal_mg
    assert all(a[i] >= a[i + 1] - 1e-9 for i in range(len(a) - 1))


def test_notes_nonempty(default_result):
    assert len(default_result.notes) > 0


# ---------------------------------------------------------------------------
# Dose scaling (linearity)
# ---------------------------------------------------------------------------


def test_double_dose_doubles_auc():
    r1 = simulate_rectal_pk("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    r2 = simulate_rectal_pk("Drug", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert abs(ratio - 2.0) < 0.01


def test_double_dose_doubles_cmax():
    r1 = simulate_rectal_pk("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    r2 = simulate_rectal_pk("Drug", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    ratio = r2.cmax_mg_L / r1.cmax_mg_L
    assert abs(ratio - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Bypass fraction effects
# ---------------------------------------------------------------------------


def test_higher_bypass_higher_auc():
    """More bypass → more drug avoids liver → higher AUC."""
    r_low = simulate_rectal_pk("Drug", dose_mg=100.0, f_bypass=0.2, fh=0.4)
    r_high = simulate_rectal_pk("Drug", dose_mg=100.0, f_bypass=0.8, fh=0.4)
    assert r_high.auc_mg_h_per_L > r_low.auc_mg_h_per_L


def test_high_extraction_rectal_advantage(high_extraction_result):
    """High hepatic extraction (fh=0.2) drug: rectal f_systemic > fh."""
    # f_systemic rectal = 0.5 + 0.5*0.2 = 0.6; oral would be fh = 0.2
    assert high_extraction_result.f_systemic > 0.2


def test_high_extraction_f_systemic_exceeds_oral_fh():
    """Rectal f_systemic > f_oral * fh for high-extraction drug."""
    fh = 0.2
    f_bypass = 0.5
    r = simulate_rectal_pk("Drug", dose_mg=100.0, f_bypass=f_bypass, fh=fh)
    # oral: F = 1.0 * fh = 0.2
    assert r.f_systemic > 1.0 * fh


# ---------------------------------------------------------------------------
# vs_oral_auc_ratio
# ---------------------------------------------------------------------------


def test_vs_oral_auc_ratio_positive(default_result):
    assert default_result.vs_oral_auc_ratio > 0


# ---------------------------------------------------------------------------
# compare_rectal_vs_oral
# ---------------------------------------------------------------------------


def test_compare_returns_dict():
    result = compare_rectal_vs_oral("Drug", dose_mg=100.0)
    assert isinstance(result, dict)


def test_compare_has_expected_keys():
    result = compare_rectal_vs_oral("Drug", dose_mg=100.0)
    assert "rectal_result" in result
    assert "oral_result" in result
    assert "auc_ratio" in result
    assert "cmax_ratio" in result
    assert "notes" in result


def test_compare_high_extraction_auc_ratio_gt_1():
    """For high extraction drug with bypass, rectal AUC > oral AUC."""
    result = compare_rectal_vs_oral(
        "Drug",
        dose_mg=100.0,
        f_oral=1.0,
        f_bypass=0.5,
        fh=0.2,
    )
    assert result["auc_ratio"] > 1.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_rectal_pk("Drug", dose_mg=0.0)


def test_validation_f_bypass_zero():
    with pytest.raises(ValueError, match="f_bypass"):
        simulate_rectal_pk("Drug", dose_mg=100.0, f_bypass=0.0)


def test_validation_f_bypass_one():
    with pytest.raises(ValueError, match="f_bypass"):
        simulate_rectal_pk("Drug", dose_mg=100.0, f_bypass=1.0)


def test_validation_fh_zero():
    with pytest.raises(ValueError, match="fh"):
        simulate_rectal_pk("Drug", dose_mg=100.0, fh=0.0)
