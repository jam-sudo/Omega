"""Tests for first-pass metabolism calculator (Phase 173)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.first_pass import (
    FirstPassResult,
    calculate_first_pass,
    sensitivity_first_pass,
)


# ── Default behaviour ──────────────────────────────────────────────────


def test_default_returns_first_pass_result():
    result = calculate_first_pass("DrugA")
    assert isinstance(result, FirstPassResult)


def test_default_drug_name():
    result = calculate_first_pass("DrugA")
    assert result.drug_name == "DrugA"


def test_default_fa_is_one():
    result = calculate_first_pass("DrugA")
    assert result.fa == 1.0


def test_default_bioavailability_equals_product():
    result = calculate_first_pass("DrugA")
    expected = result.fa * result.fg * result.fh * result.fl
    assert math.isclose(result.bioavailability_f, expected, rel_tol=1e-9)


def test_default_first_pass_loss_pct():
    result = calculate_first_pass("DrugA")
    expected_loss = (1.0 - result.bioavailability_f) * 100.0
    assert math.isclose(result.first_pass_loss_pct, expected_loss, rel_tol=1e-9)


# ── High / Low extraction ──────────────────────────────────────────────


def test_high_clh_gives_high_extraction():
    result = calculate_first_pass("DrugB", clh_intrinsic_L_per_h=500.0)
    assert result.eh_liver > 0.7
    assert result.high_extraction is True
    assert result.bioavailability_f < 0.3


def test_low_clh_gives_low_extraction():
    result = calculate_first_pass("DrugC", clh_intrinsic_L_per_h=1.0)
    assert result.eh_liver < 0.3
    assert result.high_extraction is False
    assert result.bioavailability_f > 0.5


def test_high_extraction_flag_at_boundary():
    # Eh exactly at 0.7 should not be flagged (> 0.7 required, not >=)
    # With fup==fub, CLh_eff = CLh. Eh=0.7 => CLh/(Qh+CLh)=0.7 => CLh=0.7*Qh/0.3
    qh = 90.0
    clh = 0.7 * qh / 0.3
    result = calculate_first_pass(
        "DrugD", clh_intrinsic_L_per_h=clh, fup=0.5, fub=0.5,
    )
    assert result.high_extraction is False


# ── Absorption edge cases ──────────────────────────────────────────────


def test_fa_zero_gives_zero_bioavailability():
    result = calculate_first_pass("DrugE", fa=0.0)
    assert result.bioavailability_f == 0.0
    assert result.first_pass_loss_pct == 100.0


def test_fa_one_full_absorption():
    result = calculate_first_pass("DrugF", fa=1.0)
    assert result.fa == 1.0
    assert result.bioavailability_f > 0.0


# ── Protein-binding correction ─────────────────────────────────────────


def test_fu_correction_changes_result():
    with_corr = calculate_first_pass("DrugG", fup=0.3, fub=0.5, use_fu_correction=True)
    without_corr = calculate_first_pass("DrugG", fup=0.3, fub=0.5, use_fu_correction=False)
    assert with_corr.bioavailability_f != without_corr.bioavailability_f


def test_fu_correction_off_uses_raw_clh():
    result = calculate_first_pass(
        "DrugH", clh_intrinsic_L_per_h=20.0, qh_L_per_h=90.0,
        use_fu_correction=False,
    )
    expected_eh = 20.0 / (90.0 + 20.0)
    assert math.isclose(result.eh_liver, expected_eh, rel_tol=1e-9)


# ── Lung extraction ───────────────────────────────────────────────────


def test_fl_less_than_one_reduces_f():
    full = calculate_first_pass("DrugI", fl=1.0)
    partial = calculate_first_pass("DrugI", fl=0.5)
    assert partial.bioavailability_f < full.bioavailability_f


# ── Validation errors ──────────────────────────────────────────────────


def test_invalid_fa_negative():
    with pytest.raises(ValueError, match="fa"):
        calculate_first_pass("X", fa=-0.1)


def test_invalid_fa_above_one():
    with pytest.raises(ValueError, match="fa"):
        calculate_first_pass("X", fa=1.1)


def test_invalid_clg_zero():
    with pytest.raises(ValueError, match="clg_L_per_h"):
        calculate_first_pass("X", clg_L_per_h=0.0)


def test_invalid_clh_negative():
    with pytest.raises(ValueError, match="clh_intrinsic_L_per_h"):
        calculate_first_pass("X", clh_intrinsic_L_per_h=-5.0)


def test_invalid_fup_zero():
    with pytest.raises(ValueError, match="fup"):
        calculate_first_pass("X", fup=0.0)


def test_invalid_fub_above_one():
    with pytest.raises(ValueError, match="fub"):
        calculate_first_pass("X", fub=1.5)


def test_invalid_fl_negative():
    with pytest.raises(ValueError, match="fl"):
        calculate_first_pass("X", fl=-0.1)


# ── Edge cases ─────────────────────────────────────────────────────────


def test_very_high_clearance():
    result = calculate_first_pass("DrugJ", clh_intrinsic_L_per_h=10000.0)
    assert result.eh_liver > 0.9
    assert result.bioavailability_f < 0.1


def test_very_low_clearance():
    result = calculate_first_pass(
        "DrugK", clg_L_per_h=0.01, clh_intrinsic_L_per_h=0.01,
    )
    assert result.bioavailability_f > 0.99


def test_first_pass_loss_in_range():
    result = calculate_first_pass("DrugL")
    assert 0.0 <= result.first_pass_loss_pct <= 100.0


def test_notes_is_string():
    result = calculate_first_pass("DrugM")
    assert isinstance(result.notes, str)


# ── Sensitivity analysis ──────────────────────────────────────────────


def test_sensitivity_returns_correct_count():
    results = sensitivity_first_pass(
        "DrugN",
        fg_range=(1.0, 10.0, 3),
        fh_range=(5.0, 50.0, 4),
    )
    assert len(results) == 3 * 4


def test_sensitivity_all_results_are_first_pass_result():
    results = sensitivity_first_pass(
        "DrugO",
        fg_range=(1.0, 5.0, 2),
        fh_range=(10.0, 30.0, 2),
    )
    for r in results:
        assert isinstance(r, FirstPassResult)
        assert 0.0 <= r.first_pass_loss_pct <= 100.0
