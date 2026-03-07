"""Tests for gut wall extraction prediction model — Phase 926."""

import pytest

from omega_pbpk.prediction.gut_wall_extraction import (
    GutWallExtractionResult,
    predict_gut_wall_extraction,
    screen_gut_extraction,
)

_VALID_LIMITING_FACTORS = {"cyp3a4_metabolism", "pgp_efflux", "poor_permeability", "none"}


# ── Basic return-type and structure ──────────────────────────────────────────


def test_return_type_is_correct():
    result = predict_gut_wall_extraction("TestDrug", logp=2.0, pka=7.0)
    assert isinstance(result, GutWallExtractionResult)


def test_fg_in_range():
    result = predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0)
    assert 0.0 <= result.fg <= 1.0


def test_e_gut_in_range():
    result = predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0)
    assert 0.0 <= result.e_gut <= 1.0


def test_fg_plus_e_gut_equals_one():
    result = predict_gut_wall_extraction("Drug", logp=1.5, pka=6.5)
    assert result.fg + result.e_gut == pytest.approx(1.0, abs=1e-9)


def test_fa_effective_leq_fa_input():
    result = predict_gut_wall_extraction(
        "Drug", logp=2.0, pka=7.0, fa_input=0.9, pgp_efflux_ratio=2.0
    )
    assert result.fa_effective <= result.fa_input + 1e-10


def test_f_intestinal_in_range():
    result = predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0)
    assert 0.0 <= result.f_intestinal <= 1.0


def test_f_intestinal_equals_fa_eff_times_fg():
    result = predict_gut_wall_extraction(
        "Drug", logp=2.0, pka=7.0, fa_input=0.85, pgp_efflux_ratio=1.5
    )
    expected = result.fa_effective * result.fg
    assert result.f_intestinal == pytest.approx(expected, rel=1e-6)


def test_fu_ent_in_valid_range():
    result = predict_gut_wall_extraction("Drug", logp=3.0, pka=7.0)
    assert 0.0 < result.fu_ent <= 1.0


def test_notes_non_empty():
    result = predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = predict_gut_wall_extraction("Midazolam", logp=3.89, pka=6.2)
    assert result.drug_name == "Midazolam"


# ── Model behaviour ──────────────────────────────────────────────────────────


def test_high_clint_gives_low_fg():
    result = predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0, clint_gut_L_per_h=200.0)
    assert result.fg < 0.5


def test_low_clint_gives_high_fg():
    result = predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0, clint_gut_L_per_h=0.1)
    assert result.fg > 0.8


def test_high_pgp_efflux_reduces_fa_effective():
    low_pgp = predict_gut_wall_extraction(
        "Drug", logp=2.0, pka=7.0, fa_input=0.9, pgp_efflux_ratio=0.0
    )
    high_pgp = predict_gut_wall_extraction(
        "Drug", logp=2.0, pka=7.0, fa_input=0.9, pgp_efflux_ratio=10.0
    )
    assert high_pgp.fa_effective < low_pgp.fa_effective


def test_zero_pgp_efflux_fa_effective_equals_fa_input():
    result = predict_gut_wall_extraction(
        "Drug", logp=2.0, pka=7.0, fa_input=0.85, pgp_efflux_ratio=0.0
    )
    assert result.fa_effective == pytest.approx(result.fa_input, rel=1e-9)


def test_limiting_factor_valid_value():
    result = predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0)
    assert result.limiting_factor in _VALID_LIMITING_FACTORS


def test_high_clint_limiting_factor_cyp3a4():
    result = predict_gut_wall_extraction(
        "Drug", logp=2.0, pka=7.0, clint_gut_L_per_h=500.0, pgp_efflux_ratio=0.0
    )
    assert result.limiting_factor == "cyp3a4_metabolism"


# ── Validation errors ─────────────────────────────────────────────────────────


def test_fa_input_negative_raises():
    with pytest.raises(ValueError, match="fa_input"):
        predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0, fa_input=-0.1)


def test_fa_input_greater_than_one_raises():
    with pytest.raises(ValueError, match="fa_input"):
        predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0, fa_input=1.1)


def test_clint_gut_negative_raises():
    with pytest.raises(ValueError, match="clint_gut_L_per_h"):
        predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0, clint_gut_L_per_h=-1.0)


def test_pgp_efflux_ratio_negative_raises():
    with pytest.raises(ValueError, match="pgp_efflux_ratio"):
        predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0, pgp_efflux_ratio=-0.5)


def test_qent_zero_raises():
    with pytest.raises(ValueError, match="qent_L_per_h"):
        predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0, qent_L_per_h=0.0)


def test_qent_negative_raises():
    with pytest.raises(ValueError, match="qent_L_per_h"):
        predict_gut_wall_extraction("Drug", logp=2.0, pka=7.0, qent_L_per_h=-5.0)


# ── screen_gut_extraction ────────────────────────────────────────────────────


def test_screen_returns_correct_length():
    compounds = [
        {"logp": 1.0, "pka": 6.0},
        {"logp": 3.0, "pka": 7.5},
        {"logp": 5.0, "pka": 8.0},
    ]
    results = screen_gut_extraction("Drug", compounds)
    assert len(results) == 3


def test_screen_sorted_by_f_intestinal_descending():
    compounds = [
        {"logp": 5.0, "pka": 7.0, "clint_gut_L_per_h": 100.0},
        {"logp": 1.0, "pka": 7.0, "clint_gut_L_per_h": 0.5},
        {"logp": 3.0, "pka": 7.0, "clint_gut_L_per_h": 10.0},
    ]
    results = screen_gut_extraction("Drug", compounds)
    for i in range(len(results) - 1):
        assert results[i].f_intestinal >= results[i + 1].f_intestinal


def test_screen_all_return_correct_type():
    compounds = [
        {"logp": 2.0, "pka": 7.0},
        {"logp": 4.0, "pka": 8.0},
    ]
    results = screen_gut_extraction("Drug", compounds)
    for r in results:
        assert isinstance(r, GutWallExtractionResult)
