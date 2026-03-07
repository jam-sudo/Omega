"""Tests for oral_bioavailability_model_p788 (Phase 788)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.oral_bioavailability_model_p788 import (
    OralBioavailabilityResultP788,
    predict_oral_bioavailability,
    screen_bioavailability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_default(**kwargs):
    defaults = dict(
        compound_name="TestCmpd",
        logp=2.5,
        mw_da=350.0,
        psa_A2=60.0,
        hbd_count=2,
        solubility_mg_L=200.0,
        cl_int_hepatic_L_per_h=5.0,
        fup=0.2,
        q_portal_L_per_h=75.0,
        pgp_substrate=False,
    )
    defaults.update(kwargs)
    return predict_oral_bioavailability(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _make_default()
    assert isinstance(result, OralBioavailabilityResultP788)


def test_result_is_frozen():
    result = _make_default()
    with pytest.raises((AttributeError, TypeError)):
        result.fa = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# f_total = fa * fg * fh
# ---------------------------------------------------------------------------


def test_f_total_equals_product():
    result = _make_default()
    expected = result.fa * result.fg * result.fh
    assert abs(result.f_total - expected) < 1e-9, (
        f"f_total={result.f_total:.6f}, fa*fg*fh={expected:.6f}"
    )


def test_f_total_pct_equals_f_total_times_100():
    result = _make_default()
    assert abs(result.f_total_pct - result.f_total * 100.0) < 1e-9


# ---------------------------------------------------------------------------
# f_total range [0, 1]
# ---------------------------------------------------------------------------


def test_f_total_in_0_1():
    result = _make_default()
    assert 0.0 <= result.f_total <= 1.0


def test_fa_in_0_1():
    result = _make_default()
    assert 0.0 <= result.fa <= 1.0


def test_fg_in_0_1():
    result = _make_default()
    assert 0.0 <= result.fg <= 1.0


def test_fh_in_0_1():
    result = _make_default()
    assert 0.0 <= result.fh <= 1.0


# ---------------------------------------------------------------------------
# High CLint → lower fh
# ---------------------------------------------------------------------------


def test_high_clint_gives_lower_fh():
    r_low = _make_default(cl_int_hepatic_L_per_h=1.0)
    r_high = _make_default(cl_int_hepatic_L_per_h=500.0)
    assert r_high.fh < r_low.fh, f"fh_high={r_high.fh:.3f} should < fh_low={r_low.fh:.3f}"


def test_high_clint_gives_lower_f_total():
    r_low = _make_default(cl_int_hepatic_L_per_h=1.0)
    r_high = _make_default(cl_int_hepatic_L_per_h=1000.0)
    assert r_high.f_total < r_low.f_total


# ---------------------------------------------------------------------------
# P-gp substrate → lower F
# ---------------------------------------------------------------------------


def test_pgp_substrate_reduces_f_total():
    r_no_pgp = _make_default(pgp_substrate=False, solubility_mg_L=200.0)
    r_pgp = _make_default(pgp_substrate=True, solubility_mg_L=200.0)
    assert r_pgp.f_total < r_no_pgp.f_total, (
        f"P-gp: {r_pgp.f_total:.3f} should < no P-gp: {r_no_pgp.f_total:.3f}"
    )


def test_pgp_substrate_reduces_fg():
    r_no_pgp = _make_default(pgp_substrate=False)
    r_pgp = _make_default(pgp_substrate=True)
    assert r_pgp.fg < r_no_pgp.fg


# ---------------------------------------------------------------------------
# BCS class
# ---------------------------------------------------------------------------


def test_bcs_class_in_valid_set():
    result = _make_default()
    assert result.bcs_class in {"I", "II", "III", "IV"}


def test_bcs_class_I_high_perm_high_sol():
    # High logP → high Peff; high solubility >> mw_da threshold
    result = _make_default(logp=4.0, psa_A2=30.0, mw_da=200.0, solubility_mg_L=5000.0)
    assert result.bcs_class == "I"


def test_bcs_class_III_low_perm_high_sol():
    # Low logP + high PSA → low Peff; high solubility
    result = _make_default(logp=-1.0, psa_A2=180.0, mw_da=200.0, solubility_mg_L=5000.0)
    assert result.bcs_class == "III"


# ---------------------------------------------------------------------------
# main_limitation
# ---------------------------------------------------------------------------


def test_main_limitation_valid_values():
    result = _make_default()
    assert result.main_limitation in {"absorption", "gut_extraction", "hepatic_extraction", "none"}


def test_main_limitation_hepatic_for_high_clint():
    # Very high CLint → hepatic extraction dominates
    result = _make_default(
        cl_int_hepatic_L_per_h=500.0, fup=0.5, solubility_mg_L=1000.0, logp=2.0, psa_A2=50.0
    )
    assert result.main_limitation == "hepatic_extraction"


def test_main_limitation_absorption_for_low_perm():
    # Very low Peff: high PSA, low logP, high MW, low CLint → absorption limited
    result = _make_default(
        logp=-2.0, psa_A2=200.0, mw_da=700.0, solubility_mg_L=1000.0, cl_int_hepatic_L_per_h=0.1
    )
    assert result.main_limitation == "absorption"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_valid_values():
    result = _make_default()
    assert result.confidence in {"high", "medium", "low"}


def test_confidence_high_for_lipinski_compliant():
    result = _make_default(logp=2.0, mw_da=350.0, psa_A2=80.0, solubility_mg_L=100.0)
    assert result.confidence == "high"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw_da"):
        _make_default(mw_da=0.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw_da"):
        _make_default(mw_da=-100.0)


def test_zero_fup_raises():
    with pytest.raises(ValueError, match="fup"):
        _make_default(fup=0.0)


def test_fup_greater_than_1_raises():
    with pytest.raises(ValueError, match="fup"):
        _make_default(fup=1.5)


# ---------------------------------------------------------------------------
# screen_bioavailability
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "logp": 2.0, "mw_da": 300.0, "psa_A2": 60.0},
        {"compound_name": "B", "logp": 1.0, "mw_da": 400.0, "psa_A2": 80.0},
    ]
    results = screen_bioavailability(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_f_total_descending():
    compounds = [
        {
            "compound_name": "Low",
            "logp": -1.0,
            "mw_da": 600.0,
            "psa_A2": 180.0,
            "cl_int_hepatic_L_per_h": 100.0,
            "fup": 0.05,
        },
        {
            "compound_name": "High",
            "logp": 2.5,
            "mw_da": 300.0,
            "psa_A2": 50.0,
            "solubility_mg_L": 500.0,
            "cl_int_hepatic_L_per_h": 1.0,
            "fup": 0.5,
        },
    ]
    results = screen_bioavailability(compounds)
    f_totals = [r.f_total for r in results]
    assert f_totals == sorted(f_totals, reverse=True), f"Not sorted descending: {f_totals}"


def test_screen_all_results_are_correct_type():
    compounds = [
        {"compound_name": "A", "logp": 2.0, "mw_da": 300.0, "psa_A2": 60.0},
        {"compound_name": "B", "logp": 3.0, "mw_da": 350.0, "psa_A2": 70.0},
        {"compound_name": "C", "logp": 1.0, "mw_da": 250.0, "psa_A2": 90.0},
    ]
    results = screen_bioavailability(compounds)
    for r in results:
        assert isinstance(r, OralBioavailabilityResultP788)
