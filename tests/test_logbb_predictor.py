"""Tests for logBB predictor (Phase 599)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.logbb_predictor import LogBBResult, predict_logbb, screen_cns_compounds

# ---------------------------------------------------------------------------
# Basic return type and field tests
# ---------------------------------------------------------------------------


def test_returns_result():
    result = predict_logbb("aspirin", logP=1.2, mw=180.0, psa=63.0)
    assert isinstance(result, LogBBResult)


def test_logbb_is_float():
    result = predict_logbb("aspirin", logP=1.2, mw=180.0, psa=63.0)
    assert isinstance(result.logBB, float)


def test_bb_ratio_positive():
    result = predict_logbb("aspirin", logP=1.2, mw=180.0, psa=63.0)
    assert result.bb_ratio > 0


# ---------------------------------------------------------------------------
# logBB model behavior tests
# ---------------------------------------------------------------------------


def test_high_logP_high_logBB():
    """Higher logP should give higher logBB."""
    result_high = predict_logbb("drug_high", logP=5.0, mw=300.0, psa=40.0)
    result_low = predict_logbb("drug_low", logP=1.0, mw=300.0, psa=40.0)
    assert result_high.logBB > result_low.logBB


def test_high_psa_low_logBB():
    """Higher PSA should give lower logBB."""
    result_low_psa = predict_logbb("drug_low_psa", logP=2.0, mw=300.0, psa=20.0)
    result_high_psa = predict_logbb("drug_high_psa", logP=2.0, mw=300.0, psa=150.0)
    assert result_high_psa.logBB < result_low_psa.logBB


def test_cns_penetration_valid():
    result = predict_logbb("drug", logP=2.0, mw=300.0, psa=50.0)
    assert result.cns_penetration in {"high", "moderate", "low", "negligible"}


def test_pgp_efflux_high_for_large_polar():
    """mw=500, psa=100 should give high P-gp efflux risk."""
    result = predict_logbb("big_polar", logP=1.0, mw=500.0, psa=100.0)
    assert result.pgp_efflux_risk == "high"


def test_pgp_efflux_low_for_small():
    """mw=200, psa=40 should give low P-gp efflux risk."""
    result = predict_logbb("small", logP=2.0, mw=200.0, psa=40.0)
    assert result.pgp_efflux_risk == "low"


def test_bbb_class_permeable():
    """psa=40, mw=300 should be permeable."""
    result = predict_logbb("permeable", logP=2.0, mw=300.0, psa=40.0)
    assert result.bbb_permeability_class == "permeable"


def test_bbb_class_impermeable():
    """psa=120, mw=500 should be impermeable."""
    result = predict_logbb("impermeable", logP=1.0, mw=500.0, psa=120.0)
    assert result.bbb_permeability_class == "impermeable"


def test_brain_conc_proportional_to_plasma():
    """2x plasma_conc should give 2x brain_conc."""
    r1 = predict_logbb("drug", logP=2.0, mw=300.0, psa=50.0, plasma_conc_mg_L=1.0)
    r2 = predict_logbb("drug", logP=2.0, mw=300.0, psa=50.0, plasma_conc_mg_L=2.0)
    assert r2.predicted_brain_conc_mg_L == pytest.approx(
        2.0 * r1.predicted_brain_conc_mg_L, rel=1e-9
    )


def test_high_mw_penalty():
    """mw=600 should give lower logBB than mw=300."""
    r_low = predict_logbb("drug_low_mw", logP=2.0, mw=300.0, psa=50.0)
    r_high = predict_logbb("drug_high_mw", logP=2.0, mw=600.0, psa=50.0)
    assert r_high.logBB < r_low.logBB


def test_hbd_penalty():
    """n_hbd=4 should give lower logBB than n_hbd=0."""
    r0 = predict_logbb("drug_no_hbd", logP=2.0, mw=300.0, psa=50.0, n_hbd=0)
    r4 = predict_logbb("drug_hbd4", logP=2.0, mw=300.0, psa=50.0, n_hbd=4)
    assert r4.logBB < r0.logBB


def test_basic_drug_bonus():
    """pka_base=9 should give higher logBB than no pka."""
    r_no_pka = predict_logbb("neutral", logP=2.0, mw=300.0, psa=50.0)
    r_basic = predict_logbb("basic", logP=2.0, mw=300.0, psa=50.0, pka_base=9.0)
    assert r_basic.logBB > r_no_pka.logBB


# ---------------------------------------------------------------------------
# screen_cns_compounds tests
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"name": "drug_a", "logP": 2.0, "mw": 300.0, "psa": 50.0, "n_hbd": 1},
        {"name": "drug_b", "logP": 4.0, "mw": 250.0, "psa": 30.0, "n_hbd": 0},
    ]
    results = screen_cns_compounds(compounds)
    assert isinstance(results, list)


def test_screen_sorted_descending():
    compounds = [
        {"name": "drug_a", "logP": 1.0, "mw": 300.0, "psa": 80.0, "n_hbd": 3},
        {"name": "drug_b", "logP": 5.0, "mw": 250.0, "psa": 20.0, "n_hbd": 0},
        {"name": "drug_c", "logP": 3.0, "mw": 350.0, "psa": 40.0, "n_hbd": 1},
    ]
    results = screen_cns_compounds(compounds)
    for i in range(len(results) - 1):
        assert results[i].logBB >= results[i + 1].logBB


def test_screen_all_logbb_results():
    compounds = [
        {"name": "drug_a", "logP": 2.0, "mw": 300.0, "psa": 50.0, "n_hbd": 1},
        {"name": "drug_b", "logP": 4.0, "mw": 250.0, "psa": 30.0, "n_hbd": 0},
    ]
    results = screen_cns_compounds(compounds)
    assert all(isinstance(r, LogBBResult) for r in results)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_logbb("drug", logP=2.0, mw=0.0, psa=50.0)


def test_invalid_psa_negative_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_logbb("drug", logP=2.0, mw=300.0, psa=-1.0)


def test_invalid_plasma_conc_raises():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        predict_logbb("drug", logP=2.0, mw=300.0, psa=50.0, plasma_conc_mg_L=0.0)


def test_invalid_n_hbd_negative_raises():
    with pytest.raises(ValueError, match="n_hbd"):
        predict_logbb("drug", logP=2.0, mw=300.0, psa=50.0, n_hbd=-1)


def test_notes_nonempty():
    result = predict_logbb("drug", logP=2.0, mw=300.0, psa=50.0)
    assert len(result.notes) > 0
