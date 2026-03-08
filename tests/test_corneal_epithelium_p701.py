"""Tests for Phase 701 — Corneal Epithelium Barrier."""

import pytest

from omega_pbpk.core.corneal_epithelium_p701 import (
    CornealEpitheliumResult,
    simulate_corneal_epithelium,
)


@pytest.fixture
def base_result():
    return simulate_corneal_epithelium("TestDrug", dose_mg=0.5, logP=2.0, mw_Da=400.0)


def test_return_type(base_result):
    assert isinstance(base_result, CornealEpitheliumResult)


def test_drug_name(base_result):
    assert base_result.drug_name == "TestDrug"


def test_dose_stored(base_result):
    assert base_result.dose_mg == 0.5


def test_times_not_empty(base_result):
    assert len(base_result.times_h) > 0


def test_all_lists_same_length(base_result):
    n = len(base_result.times_h)
    assert len(base_result.c_precorneal_mg_mL) == n
    assert len(base_result.c_epithelium_mg_g) == n
    assert len(base_result.c_stroma_mg_mL) == n
    assert len(base_result.c_endothelium_mg_mL) == n
    assert len(base_result.c_aqueous_mg_mL) == n


def test_non_negative_precorneal(base_result):
    assert all(c >= 0 for c in base_result.c_precorneal_mg_mL)


def test_non_negative_epithelium(base_result):
    assert all(c >= 0 for c in base_result.c_epithelium_mg_g)


def test_non_negative_stroma(base_result):
    assert all(c >= 0 for c in base_result.c_stroma_mg_mL)


def test_non_negative_endothelium(base_result):
    assert all(c >= 0 for c in base_result.c_endothelium_mg_mL)


def test_non_negative_aqueous(base_result):
    assert all(c >= 0 for c in base_result.c_aqueous_mg_mL)


def test_initial_precorneal_high(base_result):
    assert base_result.c_precorneal_mg_mL[0] == pytest.approx(0.5 / 0.03, rel=0.01)


def test_precorneal_decays_rapidly(base_result):
    # After ~10 time steps, precorneal should be much lower than initial
    mid = len(base_result.c_precorneal_mg_mL) // 4
    assert base_result.c_precorneal_mg_mL[mid] < base_result.c_precorneal_mg_mL[0] * 0.1


def test_aqueous_rises_then_falls():
    r = simulate_corneal_epithelium("T", dose_mg=0.5, logP=2.0, mw_Da=400.0, t_end_h=12.0)
    aq = r.c_aqueous_mg_mL
    max_idx = aq.index(max(aq))
    assert max_idx > 0
    assert max_idx < len(aq) - 1


def test_cmax_aqueous_positive(base_result):
    assert base_result.cmax_aqueous_mg_mL > 0


def test_cmax_precorneal_positive(base_result):
    assert base_result.cmax_precorneal_mg_mL > 0


def test_kp_epithelium_range(base_result):
    assert 0.5 <= base_result.kp_epithelium <= 10.0


def test_kp_stroma_range(base_result):
    assert 0.1 <= base_result.kp_stroma <= 5.0


def test_higher_logP_higher_kp_epithelium():
    r1 = simulate_corneal_epithelium("A", 0.5, logP=0.5, mw_Da=300.0)
    r2 = simulate_corneal_epithelium("B", 0.5, logP=3.0, mw_Da=300.0)
    assert r2.kp_epithelium > r1.kp_epithelium


def test_higher_logP_lower_kp_stroma():
    r1 = simulate_corneal_epithelium("A", 0.5, logP=0.5, mw_Da=300.0)
    r2 = simulate_corneal_epithelium("B", 0.5, logP=3.0, mw_Da=300.0)
    assert r2.kp_stroma < r1.kp_stroma


def test_auc_precorneal_positive(base_result):
    assert base_result.auc_precorneal > 0


def test_auc_aqueous_positive(base_result):
    assert base_result.auc_aqueous > 0


def test_ocular_bioavailability_range(base_result):
    assert 0 <= base_result.ocular_bioavailability_pct <= 100


def test_dose_linearity_cmax():
    r1 = simulate_corneal_epithelium("A", dose_mg=0.5, logP=2.0, mw_Da=400.0)
    r2 = simulate_corneal_epithelium("A", dose_mg=1.0, logP=2.0, mw_Da=400.0)
    assert r2.cmax_precorneal_mg_mL == pytest.approx(2 * r1.cmax_precorneal_mg_mL, rel=0.05)
    assert r2.cmax_aqueous_mg_mL == pytest.approx(2 * r1.cmax_aqueous_mg_mL, rel=0.05)


def test_validation_negative_dose():
    with pytest.raises(ValueError):
        simulate_corneal_epithelium("X", dose_mg=-1, logP=1.0, mw_Da=300.0)


def test_validation_zero_dose():
    with pytest.raises(ValueError):
        simulate_corneal_epithelium("X", dose_mg=0, logP=1.0, mw_Da=300.0)


def test_validation_negative_mw():
    with pytest.raises(ValueError):
        simulate_corneal_epithelium("X", dose_mg=0.5, logP=1.0, mw_Da=-100)


def test_validation_zero_mw():
    with pytest.raises(ValueError):
        simulate_corneal_epithelium("X", dose_mg=0.5, logP=1.0, mw_Da=0)


def test_notes_is_string(base_result):
    assert isinstance(base_result.notes, str)
    assert len(base_result.notes) > 0


def test_negative_logP():
    r = simulate_corneal_epithelium("Neg", dose_mg=0.5, logP=-2.0, mw_Da=300.0)
    assert r.kp_epithelium >= 0.5
    assert r.kp_stroma >= 0.1
    assert r.cmax_aqueous_mg_mL > 0
