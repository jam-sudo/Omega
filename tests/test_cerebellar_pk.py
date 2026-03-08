"""Tests for Phase 1051 — Drug Cerebellar Distribution."""

import pytest

from omega_pbpk.core.cerebellar_pk import CerebellarPKResult, simulate_cerebellar_pk


def _default() -> CerebellarPKResult:
    return simulate_cerebellar_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        logp=2.0,
        mw_Da=300.0,
        fu_plasma=0.3,
        cl_sys_L_per_h=5.0,
        vd_sys_L=30.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


# --- Return type ---
def test_returns_cerebellar_pk_result():
    result = _default()
    assert isinstance(result, CerebellarPKResult)


# --- Initial conditions ---
def test_plasma_initial_concentration():
    result = _default()
    expected = 100.0 / 30.0  # dose / vd
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-9


def test_cerebellum_initial_zero():
    result = _default()
    assert result.c_cerebellum_mg_L[0] == 0.0


def test_cortex_initial_zero():
    result = _default()
    assert result.c_cortex_mg_L[0] == 0.0


def test_csf_initial_zero():
    result = _default()
    assert result.c_csf_mg_L[0] == 0.0


# --- AUC positivity ---
def test_auc_plasma_positive():
    result = _default()
    assert result.auc_plasma > 0


def test_auc_cerebellum_positive():
    result = _default()
    assert result.auc_cerebellum > 0


def test_auc_cortex_positive():
    result = _default()
    assert result.auc_cortex > 0


# --- Cmax / Tmax ---
def test_cmax_cerebellum_positive():
    result = _default()
    assert result.cmax_cerebellum > 0


def test_tmax_cerebellum_nonnegative():
    result = _default()
    assert result.tmax_cerebellum_h >= 0


# --- Ratios ---
def test_cerebellum_to_plasma_ratio_positive():
    result = _default()
    assert result.cerebellum_to_plasma_ratio > 0


def test_cortex_to_cerebellum_ratio_positive():
    result = _default()
    assert result.cortex_to_cerebellum_ratio > 0


# --- logP effect on brain penetration ---
def test_high_logp_higher_cbt_ratio():
    high = simulate_cerebellar_pk("Drug", 100.0, logp=4.0, mw_Da=300.0, fu_plasma=0.5)
    low = simulate_cerebellar_pk("Drug", 100.0, logp=-1.0, mw_Da=300.0, fu_plasma=0.5)
    assert high.cerebellum_to_plasma_ratio > low.cerebellum_to_plasma_ratio


# --- Enrichment effect ---
def test_cerebellum_enrichment_increases_auc():
    enriched = simulate_cerebellar_pk(
        "Drug", 100.0, cerebellum_enrichment=2.0, cortex_enrichment=0.5
    )
    normal = simulate_cerebellar_pk("Drug", 100.0, cerebellum_enrichment=1.0, cortex_enrichment=0.5)
    assert enriched.auc_cerebellum > normal.auc_cerebellum


def test_cerebellum_enrichment_increases_ratio_vs_cortex():
    enriched = simulate_cerebellar_pk(
        "Drug", 100.0, cerebellum_enrichment=2.0, cortex_enrichment=0.8
    )
    # cortex_to_cerebellum_ratio should be lower when cerebellum is enriched more
    low_enrich = simulate_cerebellar_pk(
        "Drug", 100.0, cerebellum_enrichment=0.5, cortex_enrichment=2.0
    )
    assert enriched.cortex_to_cerebellum_ratio < low_enrich.cortex_to_cerebellum_ratio


# --- BBB penetration class ---
def test_bbb_class_valid_values():
    result = _default()
    assert result.bbb_penetration_class in {"high", "moderate", "low"}


def test_bbb_class_high_for_large_kp():
    result = simulate_cerebellar_pk("Drug", 100.0, kp_bbb=0.5)
    assert result.bbb_penetration_class == "high"


def test_bbb_class_low_for_small_kp():
    result = simulate_cerebellar_pk("Drug", 100.0, kp_bbb=0.05)
    assert result.bbb_penetration_class == "low"


def test_bbb_class_moderate():
    result = simulate_cerebellar_pk("Drug", 100.0, kp_bbb=0.2)
    assert result.bbb_penetration_class == "moderate"


# --- Notes ---
def test_notes_nonempty():
    result = _default()
    assert len(result.notes) > 0


# --- Validation errors ---
def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        simulate_cerebellar_pk("Drug", dose_mg=0.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError):
        simulate_cerebellar_pk("Drug", 100.0, cl_sys_L_per_h=0.0)


def test_raises_on_negative_kp():
    with pytest.raises(ValueError):
        simulate_cerebellar_pk("Drug", 100.0, kp_bbb=-0.1)


def test_raises_on_invalid_fu():
    with pytest.raises(ValueError):
        simulate_cerebellar_pk("Drug", 100.0, fu_plasma=0.0)
