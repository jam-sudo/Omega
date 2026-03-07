"""Tests for Phase 639 — core/intracellular_pk.py."""

import pytest

from omega_pbpk.core.intracellular_pk import (
    IntracellularPKResult,
    predict_lysosomal_trapping,
    simulate_intracellular_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_result():
    """Standard basic drug simulation: pKa=8, chloroquine-like."""
    return simulate_intracellular_pk(
        drug_name="TestBase",
        dose_mg=100.0,
        pka_base=8.0,
        logP=3.0,
        vd_plasma_L=50.0,
        cl_L_per_h=5.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


@pytest.fixture
def neutral_result():
    """Neutral drug: pKa=2 (no significant trapping)."""
    return simulate_intracellular_pk(
        drug_name="TestNeutral",
        dose_mg=100.0,
        pka_base=2.0,
        logP=1.0,
        vd_plasma_L=50.0,
        cl_L_per_h=5.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type and field types
# ---------------------------------------------------------------------------


def test_returns_intracellular_pk_result(basic_result):
    assert isinstance(basic_result, IntracellularPKResult)


def test_field_types(basic_result):
    assert isinstance(basic_result.drug_name, str)
    assert isinstance(basic_result.pka_base, float)
    assert isinstance(basic_result.dose_mg, float)
    assert isinstance(basic_result.times_h, list)
    assert isinstance(basic_result.c_plasma_mg_L, list)
    assert isinstance(basic_result.c_cytosol_mg_L, list)
    assert isinstance(basic_result.c_lysosome_mg_L, list)
    assert isinstance(basic_result.cmax_plasma, float)
    assert isinstance(basic_result.cmax_cytosol, float)
    assert isinstance(basic_result.cmax_lysosome, float)
    assert isinstance(basic_result.auc_plasma, float)
    assert isinstance(basic_result.auc_lysosome, float)
    assert isinstance(basic_result.lysosomal_trap_ratio, float)
    assert isinstance(basic_result.accumulation_index, float)
    assert isinstance(basic_result.t_half_plasma_h, float)
    assert isinstance(basic_result.notes, str)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive(basic_result):
    assert basic_result.cmax_plasma > 0.0


def test_cmax_lysosome_greater_than_cytosol_for_basic_drug(basic_result):
    """Lysosomal trapping: basic drug should accumulate more in lysosomes than cytosol."""
    assert basic_result.cmax_lysosome > basic_result.cmax_cytosol


def test_lysosomal_trap_ratio_gt_1_for_basic_drug(basic_result):
    """pKa=8, pH_lyso=5 → trap_ratio >> 1."""
    assert basic_result.lysosomal_trap_ratio > 1.0


def test_lysosomal_trap_ratio_near_1_for_neutral_drug(neutral_result):
    """pKa=2, pH_lyso=5 → minimal trapping; trap_ratio close to 1."""
    # For pKa=2, pH_lyso=5: frac_ionized_lyso ≈ 0, frac_ionized_cyto ≈ 0 → ratio ≈ 1
    assert neutral_result.lysosomal_trap_ratio < 10.0


def test_higher_pka_higher_trap_ratio():
    """Higher pKa → higher lysosomal trap ratio."""
    r1 = simulate_intracellular_pk("Drug1", dose_mg=100.0, pka_base=7.0, t_end_h=24.0, dt_h=0.1)
    r2 = simulate_intracellular_pk("Drug2", dose_mg=100.0, pka_base=9.0, t_end_h=24.0, dt_h=0.1)
    assert r2.lysosomal_trap_ratio > r1.lysosomal_trap_ratio


def test_auc_lysosome_positive(basic_result):
    assert basic_result.auc_lysosome > 0.0


def test_accumulation_index_ge_1_for_highly_basic_drug():
    """pKa > 7 → lysosomal concentration should exceed plasma."""
    result = simulate_intracellular_pk(
        "HighlyBasic",
        dose_mg=100.0,
        pka_base=9.0,
        logP=3.0,
        vd_plasma_L=50.0,
        cl_L_per_h=2.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    assert result.accumulation_index >= 1.0


def test_t_half_plasma_positive(basic_result):
    assert basic_result.t_half_plasma_h > 0.0


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax_plasma():
    r1 = simulate_intracellular_pk("Drug", dose_mg=100.0, pka_base=8.0, t_end_h=24.0, dt_h=0.1)
    r2 = simulate_intracellular_pk("Drug", dose_mg=200.0, pka_base=8.0, t_end_h=24.0, dt_h=0.1)
    assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.01


def test_double_dose_doubles_cmax_lysosome():
    r1 = simulate_intracellular_pk("Drug", dose_mg=100.0, pka_base=8.0, t_end_h=24.0, dt_h=0.1)
    r2 = simulate_intracellular_pk("Drug", dose_mg=200.0, pka_base=8.0, t_end_h=24.0, dt_h=0.1)
    assert abs(r2.cmax_lysosome / r1.cmax_lysosome - 2.0) < 0.05


# ---------------------------------------------------------------------------
# predict_lysosomal_trapping
# ---------------------------------------------------------------------------


def test_predict_returns_dict():
    result = predict_lysosomal_trapping(pka_base=8.0)
    assert isinstance(result, dict)


def test_predict_has_required_keys():
    result = predict_lysosomal_trapping(pka_base=8.0)
    assert "trap_ratio" in result
    assert "fold_accumulation" in result
    assert "risk_category" in result


def test_predict_trap_ratio_gt_1000_for_high_pka():
    """pKa=10: trap_ratio should exceed 1000."""
    result = predict_lysosomal_trapping(pka_base=10.0)
    assert result["trap_ratio"] > 1000.0


def test_predict_risk_category_high_for_pka10():
    result = predict_lysosomal_trapping(pka_base=10.0)
    assert result["risk_category"] == "high"


def test_predict_risk_category_low_for_pka2():
    result = predict_lysosomal_trapping(pka_base=2.0)
    assert result["risk_category"] == "low"


# ---------------------------------------------------------------------------
# List length consistency
# ---------------------------------------------------------------------------


def test_list_lengths_equal(basic_result):
    assert len(basic_result.times_h) == len(basic_result.c_plasma_mg_L)
    assert len(basic_result.times_h) == len(basic_result.c_lysosome_mg_L)
    assert len(basic_result.times_h) == len(basic_result.c_cytosol_mg_L)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_intracellular_pk("Drug", dose_mg=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError):
        simulate_intracellular_pk("Drug", dose_mg=100.0, vd_plasma_L=0.0)


def test_validation_f_lysosome_zero():
    with pytest.raises(ValueError):
        simulate_intracellular_pk("Drug", dose_mg=100.0, f_lysosome_vol=0.0)


def test_validation_ph_lysosome_too_low():
    with pytest.raises(ValueError):
        simulate_intracellular_pk("Drug", dose_mg=100.0, ph_lysosome=3.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty(basic_result):
    assert len(basic_result.notes) > 0
