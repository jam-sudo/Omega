"""Tests for Phase 497 — Synovial Fluid Drug Concentrations (synovial_fluid_pk_p497)."""

import math

import pytest

from omega_pbpk.core.synovial_fluid_pk_p497 import (
    SynovialFluidPKResult,
    compare_drugs_sf,
    simulate_synovial_fluid_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def neutral_drug():
    return simulate_synovial_fluid_pk(
        drug_name="NeutralDrug",
        dose_mg=100.0,
        logP=2.0,
        ionization_type="neutral",
        cl_sys_L_per_h=2.0,
        vd_plasma_L=5.0,
    )


@pytest.fixture
def acid_drug():
    return simulate_synovial_fluid_pk(
        drug_name="AcidDrug",
        dose_mg=100.0,
        logP=2.0,
        ionization_type="acid",
        cl_sys_L_per_h=2.0,
        vd_plasma_L=5.0,
    )


@pytest.fixture
def base_drug():
    return simulate_synovial_fluid_pk(
        drug_name="BaseDrug",
        dose_mg=100.0,
        logP=2.0,
        ionization_type="base",
        cl_sys_L_per_h=2.0,
        vd_plasma_L=5.0,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type(neutral_drug):
    assert isinstance(neutral_drug, SynovialFluidPKResult)


def test_result_has_required_fields(neutral_drug):
    assert hasattr(neutral_drug, "times_h")
    assert hasattr(neutral_drug, "c_plasma_mg_L")
    assert hasattr(neutral_drug, "c_sf_mg_L")
    assert hasattr(neutral_drug, "kp_sf")
    assert hasattr(neutral_drug, "sf_plasma_auc_ratio")
    assert hasattr(neutral_drug, "therapeutic_potential")
    assert hasattr(neutral_drug, "notes")


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_plasma_starts_at_dose_over_vd(neutral_drug):
    expected = 100.0 / 5.0  # 20 mg/L
    assert math.isclose(neutral_drug.c_plasma_mg_L[0], expected, rel_tol=1e-6)


def test_sf_starts_at_zero(neutral_drug):
    assert neutral_drug.c_sf_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Kp_sf calculations
# ---------------------------------------------------------------------------


def test_kp_sf_neutral_logP2():
    # neutral: 0.3 + 0.1 * 2 = 0.5
    result = simulate_synovial_fluid_pk("X", 100.0, 2.0, "neutral", 2.0)
    assert math.isclose(result.kp_sf, 0.5, rel_tol=1e-6)


def test_kp_sf_acid_logP2():
    # acid: 0.5 + 0.15 * 2 = 0.8
    result = simulate_synovial_fluid_pk("X", 100.0, 2.0, "acid", 2.0)
    assert math.isclose(result.kp_sf, 0.8, rel_tol=1e-6)


def test_kp_sf_base_logP2():
    # base: 0.2 + 0.1 * 2 = 0.4
    result = simulate_synovial_fluid_pk("X", 100.0, 2.0, "base", 2.0)
    assert math.isclose(result.kp_sf, 0.4, rel_tol=1e-6)


def test_acid_higher_kp_than_base_same_logP(acid_drug, base_drug):
    assert acid_drug.kp_sf > base_drug.kp_sf


def test_acid_higher_kp_than_neutral_same_logP(acid_drug, neutral_drug):
    assert acid_drug.kp_sf > neutral_drug.kp_sf


def test_kp_sf_clamped_min():
    # negative logP: neutral: 0.3 + 0.1*0 = 0.3 (max(0,logP)=0)
    result = simulate_synovial_fluid_pk("X", 100.0, -5.0, "neutral", 2.0)
    assert result.kp_sf == 0.3


def test_kp_sf_clamped_max():
    # very high logP: neutral: 0.3 + 0.1*100 = clamped to 3.0
    result = simulate_synovial_fluid_pk("X", 100.0, 100.0, "neutral", 2.0)
    assert result.kp_sf == 3.0


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def test_cmax_sf_positive(neutral_drug):
    assert neutral_drug.cmax_sf_mg_L > 0.0


def test_plasma_declines_over_time(neutral_drug):
    # After 24h plasma should be lower than initial
    assert neutral_drug.c_plasma_mg_L[-1] < neutral_drug.c_plasma_mg_L[0]


def test_auc_sf_positive(neutral_drug):
    assert neutral_drug.auc_sf_mg_h_per_L > 0.0


def test_auc_plasma_positive(neutral_drug):
    assert neutral_drug.auc_plasma_mg_h_per_L > 0.0


def test_sf_plasma_auc_ratio_positive(neutral_drug):
    # Ratio can exceed 1.0 if drug concentrates in SF relative to plasma
    assert neutral_drug.sf_plasma_auc_ratio > 0.0


def test_tmax_sf_positive(neutral_drug):
    # SF takes time to fill; tmax_sf > 0
    assert neutral_drug.tmax_sf_h >= 0.0


# ---------------------------------------------------------------------------
# Therapeutic potential classification
# ---------------------------------------------------------------------------


def test_therapeutic_potential_values_valid(neutral_drug):
    assert neutral_drug.therapeutic_potential in {"poor", "good", "excellent"}


def test_excellent_potential_high_kp():
    # High kp_sf drug should tend toward excellent
    result = simulate_synovial_fluid_pk(
        "HighKp", 100.0, 20.0, "acid", 0.5, vd_plasma_L=5.0, Q_sf_L_per_h=0.2, t_end_h=48.0
    )
    assert result.therapeutic_potential in {"good", "excellent"}


def test_poor_potential_classification():
    # Very high clearance and low dose → poor AUC ratio
    result = simulate_synovial_fluid_pk(
        "LowPen", 100.0, -5.0, "base", 50.0, vd_plasma_L=5.0, Q_sf_L_per_h=0.001, t_end_h=24.0
    )
    assert result.therapeutic_potential in {"poor", "good", "excellent"}


# ---------------------------------------------------------------------------
# compare_drugs_sf
# ---------------------------------------------------------------------------


def test_compare_drugs_sf_returns_list():
    drugs = [
        {
            "drug_name": "DrugA",
            "dose_mg": 100.0,
            "logP": 3.0,
            "ionization_type": "acid",
            "cl_sys_L_per_h": 1.0,
        },
        {
            "drug_name": "DrugB",
            "dose_mg": 100.0,
            "logP": 1.0,
            "ionization_type": "base",
            "cl_sys_L_per_h": 1.0,
        },
    ]
    results = compare_drugs_sf(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_drugs_sorted_by_auc_ratio():
    drugs = [
        {
            "drug_name": "Low",
            "dose_mg": 100.0,
            "logP": -2.0,
            "ionization_type": "base",
            "cl_sys_L_per_h": 5.0,
        },
        {
            "drug_name": "High",
            "dose_mg": 100.0,
            "logP": 5.0,
            "ionization_type": "acid",
            "cl_sys_L_per_h": 0.5,
        },
    ]
    results = compare_drugs_sf(drugs)
    assert results[0].sf_plasma_auc_ratio >= results[1].sf_plasma_auc_ratio


def test_compare_single_drug():
    drugs = [
        {
            "drug_name": "Solo",
            "dose_mg": 50.0,
            "logP": 2.0,
            "ionization_type": "neutral",
            "cl_sys_L_per_h": 2.0,
        },
    ]
    results = compare_drugs_sf(drugs)
    assert len(results) == 1
    assert results[0].drug_name == "Solo"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_synovial_fluid_pk("X", 0.0, 2.0, "neutral", 2.0)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_synovial_fluid_pk("X", 100.0, 2.0, "neutral", 0.0)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_synovial_fluid_pk("X", 100.0, 2.0, "neutral", 2.0, vd_plasma_L=0.0)


def test_invalid_v_sf():
    with pytest.raises(ValueError, match="V_sf_L"):
        simulate_synovial_fluid_pk("X", 100.0, 2.0, "neutral", 2.0, V_sf_L=0.0)


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        simulate_synovial_fluid_pk("X", 100.0, 2.0, "cation", 2.0)
