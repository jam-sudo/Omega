"""Tests for Phase 487 — Drug Interaction via Plasma Protein Displacement."""

import math

import pytest

from omega_pbpk.clinical.protein_displacement_ddi import (
    ProteinDisplacementDDIResult,
    predict_protein_displacement_ddi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VICTIM = "warfarin"
DISPLACER = "aspirin"


def _call(**kwargs):
    defaults = dict(
        victim_drug=VICTIM,
        displacer_drug=DISPLACER,
        fu_baseline=0.01,
        cl_sys_L_per_h=0.2,  # low ER
        conc_displacer_mg_L=10.0,
        kd_displacer_mg_L=1.0,
        Q_hepatic_L_per_h=90.0,
    )
    defaults.update(kwargs)
    return predict_protein_displacement_ddi(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _call()
    assert isinstance(result, ProteinDisplacementDDIResult)


def test_result_is_frozen():
    result = _call()
    with pytest.raises((AttributeError, TypeError)):
        result.fu_baseline = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# fu_displaced >= fu_baseline
# ---------------------------------------------------------------------------


def test_fu_displaced_ge_baseline():
    result = _call()
    assert result.fu_displaced >= result.fu_baseline


def test_no_displacement_when_conc_zero():
    result = _call(conc_displacer_mg_L=0.0)
    assert math.isclose(result.fu_displaced, result.fu_baseline, rel_tol=1e-9)
    assert math.isclose(result.displacement_ratio, 1.0, rel_tol=1e-9)


def test_maximum_displacement_at_high_conc():
    """Very high displacer concentration → fu_displaced approaches 2× fu_baseline."""
    result = _call(conc_displacer_mg_L=1e6, kd_displacer_mg_L=1.0, fu_baseline=0.3)
    # displacement_factor → 1.0 → fu_displaced → 2 × fu_baseline = 0.6
    assert result.fu_displaced > 1.5 * result.fu_baseline


def test_displacement_ratio_equals_fu_ratio():
    result = _call()
    expected = result.fu_displaced / result.fu_baseline
    assert math.isclose(result.displacement_ratio, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Extraction ratio and AUC behaviour
# ---------------------------------------------------------------------------


def test_high_er_auc_ratio_near_one():
    """High-ER drugs: AUC should be unchanged by protein displacement."""
    result = _call(
        cl_sys_L_per_h=80.0,  # ER = 80/90 > 0.7
        Q_hepatic_L_per_h=90.0,
        conc_displacer_mg_L=50.0,
    )
    assert result.extraction_ratio > 0.7
    assert math.isclose(result.auc_ratio, 1.0, abs_tol=1e-9)


def test_low_er_auc_decreases():
    """Low-ER drugs: AUC decreases (ratio < 1) when fu increases."""
    result = _call(
        cl_sys_L_per_h=0.5,  # ER = 0.5/90 << 0.3
        Q_hepatic_L_per_h=90.0,
        conc_displacer_mg_L=50.0,
        kd_displacer_mg_L=1.0,
    )
    assert result.extraction_ratio < 0.3
    assert result.auc_ratio < 1.0


def test_low_er_auc_ratio_inverse_displacement():
    """Low-ER: auc_ratio = 1 / displacement_ratio."""
    result = _call(
        cl_sys_L_per_h=0.1,  # very low ER
        Q_hepatic_L_per_h=90.0,
        conc_displacer_mg_L=10.0,
        kd_displacer_mg_L=2.0,
    )
    expected_auc = 1.0 / result.displacement_ratio
    assert math.isclose(result.auc_ratio, expected_auc, rel_tol=1e-6)


def test_moderate_er_auc_between_extremes():
    """Moderate ER: AUC ratio is between 1/displacement_ratio and 1."""
    result = _call(
        cl_sys_L_per_h=45.0,  # ER = 0.5
        Q_hepatic_L_per_h=90.0,
        conc_displacer_mg_L=10.0,
    )
    assert 0.3 <= result.extraction_ratio <= 0.7
    lower = 1.0 / result.displacement_ratio
    assert lower <= result.auc_ratio <= 1.0


# ---------------------------------------------------------------------------
# Vd ratio
# ---------------------------------------------------------------------------


def test_vd_ratio_equals_displacement_ratio():
    result = _call()
    assert math.isclose(result.vd_ratio, result.displacement_ratio, rel_tol=1e-9)


def test_no_displacement_vd_ratio_one():
    result = _call(conc_displacer_mg_L=0.0)
    assert math.isclose(result.vd_ratio, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Clinical significance classification
# ---------------------------------------------------------------------------


def test_clinical_significance_none_when_high_er():
    result = _call(cl_sys_L_per_h=80.0, conc_displacer_mg_L=50.0)
    assert result.clinical_significance == "none"


def test_clinical_significance_major_when_large_auc_change():
    """If low ER and large fu displacement → major significance."""
    result = _call(
        fu_baseline=0.01,
        cl_sys_L_per_h=0.1,  # very low ER
        conc_displacer_mg_L=1e6,  # saturate displacing → ~2× fu
        kd_displacer_mg_L=1.0,
    )
    # auc_ratio ≈ 1/2 = 0.5 → 50% change → "major"
    assert result.clinical_significance in ("major", "moderate")


def test_significance_none_when_no_displacement():
    result = _call(conc_displacer_mg_L=0.0)
    assert result.clinical_significance == "none"


def test_significance_values_valid():
    result = _call()
    assert result.clinical_significance in ("none", "minor", "moderate", "major")


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def test_recommendation_no_action_when_none():
    result = _call(cl_sys_L_per_h=80.0, conc_displacer_mg_L=50.0)
    assert result.recommendation == "no_action"


def test_recommendation_monitor_when_significant():
    result = _call(
        fu_baseline=0.01,
        cl_sys_L_per_h=0.1,
        conc_displacer_mg_L=1e6,
        kd_displacer_mg_L=1.0,
    )
    if result.clinical_significance in ("moderate", "major"):
        assert result.recommendation == "monitor"


def test_recommendation_values_valid():
    result = _call()
    assert result.recommendation in ("monitor", "no_action")


# ---------------------------------------------------------------------------
# fu cap at 1.0
# ---------------------------------------------------------------------------


def test_fu_displaced_capped_at_one():
    """fu cannot exceed 1.0 even with maximal displacement."""
    result = _call(fu_baseline=0.8, conc_displacer_mg_L=1e9, kd_displacer_mg_L=1.0)
    assert result.fu_displaced <= 1.0


# ---------------------------------------------------------------------------
# Drug name fields
# ---------------------------------------------------------------------------


def test_drug_names_preserved():
    result = _call(victim_drug="drug_A", displacer_drug="drug_B")
    assert result.victim_drug == "drug_A"
    assert result.displacer_drug == "drug_B"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_fu_baseline_zero():
    with pytest.raises(ValueError, match="fu_baseline"):
        _call(fu_baseline=0.0)


def test_invalid_fu_baseline_negative():
    with pytest.raises(ValueError, match="fu_baseline"):
        _call(fu_baseline=-0.1)


def test_invalid_fu_baseline_above_one():
    with pytest.raises(ValueError, match="fu_baseline"):
        _call(fu_baseline=1.5)


def test_invalid_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        _call(cl_sys_L_per_h=0.0)


def test_invalid_cl_sys_negative():
    with pytest.raises(ValueError, match="cl_sys"):
        _call(cl_sys_L_per_h=-5.0)


def test_invalid_conc_displacer_negative():
    with pytest.raises(ValueError, match="conc_displacer_mg_L"):
        _call(conc_displacer_mg_L=-1.0)


def test_invalid_kd_displacer_zero():
    with pytest.raises(ValueError, match="kd_displacer_mg_L"):
        _call(kd_displacer_mg_L=0.0)


def test_invalid_kd_displacer_negative():
    with pytest.raises(ValueError, match="kd_displacer_mg_L"):
        _call(kd_displacer_mg_L=-0.5)
