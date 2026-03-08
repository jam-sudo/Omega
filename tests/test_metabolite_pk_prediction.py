"""Tests for Phase 444 — metabolite PK prediction from parent drug properties."""

import pytest

from omega_pbpk.prediction.metabolite_pk_prediction import (
    TRANSFORMATIONS,
    MetabolitePKResult,
    predict_metabolite_panel,
    predict_metabolite_pk,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs):
    """Run prediction with midazolam-like defaults."""
    defaults = dict(
        drug_name="midazolam",
        metabolite_name="1-hydroxymidazolam",
        metabolite_type="hydroxylation",
        parent_mw_Da=325.8,
        parent_logp=3.89,
        parent_fu_plasma=0.03,
        parent_cl_L_per_h=25.0,
        parent_vd_L=70.0,
        fm=0.9,
    )
    defaults.update(kwargs)
    return predict_metabolite_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic fields
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default_result()
    assert isinstance(result, MetabolitePKResult)


def test_drug_name_preserved():
    result = _default_result(drug_name="midazolam")
    assert result.drug_name == "midazolam"


def test_metabolite_name_preserved():
    result = _default_result(metabolite_name="1-hydroxymidazolam")
    assert result.metabolite_name == "1-hydroxymidazolam"


def test_metabolite_type_preserved():
    result = _default_result(metabolite_type="hydroxylation")
    assert result.metabolite_type == "hydroxylation"


# ---------------------------------------------------------------------------
# MW and logP transformations
# ---------------------------------------------------------------------------


def test_hydroxylation_mw_delta():
    result = _default_result(metabolite_type="hydroxylation")
    expected_mw = 325.8 + 16
    assert result.metabolite_mw_Da == pytest.approx(expected_mw)


def test_glucuronidation_mw_delta():
    result = _default_result(metabolite_type="glucuronidation")
    expected_mw = 325.8 + 176
    assert result.metabolite_mw_Da == pytest.approx(expected_mw)


def test_n_demethylation_mw_delta():
    result = _default_result(metabolite_type="N_demethylation")
    expected_mw = 325.8 - 14
    assert result.metabolite_mw_Da == pytest.approx(expected_mw)


def test_hydroxylation_logp_delta():
    result = _default_result(metabolite_type="hydroxylation", parent_logp=3.89)
    expected_logp = 3.89 - 0.5
    assert result.metabolite_logp == pytest.approx(expected_logp, abs=1e-6)


def test_glucuronidation_logp_much_lower():
    """Glucuronidation adds MW 176 and reduces logP by 3.0 → more polar."""
    result_gluc = _default_result(metabolite_type="glucuronidation", parent_logp=3.89)
    result_hydrox = _default_result(metabolite_type="hydroxylation", parent_logp=3.89)
    assert result_gluc.metabolite_logp < result_hydrox.metabolite_logp


def test_logp_clamped_minimum():
    """Glucuronidation of very low logP parent should not go below -5."""
    result = _default_result(
        metabolite_type="glucuronidation",
        parent_logp=-3.0,
    )
    assert result.metabolite_logp >= -5.0


# ---------------------------------------------------------------------------
# fu_plasma transformations
# ---------------------------------------------------------------------------


def test_metabolite_fu_gte_parent_fu():
    """All transformations have fu_factor >= 1, so metabolite_fu >= parent_fu."""
    for met_type in TRANSFORMATIONS:
        result = _default_result(metabolite_type=met_type)
        assert result.metabolite_fu_plasma >= result.parent_fu_plasma - 1e-9, (
            f"Failed for {met_type}: metabolite_fu={result.metabolite_fu_plasma}, "
            f"parent_fu={result.parent_fu_plasma}"
        )


def test_metabolite_fu_capped_at_one():
    """fu_plasma cannot exceed 1.0."""
    result = _default_result(
        metabolite_type="glucuronidation",
        parent_fu_plasma=0.9,  # fu_factor=2.0 → would exceed 1.0
    )
    assert result.metabolite_fu_plasma <= 1.0


# ---------------------------------------------------------------------------
# PK predictions
# ---------------------------------------------------------------------------


def test_metabolite_t_half_positive():
    result = _default_result()
    assert result.metabolite_t_half_h > 0.0


def test_auc_ratio_positive():
    result = _default_result()
    assert result.auc_ratio_metabolite_to_parent > 0.0


def test_metabolite_cl_equals_parent_cl_times_factor():
    result = _default_result(metabolite_type="hydroxylation")
    expected_cl = 25.0 * TRANSFORMATIONS["hydroxylation"]["cl_factor"]
    assert result.metabolite_cl_L_per_h == pytest.approx(expected_cl)


def test_t_half_formula():
    """t_half = 0.693 * Vd / CL."""
    result = _default_result()
    expected = 0.693 * result.metabolite_vd_L / result.metabolite_cl_L_per_h
    assert result.metabolite_t_half_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Activity prediction
# ---------------------------------------------------------------------------


def test_activity_hydroxylation():
    result = _default_result(metabolite_type="hydroxylation")
    assert result.activity_prediction == "active"


def test_activity_glucuronidation_inactive():
    result = _default_result(metabolite_type="glucuronidation")
    assert result.activity_prediction == "inactive"


def test_activity_sulfation_inactive():
    result = _default_result(metabolite_type="sulfation")
    assert result.activity_prediction == "inactive"


def test_activity_hydrolysis_inactive():
    result = _default_result(metabolite_type="hydrolysis")
    assert result.activity_prediction == "inactive"


def test_activity_n_demethylation_active():
    result = _default_result(metabolite_type="N_demethylation")
    assert result.activity_prediction == "active"


# ---------------------------------------------------------------------------
# Excretion route
# ---------------------------------------------------------------------------


def test_glucuronide_biliary_excretion():
    """Glucuronide MW = 325.8 + 176 = 501.8 > 500 → biliary."""
    result = _default_result(
        metabolite_type="glucuronidation",
        parent_mw_Da=325.8,
    )
    assert result.excretion_route == "biliary"


def test_low_logp_renal_excretion():
    """Highly polar metabolite (logP < 0) → renal excretion."""
    result = _default_result(
        metabolite_type="glucuronidation",
        parent_mw_Da=200.0,  # MW after glucuronidation = 376 < 500
        parent_logp=-1.0,  # logP after glucuronidation = -4.0 < 0
    )
    # MW = 200+176=376 < 500, logP = -1-3=-4 < 0 → renal
    assert result.excretion_route == "renal"


# ---------------------------------------------------------------------------
# predict_metabolite_panel
# ---------------------------------------------------------------------------


def test_panel_returns_seven():
    panel = predict_metabolite_panel(
        drug_name="midazolam",
        parent_mw_Da=325.8,
        parent_logp=3.89,
        parent_fu_plasma=0.03,
        parent_cl_L_per_h=25.0,
        parent_vd_L=70.0,
    )
    assert len(panel) == 7


def test_panel_sorted_descending():
    panel = predict_metabolite_panel(
        drug_name="midazolam",
        parent_mw_Da=325.8,
        parent_logp=3.89,
        parent_fu_plasma=0.03,
        parent_cl_L_per_h=25.0,
        parent_vd_L=70.0,
    )
    ratios = [r.auc_ratio_metabolite_to_parent for r in panel]
    assert ratios == sorted(ratios, reverse=True)


def test_panel_all_types_covered():
    panel = predict_metabolite_panel(
        drug_name="midazolam",
        parent_mw_Da=325.8,
        parent_logp=3.89,
        parent_fu_plasma=0.03,
        parent_cl_L_per_h=25.0,
        parent_vd_L=70.0,
    )
    types_in_panel = {r.metabolite_type for r in panel}
    assert types_in_panel == set(TRANSFORMATIONS.keys())


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_metabolite_type_raises():
    with pytest.raises(ValueError, match="Unknown metabolite_type"):
        predict_metabolite_pk(
            drug_name="midazolam",
            metabolite_name="unknown_met",
            metabolite_type="acetylation",  # not in TRANSFORMATIONS
            parent_mw_Da=325.8,
            parent_logp=3.89,
            parent_fu_plasma=0.03,
            parent_cl_L_per_h=25.0,
            parent_vd_L=70.0,
        )


def test_negative_parent_mw_raises():
    with pytest.raises(ValueError):
        _default_result(parent_mw_Da=-100.0)


def test_fm_out_of_range_raises():
    with pytest.raises(ValueError):
        _default_result(fm=1.5)


def test_fm_zero_raises():
    with pytest.raises(ValueError):
        _default_result(fm=0.0)


def test_fu_out_of_range_raises():
    with pytest.raises(ValueError):
        _default_result(parent_fu_plasma=1.5)
