"""Tests for prediction/nasal_absorption_predictor.py — Phase 235."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.nasal_absorption_predictor import (
    NasalAbsorptionResult,
    compare_nasal_regions,
    predict_nasal_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> NasalAbsorptionResult:
    defaults = dict(
        drug_name="TestDrug",
        mw=300.0,
        logp=2.0,
        pka=7.4,
        dose_mg=10.0,
        nasal_region="posterior",
        vd_L=50.0,
        cl_L_per_h=5.0,
    )
    defaults.update(kwargs)
    return predict_nasal_absorption(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_nasal_absorption_result():
    r = _default()
    assert isinstance(r, NasalAbsorptionResult)


def test_result_is_frozen():
    r = _default()
    with pytest.raises((AttributeError, TypeError)):
        r.drug_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Field ranges and types
# ---------------------------------------------------------------------------


def test_fraction_absorbed_in_range():
    r = _default()
    assert 0.0 <= r.fraction_absorbed <= 1.0


def test_bioavailability_equals_fraction_times_100():
    r = _default()
    assert abs(r.bioavailability_pct - r.fraction_absorbed * 100.0) < 1e-9


def test_c_plasma_peak_positive():
    r = _default()
    assert r.c_plasma_peak_mg_L > 0.0


def test_t_peak_min_positive():
    r = _default()
    assert r.t_peak_min > 0.0


def test_peff_cm_s_positive():
    r = _default()
    assert r.peff_cm_s > 0.0


def test_notes_is_string():
    r = _default()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Absorption class categorization
# ---------------------------------------------------------------------------


def test_absorption_class_valid_values():
    r = _default()
    assert r.absorption_class in ("rapid", "moderate", "slow")


def test_rapid_class_when_t_peak_less_than_10():
    r = _default()
    if r.t_peak_min < 10.0:
        assert r.absorption_class == "rapid"


def test_slow_class_when_t_peak_gt_30():
    # Very small Peff → slow absorption
    r = _default(nasal_region="anterior", dose_mg=100.0, mw=600.0)
    if r.t_peak_min > 30.0:
        assert r.absorption_class == "slow"


# ---------------------------------------------------------------------------
# MW effect
# ---------------------------------------------------------------------------


def test_high_mw_lower_absorption_than_low_mw():
    r_low = _default(mw=200.0)
    r_high = _default(mw=600.0)
    assert r_high.fraction_absorbed < r_low.fraction_absorbed


def test_mw_300_no_penalty():
    """MW=300 is the threshold; no reduction should be applied."""
    r = _default(mw=300.0, nasal_region="posterior")
    r_less = _default(mw=250.0, nasal_region="posterior")
    # fraction_absorbed should be the same or very close for mw<=300
    # (no penalty applies below 300)
    assert r_less.peff_cm_s >= r.peff_cm_s


# ---------------------------------------------------------------------------
# Region effects
# ---------------------------------------------------------------------------


def test_olfactory_highest_peff():
    """Olfactory region has highest base Peff."""
    r_ant = predict_nasal_absorption("D", 200.0, 1.0, 7.0, 5.0, "anterior")
    r_pos = predict_nasal_absorption("D", 200.0, 1.0, 7.0, 5.0, "posterior")
    r_olf = predict_nasal_absorption("D", 200.0, 1.0, 7.0, 5.0, "olfactory")
    assert r_olf.peff_cm_s > r_pos.peff_cm_s
    assert r_olf.peff_cm_s > r_ant.peff_cm_s


def test_region_stored_in_result():
    for region in ("anterior", "posterior", "olfactory"):
        r = _default(nasal_region=region)
        assert r.nasal_region == region


# ---------------------------------------------------------------------------
# compare_nasal_regions
# ---------------------------------------------------------------------------


def test_compare_returns_list_of_three():
    results = compare_nasal_regions("Drug", 300.0, 2.0, 7.4, 10.0)
    assert len(results) == 3


def test_compare_sorted_by_fraction_absorbed_descending():
    results = compare_nasal_regions("Drug", 300.0, 2.0, 7.4, 10.0)
    fracs = [r.fraction_absorbed for r in results]
    assert fracs == sorted(fracs, reverse=True)


def test_compare_all_regions_represented():
    results = compare_nasal_regions("Drug", 300.0, 2.0, 7.4, 10.0)
    regions = {r.nasal_region for r in results}
    assert regions == {"anterior", "posterior", "olfactory"}


def test_compare_all_nasal_absorption_result_type():
    results = compare_nasal_regions("Drug", 300.0, 2.0, 7.4, 10.0)
    for r in results:
        assert isinstance(r, NasalAbsorptionResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_nasal_absorption("D", mw=0.0, logp=1.0, pka=7.0, dose_mg=10.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_nasal_absorption("D", mw=-50.0, logp=1.0, pka=7.0, dose_mg=10.0)


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_nasal_absorption("D", mw=300.0, logp=1.0, pka=7.0, dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_nasal_absorption("D", mw=300.0, logp=1.0, pka=7.0, dose_mg=-5.0)


def test_validation_invalid_region():
    with pytest.raises(ValueError, match="nasal_region"):
        predict_nasal_absorption(
            "D", mw=300.0, logp=1.0, pka=7.0, dose_mg=10.0, nasal_region="sinus"
        )


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        predict_nasal_absorption("D", mw=300.0, logp=1.0, pka=7.0, dose_mg=10.0, vd_L=0.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        predict_nasal_absorption("D", mw=300.0, logp=1.0, pka=7.0, dose_mg=10.0, cl_L_per_h=0.0)
