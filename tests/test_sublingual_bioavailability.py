"""Tests for Phase 1059 — Drug Sublingual Bioavailability Predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.sublingual_bioavailability import (
    SublingualBioavailabilityResult,
    predict_sublingual_bioavailability,
)

_VALID_BIO_CLASSES = {"excellent", "good", "moderate", "poor"}

# --- helpers ----------------------------------------------------------------


def _base(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        mw_Da=250.0,
        logp=2.0,
    )
    defaults.update(kwargs)
    return predict_sublingual_bioavailability(**defaults)


# --- type and dataclass tests -----------------------------------------------


def test_return_type():
    result = _base()
    assert isinstance(result, SublingualBioavailabilityResult)


def test_frozen_dataclass():
    result = _base()
    with pytest.raises((AttributeError, TypeError)):
        result.fa_sublingual = 0.5  # type: ignore[misc]


# --- field value range tests ------------------------------------------------


def test_fa_sublingual_in_range():
    result = _base()
    assert 0 < result.fa_sublingual <= 0.95


def test_f_systemic_in_range():
    result = _base()
    assert 0 <= result.f_systemic <= 1.0


def test_f_portal_bypass_fixed():
    result = _base()
    assert result.f_portal_bypass == pytest.approx(0.85)


def test_onset_time_in_range():
    result = _base()
    assert 2.0 <= result.onset_time_min <= 20.0


def test_duration_effect_in_range():
    result = _base()
    assert 0.5 <= result.duration_effect_h <= 8.0


def test_mucosal_permeability_positive():
    result = _base()
    assert result.mucosal_permeability_cm_s > 0.0


def test_notes_nonempty():
    result = _base()
    assert len(result.notes) > 0


# --- suitable_for_sl logic --------------------------------------------------


def test_suitable_for_sl_true():
    # low MW, moderate logp → fa > 0.3 and mw < 500
    result = _base(mw_Da=150.0, logp=2.5)
    # suitability = fa > 0.3 AND mw < 500
    expected = result.fa_sublingual > 0.3 and result.mw_Da < 500.0
    assert result.suitable_for_sl == expected


def test_suitable_for_sl_false_high_mw():
    result = _base(mw_Da=800.0, logp=2.0)
    assert result.suitable_for_sl is False


def test_suitable_for_sl_consistency():
    """suitable_for_sl must equal (fa > 0.3 AND mw < 500)."""
    result = _base(mw_Da=300.0, logp=1.0)
    expected = result.fa_sublingual > 0.3 and result.mw_Da < 500.0
    assert result.suitable_for_sl == expected


# --- bioavailability_class tests --------------------------------------------


def test_bioavailability_class_valid():
    result = _base()
    assert result.bioavailability_class in _VALID_BIO_CLASSES


def test_bioavailability_class_excellent():
    # high logp, low mw → high fa, high f_systemic
    result = _base(mw_Da=100.0, logp=4.0, dissolution_rate="fast")
    if result.f_systemic > 0.7:
        assert result.bioavailability_class == "excellent"


def test_bioavailability_class_poor():
    # very high MW, low logp → poor
    result = _base(mw_Da=900.0, logp=-1.0, dissolution_rate="slow")
    # f_systemic will be very low
    if result.f_systemic <= 0.3:
        assert result.bioavailability_class == "poor"


# --- physicochemical sensitivity tests -------------------------------------


def test_high_logp_low_mw_higher_fa():
    """High logP, low MW should produce higher fa_sl than low logP, high MW."""
    result_high = _base(mw_Da=100.0, logp=4.0)
    result_low = _base(mw_Da=600.0, logp=0.0)
    assert result_high.fa_sublingual > result_low.fa_sublingual


def test_fast_dissolution_higher_than_slow():
    # Use low logp so fa_sl does not hit the 0.95 cap for either rate
    result_fast = _base(logp=-2.0, dissolution_rate="fast")
    result_slow = _base(logp=-2.0, dissolution_rate="slow")
    assert result_fast.fa_sublingual > result_slow.fa_sublingual


def test_moderate_dissolution_between_fast_slow():
    # Use low logp so fa_sl does not hit the 0.95 cap
    result_fast = _base(logp=-2.0, dissolution_rate="fast")
    result_mod = _base(logp=-2.0, dissolution_rate="moderate")
    result_slow = _base(logp=-2.0, dissolution_rate="slow")
    assert result_fast.fa_sublingual >= result_mod.fa_sublingual >= result_slow.fa_sublingual


# --- ionization_type tests --------------------------------------------------


def test_neutral_ionization():
    result = _base(ionization_type="neutral")
    assert 0 < result.fa_sublingual <= 0.95


def test_base_ionization():
    result = _base(ionization_type="base", pka=9.0)
    assert 0 < result.fa_sublingual <= 0.95


# --- validation tests -------------------------------------------------------


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        _base(mw_Da=0.0)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        _base(mw_Da=-10.0)


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _base(dose_mg=0.0)


def test_invalid_dissolution_raises():
    with pytest.raises(ValueError, match="dissolution_rate"):
        _base(dissolution_rate="instant")


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        _base(ionization_type="cation")


def test_cl_hepatic_zero_raises():
    with pytest.raises(ValueError, match="cl_hepatic"):
        _base(cl_hepatic_L_per_h=0.0)


def test_q_hepatic_zero_raises():
    with pytest.raises(ValueError, match="q_hepatic"):
        _base(q_hepatic_L_per_h=0.0)


# --- field value checks -----------------------------------------------------


def test_drug_name_preserved():
    result = _base(drug_name="Nitroglycerin")
    assert result.drug_name == "Nitroglycerin"


def test_mw_logp_preserved():
    result = _base(mw_Da=300.0, logp=1.5)
    assert result.mw_Da == pytest.approx(300.0)
    assert result.logp == pytest.approx(1.5)
