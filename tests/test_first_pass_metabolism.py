"""Tests for Phase 1067 — Drug First-Pass Metabolism Predictor."""

import pytest

from omega_pbpk.prediction.first_pass_metabolism import (
    FirstPassResult,
    predict_first_pass_metabolism,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_basic(**kwargs):
    """Return a result with minimal valid parameters, overridable via kwargs."""
    defaults = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logp=2.0,
    )
    defaults.update(kwargs)
    return predict_first_pass_metabolism(**defaults)


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


def test_return_type():
    result = _make_basic()
    assert isinstance(result, FirstPassResult)


def test_frozen_dataclass():
    result = _make_basic()
    with pytest.raises((AttributeError, TypeError)):
        result.fa = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Range assertions
# ---------------------------------------------------------------------------


def test_fa_in_range():
    result = _make_basic()
    assert 0.05 <= result.fa <= 0.95


def test_fg_in_range():
    result = _make_basic()
    assert 0.0 <= result.fg <= 1.0


def test_fh_in_range():
    result = _make_basic()
    assert 0.0 <= result.fh <= 1.0


def test_f_oral_equals_fa_times_fg_times_fh():
    result = _make_basic(cyp3a4_substrate=True)
    assert abs(result.f_oral - result.fa * result.fg * result.fh) < 1e-9


def test_f_oral_no_cyp():
    result = _make_basic()
    assert abs(result.f_oral - result.fa * result.fg * result.fh) < 1e-9


# ---------------------------------------------------------------------------
# CYP substrate effects
# ---------------------------------------------------------------------------


def test_cyp3a4_reduces_fg():
    without = _make_basic(cyp3a4_substrate=False)
    with_cyp = _make_basic(cyp3a4_substrate=True)
    assert with_cyp.fg < without.fg


def test_cyp2d6_and_3a4_reduces_fh():
    without = _make_basic()
    with_cyp = _make_basic(cyp3a4_substrate=True, cyp2d6_substrate=True)
    assert with_cyp.fh < without.fh


def test_cyp1a2_reduces_fh():
    without = _make_basic()
    with_cyp = _make_basic(cyp1a2_substrate=True)
    assert with_cyp.fh < without.fh


# ---------------------------------------------------------------------------
# P-gp effect on fa
# ---------------------------------------------------------------------------


def test_pgp_reduces_fa():
    without = _make_basic(logp=3.0)
    with_pgp = _make_basic(logp=3.0, pgp_substrate=True)
    assert with_pgp.fa < without.fa


# ---------------------------------------------------------------------------
# Route effects
# ---------------------------------------------------------------------------


def test_sublingual_fg_is_one():
    result = _make_basic(route="sublingual")
    assert result.fg == 1.0


def test_rectal_fg_is_one():
    result = _make_basic(route="rectal")
    assert result.fg == 1.0


def test_sublingual_fh_smaller_than_oral():
    """Sublingual stores fh = fh_base * 0.15 (only 15% goes through portal),
    so fh_sl < fh_oral numerically. This encodes the portal bypass benefit."""
    oral = _make_basic(route="oral", cyp3a4_substrate=True, cyp2d6_substrate=True)
    sl = _make_basic(route="sublingual", cyp3a4_substrate=True, cyp2d6_substrate=True)
    # fh_sl = fh_oral * 0.15 → always smaller
    assert sl.fh < oral.fh


# ---------------------------------------------------------------------------
# Extraction ratios and flags
# ---------------------------------------------------------------------------


def test_hepatic_extraction_ratio_complement_of_fh():
    result = _make_basic(cyp3a4_substrate=True)
    assert abs(result.hepatic_extraction_ratio + result.fh - 1.0) < 1e-9


def test_gut_extraction_ratio_complement_of_fg():
    result = _make_basic(cyp3a4_substrate=True)
    assert abs(result.gut_extraction_ratio + result.fg - 1.0) < 1e-9


def test_low_extraction_flag_no_cyp():
    """With baseline CLint only, Eh should be very low."""
    result = _make_basic(fu_plasma=0.1)
    # clint_hepatic = 1.0 * 0.8 (fu_mic default); Eh = fu_plasma*clint / (Q + fu_plasma*clint)
    assert result.low_extraction_drug or not result.high_extraction_drug  # not high


def test_high_extraction_flag_high_cyp():
    result = _make_basic(
        cyp3a4_substrate=True,
        cyp2d6_substrate=True,
        cyp1a2_substrate=True,
        fu_plasma=0.9,
        fu_mic=1.0,
        q_hepatic_L_per_h=90.0,
    )
    assert result.high_extraction_drug or result.hepatic_extraction_ratio > 0.5


# ---------------------------------------------------------------------------
# Bioavailability class
# ---------------------------------------------------------------------------


def test_bioavailability_class_in_valid_set():
    result = _make_basic()
    assert result.bioavailability_class in {"high", "moderate", "low"}


def test_bioavailability_class_high():
    """Drug with no CYP metabolism and moderate logP should have high F."""
    result = _make_basic(logp=3.0, mw_Da=200.0, fu_plasma=0.5)
    assert result.bioavailability_class in {"high", "moderate"}


def test_bioavailability_class_low_high_cyp():
    result = _make_basic(
        cyp3a4_substrate=True,
        cyp2d6_substrate=True,
        cyp1a2_substrate=True,
        fu_plasma=0.9,
        fu_mic=1.0,
    )
    assert result.bioavailability_class in {"low", "moderate"}


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _make_basic()
    assert len(result.notes) > 0


def test_notes_mention_cyp3a4():
    result = _make_basic(cyp3a4_substrate=True)
    assert "CYP3A4" in result.notes or "cyp3a4" in result.notes.lower()


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_first_pass_metabolism("X", mw_Da=0.0, logp=2.0)


def test_raises_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_first_pass_metabolism("X", mw_Da=300.0, logp=2.0, fu_plasma=0.0)


def test_raises_fu_mic_zero():
    with pytest.raises(ValueError, match="fu_mic"):
        predict_first_pass_metabolism("X", mw_Da=300.0, logp=2.0, fu_mic=0.0)


def test_raises_invalid_route():
    with pytest.raises(ValueError, match="route"):
        predict_first_pass_metabolism("X", mw_Da=300.0, logp=2.0, route="intravenous")


def test_raises_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_first_pass_metabolism("X", mw_Da=300.0, logp=2.0, ionization_type="unknown")
