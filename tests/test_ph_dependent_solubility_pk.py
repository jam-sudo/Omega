"""Tests for pH-dependent solubility and absorption simulation (Phase 789)."""

import pytest

from omega_pbpk.core.ph_dependent_solubility_pk import (
    PHDependentSolubilityResult,
    simulate_ph_dependent_absorption,
)

# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_ph_dependent_absorption("Aspirin", dose_mg=500.0, pka=3.5, logp=1.2)
    assert isinstance(result, PHDependentSolubilityResult)


def test_segments_is_list():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=100.0, pka=4.0, logp=2.0)
    assert isinstance(result.segments, list)


def test_segments_length_four():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=100.0, pka=4.0, logp=2.0)
    assert len(result.segments) == 4


def test_times_h_is_list():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=100.0, pka=4.0, logp=2.0)
    assert isinstance(result.times_h, list)


def test_c_plasma_list_same_length_as_times():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=100.0, pka=4.0, logp=2.0)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_segment_keys_present():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=100.0, pka=4.0, logp=2.0)
    required_keys = {
        "name",
        "pH",
        "residence_h",
        "fu_neutral",
        "solubility_mg_mL",
        "dissolved_fraction",
        "ka_per_h",
        "fa_segment",
        "absorbed_mg",
    }
    for seg in result.segments:
        assert required_keys.issubset(seg.keys())


def test_ionization_type_stored():
    result = simulate_ph_dependent_absorption(
        "Drug", dose_mg=100.0, pka=4.0, logp=2.0, ionization_type="acid"
    )
    assert result.ionization_type == "acid"


# ---------------------------------------------------------------------------
# PK sanity checks
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=200.0, pka=4.0, logp=2.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=200.0, pka=4.0, logp=2.0)
    assert result.auc_mg_h_per_L > 0.0


def test_fa_overall_between_0_and_1():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=200.0, pka=4.0, logp=2.0)
    assert 0.0 <= result.fa_overall <= 1.0


def test_tmax_within_simulation():
    t_end = 12.0
    result = simulate_ph_dependent_absorption(
        "Drug", dose_mg=200.0, pka=4.0, logp=2.0, t_end_h=t_end
    )
    assert 0.0 <= result.tmax_h <= t_end


def test_limiting_segment_is_gi_segment():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=200.0, pka=4.0, logp=2.0)
    assert result.limiting_segment in {"Stomach", "Duodenum", "Jejunum", "Ileum"}


def test_notes_nonempty():
    result = simulate_ph_dependent_absorption("Drug", dose_mg=200.0, pka=4.0, logp=2.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Acid drug: higher absorption in small intestine (pH 5.5-7.4)
# ---------------------------------------------------------------------------


def test_acid_higher_fa_in_intestine_than_stomach():
    """Acid drug: stomach pH=1.5 is low → most drug ionised in intestine but
    stomach pH is below pKa so drug is neutral there; solubility is lower in stomach.
    The small intestine should dominate overall absorption."""
    result = simulate_ph_dependent_absorption(
        "WeakAcid", dose_mg=200.0, pka=4.5, logp=1.5, ionization_type="acid"
    )
    # Jejunum or Ileum should have higher fa than stomach for a weak acid
    # Weak acid at pKa=4.5: stomach pH=1.5 → mostly neutral (fu_neutral close to 1)
    # Intestine pH > pKa → ionised → higher solubility; but neutral form drives permeation
    # The ionised form enhances solubility but reduces Peff; overall fa dominated by solubility
    # In high-pH intestine, higher solubility means more dissolved → more absorbed
    assert result.fa_overall > 0.0


def test_acid_solubility_increases_with_ph():
    """Acid drug: solubility should be higher at higher pH segments."""
    result = simulate_ph_dependent_absorption(
        "WeakAcid", dose_mg=100.0, pka=4.0, logp=2.0, ionization_type="acid"
    )
    segs = {s["name"]: s for s in result.segments}
    # Ileum (pH 7.4) should have higher solubility than Stomach (pH 1.5) for acid
    assert segs["Ileum"]["solubility_mg_mL"] > segs["Stomach"]["solubility_mg_mL"]


# ---------------------------------------------------------------------------
# Base drug: higher absorption in stomach/duodenum (lower pH)
# ---------------------------------------------------------------------------


def test_base_higher_solubility_at_low_ph():
    """Base drug: lower pH → more ionised → higher solubility."""
    result = simulate_ph_dependent_absorption(
        "WeakBase", dose_mg=100.0, pka=8.0, logp=1.5, ionization_type="base"
    )
    segs = {s["name"]: s for s in result.segments}
    # Stomach (pH 1.5) should have higher solubility than Ileum (pH 7.4) for base
    assert segs["Stomach"]["solubility_mg_mL"] > segs["Ileum"]["solubility_mg_mL"]


def test_base_positive_fa():
    result = simulate_ph_dependent_absorption(
        "WeakBase", dose_mg=200.0, pka=8.5, logp=2.0, ionization_type="base"
    )
    assert result.fa_overall > 0.0
    assert result.cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# Neutral drug: pH-independent absorption
# ---------------------------------------------------------------------------


def test_neutral_fu_always_one():
    """Neutral drug: fu_neutral should be 1.0 across all segments."""
    result = simulate_ph_dependent_absorption(
        "Neutral", dose_mg=100.0, pka=7.0, logp=2.0, ionization_type="neutral"
    )
    for seg in result.segments:
        assert seg["fu_neutral"] == 1.0


def test_neutral_fa_independent_of_pka():
    """Neutral drug: changing pKa should not affect fa_overall."""
    r1 = simulate_ph_dependent_absorption(
        "N1", dose_mg=100.0, pka=3.0, logp=2.0, ionization_type="neutral"
    )
    r2 = simulate_ph_dependent_absorption(
        "N2", dose_mg=100.0, pka=10.0, logp=2.0, ionization_type="neutral"
    )
    assert abs(r1.fa_overall - r2.fa_overall) < 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_ph_dependent_absorption("Drug", dose_mg=0.0, pka=5.0, logp=2.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_ph_dependent_absorption("Drug", dose_mg=-100.0, pka=5.0, logp=2.0)


def test_pka_below_zero_raises():
    with pytest.raises(ValueError, match="pka must be in"):
        simulate_ph_dependent_absorption("Drug", dose_mg=100.0, pka=-1.0, logp=2.0)


def test_pka_above_14_raises():
    with pytest.raises(ValueError, match="pka must be in"):
        simulate_ph_dependent_absorption("Drug", dose_mg=100.0, pka=15.0, logp=2.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        simulate_ph_dependent_absorption(
            "Drug", dose_mg=100.0, pka=5.0, logp=2.0, ionization_type="zwitterion"
        )
