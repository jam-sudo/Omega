"""Tests for Phase 1047: GI Dissolution Prediction."""

import pytest

from omega_pbpk.prediction.gi_dissolution_prediction import (
    GIDissolutionResult,
    predict_gi_dissolution,
)

# ---------------------------------------------------------------------------
# Helper: default call
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logp=2.0,
        pka=4.0,
        ionization_type="acid",
        dose_mg=100.0,
        particle_size_um=50.0,
        gi_segment="small_intestine",
        bcs_solubility_class="low",
        surface_area_cm2=1.0,
        t_end_min=60.0,
        dt_min=0.5,
    )
    params.update(kwargs)
    return predict_gi_dissolution(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default()
    assert isinstance(result, GIDissolutionResult)


# ---------------------------------------------------------------------------
# fraction_dissolved starts at 0
# ---------------------------------------------------------------------------


def test_fraction_dissolved_starts_zero():
    result = _default()
    assert result.fraction_dissolved[0] == 0.0


# ---------------------------------------------------------------------------
# fraction_dissolved is monotonically non-decreasing
# ---------------------------------------------------------------------------


def test_fraction_dissolved_monotone():
    result = _default()
    fd = result.fraction_dissolved
    for i in range(1, len(fd)):
        assert fd[i] >= fd[i - 1] - 1e-12, f"Not monotone at index {i}"


# ---------------------------------------------------------------------------
# fraction_dissolved final value in [0, 1]
# ---------------------------------------------------------------------------


def test_fraction_dissolved_final_in_range():
    result = _default()
    assert 0.0 <= result.fraction_dissolved[-1] <= 1.0


# ---------------------------------------------------------------------------
# dissolution_rate_mg_min non-negative
# ---------------------------------------------------------------------------


def test_dissolution_rate_non_negative():
    result = _default()
    for rate in result.dissolution_rate_mg_min:
        assert rate >= 0.0


# ---------------------------------------------------------------------------
# t50 <= t90
# ---------------------------------------------------------------------------


def test_t50_leq_t90():
    result = _default()
    assert result.t50_min <= result.t90_min


# ---------------------------------------------------------------------------
# intrinsic_dissolution_rate > 0
# ---------------------------------------------------------------------------


def test_idr_positive():
    result = _default()
    assert result.intrinsic_dissolution_rate > 0.0


# ---------------------------------------------------------------------------
# biorelevant_solubility_mg_mL > 0
# ---------------------------------------------------------------------------


def test_biorelevant_solubility_positive():
    result = _default()
    assert result.biorelevant_solubility_mg_mL > 0.0


# ---------------------------------------------------------------------------
# bcs_absorption_potential in valid set
# ---------------------------------------------------------------------------


def test_bcs_absorption_potential_valid():
    result = _default()
    assert result.bcs_absorption_potential in {"high", "moderate", "low"}


# ---------------------------------------------------------------------------
# Acid drug: higher solubility in small_intestine (pH 6.5) than stomach (pH 1.5)
# ---------------------------------------------------------------------------


def test_acid_higher_solubility_small_intestine_vs_stomach():
    si = _default(ionization_type="acid", pka=4.0, gi_segment="small_intestine")
    st = _default(ionization_type="acid", pka=4.0, gi_segment="stomach")
    assert si.biorelevant_solubility_mg_mL > st.biorelevant_solubility_mg_mL


# ---------------------------------------------------------------------------
# High BCS solubility class → lower t50 than low class
# ---------------------------------------------------------------------------


def test_high_bcs_lower_t50():
    high = _default(bcs_solubility_class="high")
    low = _default(bcs_solubility_class="low")
    assert high.t50_min <= low.t50_min


# ---------------------------------------------------------------------------
# Smaller particles: surface area same but IDR same; use surface_area_cm2 proxy:
# Larger surface area → faster dissolution (equivalent to smaller particles)
# ---------------------------------------------------------------------------


def test_larger_surface_area_faster_dissolution():
    fast = _default(surface_area_cm2=5.0)
    slow = _default(surface_area_cm2=1.0)
    # Larger SA means more dissolution; final fraction should be >= slow
    assert fast.fraction_dissolved[-1] >= slow.fraction_dissolved[-1]


# ---------------------------------------------------------------------------
# notes nonempty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _default()
    assert result.notes and len(result.notes) > 0


# ---------------------------------------------------------------------------
# times_min has expected length
# ---------------------------------------------------------------------------


def test_times_length_consistent():
    result = _default(t_end_min=30.0, dt_min=1.0)
    assert len(result.times_min) == len(result.fraction_dissolved)
    assert len(result.times_min) == len(result.dissolution_rate_mg_min)


# ---------------------------------------------------------------------------
# Neutral drug: solubility not pH-dependent beyond bile factor
# ---------------------------------------------------------------------------


def test_neutral_drug_solubility():
    result = _default(ionization_type="neutral")
    assert result.biorelevant_solubility_mg_mL > 0.0


# ---------------------------------------------------------------------------
# Base drug: higher solubility in stomach (low pH) than colon (high pH)
# ---------------------------------------------------------------------------


def test_base_higher_solubility_stomach():
    st = _default(ionization_type="base", pka=8.0, gi_segment="stomach")
    co = _default(ionization_type="base", pka=8.0, gi_segment="colon")
    assert st.biorelevant_solubility_mg_mL > co.biorelevant_solubility_mg_mL


# ---------------------------------------------------------------------------
# t50 and t90 are within [0, t_end_min]
# ---------------------------------------------------------------------------


def test_t50_t90_within_range():
    result = _default(t_end_min=60.0)
    assert 0.0 <= result.t50_min <= 60.0
    assert 0.0 <= result.t90_min <= 60.0


# ---------------------------------------------------------------------------
# High IDR drug dissolves substantially
# ---------------------------------------------------------------------------


def test_high_idr_dissolves():
    result = _default(
        bcs_solubility_class="high",
        dose_mg=10.0,
        surface_area_cm2=10.0,
        t_end_min=120.0,
    )
    assert result.fraction_dissolved[-1] > 0.05  # at least 5% dissolved


# ---------------------------------------------------------------------------
# bcs_absorption_potential consistency with t50
# ---------------------------------------------------------------------------


def test_bcs_potential_consistency():
    result = _default(
        bcs_solubility_class="high", surface_area_cm2=10.0, dose_mg=10.0, t_end_min=120.0
    )
    if result.t50_min < 15.0:
        assert result.bcs_absorption_potential == "high"
    elif result.t50_min < 45.0:
        assert result.bcs_absorption_potential == "moderate"
    else:
        assert result.bcs_absorption_potential == "low"


# ---------------------------------------------------------------------------
# Validation: dose=0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _default(dose_mg=0.0)


# ---------------------------------------------------------------------------
# Validation: invalid gi_segment raises ValueError
# ---------------------------------------------------------------------------


def test_validation_invalid_segment():
    with pytest.raises(ValueError, match="gi_segment"):
        _default(gi_segment="bloodstream")


# ---------------------------------------------------------------------------
# Validation: invalid ionization_type raises ValueError
# ---------------------------------------------------------------------------


def test_validation_invalid_ionization():
    with pytest.raises(ValueError, match="ionization_type"):
        _default(ionization_type="zwitterion")


# ---------------------------------------------------------------------------
# Validation: invalid bcs_solubility_class raises ValueError
# ---------------------------------------------------------------------------


def test_validation_invalid_bcs_class():
    with pytest.raises(ValueError, match="bcs_solubility_class"):
        _default(bcs_solubility_class="medium")
