"""Tests for inhalational anesthetic PK model — Phase 272."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.anesthetic_pk import (
    AnestheticPKResult,
    simulate_inhalational_anesthetic,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default(**kwargs) -> AnestheticPKResult:
    defaults = dict(
        drug_name="sevoflurane",
        lambda_bg=0.65,
        lambda_brain_blood=1.7,
        lambda_muscle_blood=3.1,
        lambda_fat_blood=48.0,
        alveolar_ventilation_L_per_min=4.0,
        cardiac_output_L_per_min=5.0,
        fi=1.0,
        t_end_min=30.0,
        dt_min=0.1,
    )
    defaults.update(kwargs)
    return simulate_inhalational_anesthetic(**defaults)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param,val",
    [
        ("lambda_bg", 0.0),
        ("lambda_bg", -1.0),
        ("lambda_brain_blood", 0.0),
        ("lambda_muscle_blood", -0.5),
        ("lambda_fat_blood", 0.0),
        ("alveolar_ventilation_L_per_min", 0.0),
        ("cardiac_output_L_per_min", -1.0),
        ("fi", 0.0),
        ("t_end_min", 0.0),
    ],
)
def test_invalid_param_raises(param, val):
    with pytest.raises(ValueError, match=param):
        _default(**{param: val})


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_lengths_match():
    result = _default()
    n = len(result.times_min)
    assert len(result.fa_fi_ratio) == n
    assert len(result.c_arterial) == n
    assert len(result.c_brain) == n
    assert len(result.c_muscle) == n
    assert len(result.c_fat) == n


def test_drug_name_stored():
    result = _default(drug_name="isoflurane")
    assert result.drug_name == "isoflurane"


def test_notes_contains_drug_name():
    result = _default(drug_name="desflurane")
    assert "desflurane" in result.notes


# ---------------------------------------------------------------------------
# Physical / pharmacological correctness
# ---------------------------------------------------------------------------


def test_fa_fi_starts_at_zero():
    result = _default()
    assert result.fa_fi_ratio[0] == pytest.approx(0.0)


def test_fa_fi_increases_over_time():
    result = _default()
    # FA/FI should be monotonically non-decreasing overall
    assert result.fa_fi_ratio[-1] > result.fa_fi_ratio[0]


def test_fa_fi_last_greater_than_zero():
    result = _default()
    assert result.fa_fi_ratio[-1] > 0.0


def test_brain_conc_positive_at_end():
    result = _default()
    assert result.c_brain[-1] > 0.0


def test_all_concentrations_non_negative():
    result = _default()
    for val in result.c_arterial:
        assert val >= 0.0
    for val in result.c_brain:
        assert val >= 0.0
    for val in result.c_muscle:
        assert val >= 0.0
    for val in result.c_fat:
        assert val >= 0.0


def test_time_to_half_fa_fi_positive():
    result = _default()
    assert result.time_to_half_fa_fi_min >= 0.0


def test_fat_conc_less_than_brain_at_30_min():
    """Fat equilibrates very slowly (large lambda_fat_blood); at 30 min brain > fat."""
    result = _default(t_end_min=30.0)
    assert result.c_brain[-1] > result.c_fat[-1]


def test_higher_lambda_bg_slower_fa_fi_rise():
    """More soluble agent (higher lambda_bg) → blood uptake faster → FA/FI rises slower."""
    result_low = _default(lambda_bg=0.5, t_end_min=20.0)
    result_high = _default(lambda_bg=1.4, t_end_min=20.0)
    # At same time, lower lambda_bg → higher FA/FI (less blood uptake)
    assert result_low.fa_fi_ratio[-1] > result_high.fa_fi_ratio[-1]


def test_fa_fi_ratio_bounded_by_one():
    result = _default()
    for r in result.fa_fi_ratio:
        assert r <= 1.0 + 1e-9


def test_times_start_at_zero():
    result = _default()
    assert result.times_min[0] == pytest.approx(0.0)


def test_arterial_conc_increases_over_time():
    result = _default()
    assert result.c_arterial[-1] > result.c_arterial[0]


def test_short_simulation():
    """Short simulation should still produce valid result."""
    result = _default(t_end_min=5.0)
    assert len(result.fa_fi_ratio) > 0
    assert result.fa_fi_ratio[-1] >= 0.0
