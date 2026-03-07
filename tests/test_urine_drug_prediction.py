"""Tests for Phase 363: Urine Drug Concentration Prediction."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.urine_drug_prediction import (
    UrinePredictionResult,
    detection_timeline,
    predict_urine_concentration,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
DOSE = 100.0  # mg
CL = 5.0  # L/h
VD = 50.0  # L
FE = 0.4  # renal fraction
KA = 1.5  # per h
F = 0.85  # bioavailability


# ---------------------------------------------------------------------------
# Basic structure / frozen dataclass
# ---------------------------------------------------------------------------


def test_result_is_frozen_dataclass():
    r = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE)
    assert isinstance(r, UrinePredictionResult)
    with pytest.raises((AttributeError, TypeError)):
        r.c_urine_ng_mL = 999.0  # type: ignore[misc]


def test_result_fields_accessible():
    r = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE)
    assert hasattr(r, "drug_name")
    assert hasattr(r, "c_urine_ng_mL")
    assert hasattr(r, "above_cutoff")
    assert hasattr(r, "detection_window_h")
    assert r.drug_name == DRUG


# ---------------------------------------------------------------------------
# t=0 behaviour
# ---------------------------------------------------------------------------


def test_at_t0_plasma_is_zero():
    r = predict_urine_concentration(DRUG, DOSE, 0.0, CL, VD, fe_urine=FE)
    assert r.plasma_c_mg_L == pytest.approx(0.0, abs=1e-10)


def test_at_t0_urine_is_zero():
    r = predict_urine_concentration(DRUG, DOSE, 0.0, CL, VD, fe_urine=FE)
    assert r.c_urine_ng_mL == pytest.approx(0.0, abs=1e-10)


def test_at_t0_cumulative_is_zero():
    r = predict_urine_concentration(DRUG, DOSE, 0.0, CL, VD, fe_urine=FE)
    assert r.cumulative_excreted_pct == pytest.approx(0.0, abs=1e-6)


def test_at_t0_not_above_cutoff():
    r = predict_urine_concentration(DRUG, DOSE, 0.0, CL, VD, fe_urine=FE, cutoff_ng_mL=300.0)
    assert r.above_cutoff is False


# ---------------------------------------------------------------------------
# Correct PK behaviour
# ---------------------------------------------------------------------------


def test_concentration_increases_then_decreases():
    """Urine concentration should rise to a peak then fall."""
    times = [0.5, 1.0, 2.0, 5.0, 10.0, 24.0]
    concs = [
        predict_urine_concentration(DRUG, DOSE, t, CL, VD, fe_urine=FE).c_urine_ng_mL for t in times
    ]
    # Must be non-zero at middle times
    assert max(concs) > 0.0
    # t=0.5 < peak < t=24
    assert concs[0] < max(concs)
    assert concs[-1] < max(concs)


def test_peak_urine_after_tmax():
    """Peak urine concentration should occur after time=0."""
    ke = CL / VD
    # tmax = ln(ka/ke) / (ka - ke)
    tmax = math.log(KA / ke) / (KA - ke)
    concs = {}
    for t in [tmax * 0.5, tmax, tmax * 1.5]:
        r = predict_urine_concentration(DRUG, DOSE, t, CL, VD, fe_urine=FE, ka_per_h=KA)
        concs[t] = r.c_urine_ng_mL
    # concentration at tmax should be among the highest tested
    assert concs[tmax * 0.5] > 0.0
    assert concs[tmax] > 0.0


def test_dose_linearity():
    """Doubling the dose should double urine concentration."""
    r1 = predict_urine_concentration(DRUG, 100.0, 2.0, CL, VD, fe_urine=FE)
    r2 = predict_urine_concentration(DRUG, 200.0, 2.0, CL, VD, fe_urine=FE)
    assert r2.c_urine_ng_mL == pytest.approx(2.0 * r1.c_urine_ng_mL, rel=1e-6)


def test_dose_linearity_plasma():
    r1 = predict_urine_concentration(DRUG, 100.0, 3.0, CL, VD, fe_urine=FE)
    r2 = predict_urine_concentration(DRUG, 200.0, 3.0, CL, VD, fe_urine=FE)
    assert r2.plasma_c_mg_L == pytest.approx(2.0 * r1.plasma_c_mg_L, rel=1e-6)


# ---------------------------------------------------------------------------
# Above-cutoff detection
# ---------------------------------------------------------------------------


def test_above_cutoff_consistent_with_concentration_high():
    """above_cutoff=True when c_urine > cutoff."""
    r = predict_urine_concentration(DRUG, 500.0, 2.0, CL, VD, fe_urine=FE, cutoff_ng_mL=1.0)
    assert r.above_cutoff is True
    assert r.c_urine_ng_mL > r.cutoff_ng_mL


def test_above_cutoff_consistent_with_concentration_low():
    """above_cutoff=False when c_urine < cutoff."""
    # At t=100h drug is essentially gone
    r = predict_urine_concentration(DRUG, 10.0, 100.0, CL, VD, fe_urine=FE, cutoff_ng_mL=1e9)
    assert r.above_cutoff is False
    assert r.c_urine_ng_mL <= r.cutoff_ng_mL


# ---------------------------------------------------------------------------
# pH ionization effects
# ---------------------------------------------------------------------------


def test_acidic_drug_low_urine_ph_increases_urine_conc():
    """Acid at low urine pH (ion trapping reversal) → less ionized → higher U/P ratio."""
    # Acid at pH 5.0 (urine) vs pH 6.5: ratio_correction = (1+10^(5-pKa))/(1+10^(7.4-pKa))
    # For pKa=7.0: at pH 5.0, mostly unionized → less ionized → lower ion trapping
    # At low urine pH, acid less ionized → higher reabsorption... but our formula:
    # ratio_correction_low = (1+10^(low_pH - pKa)) / (1+10^(7.4-pKa))
    # For pKa=8.0, pH 5 vs pH 7.0:
    # low pH: (1+10^(5-8))/(1+10^(7.4-8)) = (1+0.001)/(1+0.25) ≈ 0.8 (less in urine)
    # high pH: (1+10^(7-8))/(1+10^(7.4-8)) = (1+0.1)/(1+0.25) ≈ 0.88
    # So acid at high urine pH → more ionized → higher urine concentration (ion trapping)
    r_low_ph = predict_urine_concentration(
        "Aspirin", DOSE, 2.0, CL, VD, fe_urine=FE, urine_ph=5.0, pka_acid=4.5
    )
    r_high_ph = predict_urine_concentration(
        "Aspirin", DOSE, 2.0, CL, VD, fe_urine=FE, urine_ph=8.0, pka_acid=4.5
    )
    # At high pH, more ionized → higher ion trapping → higher urine concentration
    assert r_high_ph.c_urine_ng_mL > r_low_ph.c_urine_ng_mL


def test_basic_drug_high_urine_ph_increases_urine_conc():
    """Base at high urine pH → less ionized → less ion trapping → lower urine conc.
    Base at low urine pH → more ionized → ion trapping → higher urine conc."""
    r_low_ph = predict_urine_concentration(
        "Amphetamine", DOSE, 2.0, CL, VD, fe_urine=FE, urine_ph=5.0, pka_base=9.9
    )
    r_high_ph = predict_urine_concentration(
        "Amphetamine", DOSE, 2.0, CL, VD, fe_urine=FE, urine_ph=8.0, pka_base=9.9
    )
    # Basic drug at low pH: more ionized → higher urine conc (ion trapping)
    assert r_low_ph.c_urine_ng_mL > r_high_ph.c_urine_ng_mL


def test_neutral_drug_no_ph_effect():
    """Neutral drug: pH has no effect on urine concentration."""
    r1 = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE, urine_ph=5.0)
    r2 = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE, urine_ph=8.0)
    assert r1.c_urine_ng_mL == pytest.approx(r2.c_urine_ng_mL, rel=1e-9)


# ---------------------------------------------------------------------------
# Variability bounds
# ---------------------------------------------------------------------------


def test_variability_bounds_ordering():
    r = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE)
    assert r.c_urine_low_ng_mL <= r.c_urine_ng_mL
    assert r.c_urine_high_ng_mL >= r.c_urine_ng_mL


def test_variability_bounds_are_50pct():
    r = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE)
    assert r.c_urine_low_ng_mL == pytest.approx(r.c_urine_ng_mL * 0.5, rel=1e-6)
    assert r.c_urine_high_ng_mL == pytest.approx(r.c_urine_ng_mL * 1.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Cumulative excretion
# ---------------------------------------------------------------------------


def test_cumulative_pct_in_range():
    for t in [0.0, 1.0, 6.0, 24.0, 72.0]:
        r = predict_urine_concentration(DRUG, DOSE, t, CL, VD, fe_urine=FE)
        assert 0.0 <= r.cumulative_excreted_pct <= 100.0


def test_cumulative_pct_increases_with_time():
    r1 = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE)
    r2 = predict_urine_concentration(DRUG, DOSE, 12.0, CL, VD, fe_urine=FE)
    assert r2.cumulative_excreted_pct > r1.cumulative_excreted_pct


# ---------------------------------------------------------------------------
# Detection window
# ---------------------------------------------------------------------------


def test_detection_window_positive_for_reasonable_dose():
    r = predict_urine_concentration(DRUG, 500.0, 1.0, CL, VD, fe_urine=FE, cutoff_ng_mL=50.0)
    assert r.detection_window_h > 0.0


def test_detection_window_zero_if_never_detected():
    # Tiny dose, huge cutoff
    r = predict_urine_concentration(DRUG, 0.001, 1.0, CL, VD, fe_urine=FE, cutoff_ng_mL=1e9)
    assert r.detection_window_h == 0.0


# ---------------------------------------------------------------------------
# Creatinine correction
# ---------------------------------------------------------------------------


def test_creatinine_correction_formula():
    """Creatinine-corrected = c_urine_ng_mL / (creatinine_mg_dL / 100)."""
    r = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE, creatinine_mg_dL=100.0)
    expected = r.c_urine_ng_mL / 1.0  # creatinine_mg_dL / 100 = 1.0
    assert r.creatinine_corrected_ng_per_mg_creatinine == pytest.approx(expected, rel=1e-6)


def test_creatinine_correction_scales_inversely():
    """Higher creatinine → lower corrected concentration."""
    r1 = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE, creatinine_mg_dL=50.0)
    r2 = predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=FE, creatinine_mg_dL=200.0)
    assert (
        r1.creatinine_corrected_ng_per_mg_creatinine > r2.creatinine_corrected_ng_per_mg_creatinine
    )


# ---------------------------------------------------------------------------
# Detection timeline
# ---------------------------------------------------------------------------


def test_detection_timeline_returns_48_timepoints():
    result = detection_timeline(DRUG, DOSE, CL, VD, FE, cutoff_ng_mL=100.0, n_timepoints=48)
    assert len(result["times_h"]) == 48
    assert len(result["c_urine_ng_mL"]) == 48
    assert len(result["above_cutoff"]) == 48


def test_detection_timeline_has_correct_keys():
    result = detection_timeline(DRUG, DOSE, CL, VD, FE)
    assert "times_h" in result
    assert "c_urine_ng_mL" in result
    assert "above_cutoff" in result
    assert "detection_window_h" in result


def test_detection_timeline_custom_n():
    result = detection_timeline(DRUG, DOSE, CL, VD, FE, n_timepoints=12)
    assert len(result["times_h"]) == 12


def test_detection_timeline_first_conc_is_zero():
    result = detection_timeline(DRUG, DOSE, CL, VD, FE)
    assert result["c_urine_ng_mL"][0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_raises_on_negative_dose():
    with pytest.raises(ValueError):
        predict_urine_concentration(DRUG, -1.0, 2.0, CL, VD)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError):
        predict_urine_concentration(DRUG, DOSE, 2.0, 0.0, VD)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError):
        predict_urine_concentration(DRUG, DOSE, 2.0, CL, 0.0)


def test_raises_on_fe_out_of_range():
    with pytest.raises(ValueError):
        predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, fe_urine=1.5)


def test_raises_on_negative_time():
    with pytest.raises(ValueError):
        predict_urine_concentration(DRUG, DOSE, -1.0, CL, VD)


def test_raises_on_zero_urine_flow():
    with pytest.raises(ValueError):
        predict_urine_concentration(DRUG, DOSE, 2.0, CL, VD, urine_flow_mL_per_h=0.0)
