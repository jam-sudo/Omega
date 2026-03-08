"""Tests for Phase 258: CornealPermeationPK."""

import pytest

from omega_pbpk.core.corneal_permeation_pk import (
    CornealPermeationResult,
    compare_dosage_forms,
    simulate_corneal_permeation,
)

DRUG = "Timolol"
DOSE = 0.1  # mg
LOGP = 1.9  # moderate
MW = 316.4  # g/mol


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW)
    assert isinstance(result, CornealPermeationResult)


# ---------------------------------------------------------------------------
# Tear concentration starts positive (after lag)
# ---------------------------------------------------------------------------


def test_tear_starts_positive_eye_drop():
    """eye_drop has lag=0 so tear[0] should be positive."""
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW, "eye_drop")
    assert result.c_tear_mg_mL[0] > 0.0


def test_tear_positive_after_lag_gel():
    """gel has lag=2 min; with dt=0.5 tear is zero at t=0 but positive at t=2."""
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW, "gel")
    # Find index where t >= lag
    lag_idx = next(i for i, t in enumerate(result.times_min) if t >= 2.0)
    assert result.c_tear_mg_mL[lag_idx] > 0.0


# ---------------------------------------------------------------------------
# Aqueous starts at 0
# ---------------------------------------------------------------------------


def test_aqueous_starts_zero():
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW, "eye_drop")
    assert result.c_aqueous_mg_mL[0] == 0.0


# ---------------------------------------------------------------------------
# Cmax aqueous > 0
# ---------------------------------------------------------------------------


def test_cmax_aqueous_positive():
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW)
    assert result.cmax_aqueous_mg_mL > 0.0


# ---------------------------------------------------------------------------
# AUC aqueous > 0
# ---------------------------------------------------------------------------


def test_auc_aqueous_positive():
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW)
    assert result.auc_aqueous_mg_min_per_mL > 0.0


# ---------------------------------------------------------------------------
# Bioavailability in [0, 100]
# ---------------------------------------------------------------------------


def test_bioavailability_range():
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW)
    assert 0.0 <= result.bioavailability_pct <= 100.0


def test_bioavailability_range_all_forms():
    for form in ("eye_drop", "gel", "ointment", "insert"):
        result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW, form)
        assert 0.0 <= result.bioavailability_pct <= 100.0, form


# ---------------------------------------------------------------------------
# Higher logP -> higher permeability
# ---------------------------------------------------------------------------


def test_higher_logp_higher_permeability():
    r_low = simulate_corneal_permeation(DRUG, DOSE, 0.5, MW)
    r_high = simulate_corneal_permeation(DRUG, DOSE, 3.0, MW)
    assert r_high.corneal_permeability_cm_s > r_low.corneal_permeability_cm_s


def test_higher_logp_higher_auc():
    r_low = simulate_corneal_permeation(DRUG, DOSE, 0.5, MW)
    r_high = simulate_corneal_permeation(DRUG, DOSE, 3.0, MW)
    assert r_high.auc_aqueous_mg_min_per_mL > r_low.auc_aqueous_mg_min_per_mL


# ---------------------------------------------------------------------------
# Permeability stored correctly
# ---------------------------------------------------------------------------


def test_permeability_positive():
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW)
    assert result.corneal_permeability_cm_s > 0.0


# ---------------------------------------------------------------------------
# Times list length consistent
# ---------------------------------------------------------------------------


def test_times_length_consistency():
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW, t_end_min=60.0, dt_min=1.0)
    n = len(result.times_min)
    assert len(result.c_tear_mg_mL) == n
    assert len(result.c_cornea_mg_mL) == n
    assert len(result.c_aqueous_mg_mL) == n


# ---------------------------------------------------------------------------
# compare_dosage_forms
# ---------------------------------------------------------------------------


def test_compare_returns_4():
    results = compare_dosage_forms(DRUG, DOSE, LOGP, MW)
    assert len(results) == 4


def test_compare_sorted_descending():
    results = compare_dosage_forms(DRUG, DOSE, LOGP, MW)
    aucs = [r.auc_aqueous_mg_min_per_mL for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_forms_represented():
    results = compare_dosage_forms(DRUG, DOSE, LOGP, MW)
    forms = {r.dosage_form for r in results}
    assert forms == {"eye_drop", "gel", "ointment", "insert"}


def test_compare_return_type():
    results = compare_dosage_forms(DRUG, DOSE, LOGP, MW)
    for r in results:
        assert isinstance(r, CornealPermeationResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_corneal_permeation(DRUG, 0.0, LOGP, MW)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_corneal_permeation(DRUG, -0.1, LOGP, MW)


def test_invalid_dosage_form():
    with pytest.raises(ValueError, match="dosage_form"):
        simulate_corneal_permeation(DRUG, DOSE, LOGP, MW, dosage_form="spray")


# ---------------------------------------------------------------------------
# Tmax within simulation window
# ---------------------------------------------------------------------------


def test_tmax_within_simulation():
    result = simulate_corneal_permeation(DRUG, DOSE, LOGP, MW, t_end_min=120.0)
    assert 0.0 <= result.tmax_aqueous_min <= 120.0
