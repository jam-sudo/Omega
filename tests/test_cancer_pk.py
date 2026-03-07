"""Tests for clinical/cancer_pk.py — Phase 397."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.cancer_pk import (
    CancerDrugPKResult,
    _trapezoidal_auc,
    nadir_prediction,
    simulate_cancer_drug_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_result(**kwargs):
    defaults = dict(
        drug_name="doxorubicin",
        drug_class="cytotoxic",
        dose_mg_m2=60.0,
        bsa_m2=1.8,
        cl_L_per_h=45.0,
        vd_L=700.0,
        interval_h=504.0,
        n_cycles=4,
        dt_h=0.5,
    )
    defaults.update(kwargs)
    return simulate_cancer_drug_pk(**defaults)


# ---------------------------------------------------------------------------
# Tests: _trapezoidal_auc
# ---------------------------------------------------------------------------


def test_trapezoidal_auc_basic():
    times = [0.0, 1.0, 2.0]
    conc = [0.0, 1.0, 0.0]
    auc = _trapezoidal_auc(times, conc)
    assert abs(auc - 1.0) < 1e-9


def test_trapezoidal_auc_constant():
    times = [0.0, 5.0]
    conc = [2.0, 2.0]
    auc = _trapezoidal_auc(times, conc)
    assert abs(auc - 10.0) < 1e-9


def test_trapezoidal_auc_single_point():
    # Only one point — no intervals
    auc = _trapezoidal_auc([0.0], [5.0])
    assert auc == 0.0


# ---------------------------------------------------------------------------
# Tests: simulate_cancer_drug_pk — validation
# ---------------------------------------------------------------------------


def test_validate_empty_drug_name():
    with pytest.raises(ValueError, match="drug_name"):
        simulate_cancer_drug_pk("", "cytotoxic", 60.0, 1.8, 45.0, 700.0)


def test_validate_whitespace_drug_name():
    with pytest.raises(ValueError, match="drug_name"):
        simulate_cancer_drug_pk("   ", "cytotoxic", 60.0, 1.8, 45.0, 700.0)


def test_validate_bad_drug_class():
    with pytest.raises(ValueError, match="drug_class"):
        simulate_cancer_drug_pk("drug", "unknown", 60.0, 1.8, 45.0, 700.0)


def test_validate_nonpositive_dose():
    with pytest.raises(ValueError, match="dose_mg_m2"):
        simulate_cancer_drug_pk("drug", "cytotoxic", 0.0, 1.8, 45.0, 700.0)


def test_validate_negative_dose():
    with pytest.raises(ValueError, match="dose_mg_m2"):
        simulate_cancer_drug_pk("drug", "cytotoxic", -10.0, 1.8, 45.0, 700.0)


def test_validate_nonpositive_bsa():
    with pytest.raises(ValueError, match="bsa_m2"):
        simulate_cancer_drug_pk("drug", "cytotoxic", 60.0, 0.0, 45.0, 700.0)


def test_validate_nonpositive_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_cancer_drug_pk("drug", "cytotoxic", 60.0, 1.8, 0.0, 700.0)


def test_validate_nonpositive_vd():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_cancer_drug_pk("drug", "cytotoxic", 60.0, 1.8, 45.0, 0.0)


def test_validate_nonpositive_interval():
    with pytest.raises(ValueError, match="interval_h"):
        simulate_cancer_drug_pk("drug", "cytotoxic", 60.0, 1.8, 45.0, 700.0, interval_h=0.0)


def test_validate_zero_cycles():
    with pytest.raises(ValueError, match="n_cycles"):
        simulate_cancer_drug_pk("drug", "cytotoxic", 60.0, 1.8, 45.0, 700.0, n_cycles=0)


def test_validate_nonpositive_dt():
    with pytest.raises(ValueError, match="dt_h"):
        simulate_cancer_drug_pk("drug", "cytotoxic", 60.0, 1.8, 45.0, 700.0, dt_h=0.0)


# ---------------------------------------------------------------------------
# Tests: cytotoxic IV simulation
# ---------------------------------------------------------------------------


def test_cytotoxic_dose_computation():
    """Dose in mg = dose_mg_m2 * bsa_m2 for cytotoxic."""
    result = _make_result(drug_class="cytotoxic", dose_mg_m2=60.0, bsa_m2=1.8)
    assert abs(result.dose_mg - 60.0 * 1.8) < 1e-6


def test_cytotoxic_result_type():
    result = _make_result()
    assert isinstance(result, CancerDrugPKResult)


def test_cytotoxic_n_cycles():
    result = _make_result(n_cycles=3)
    assert result.n_cycles == 3
    assert len(result.cmax_per_cycle) == 3
    assert len(result.auc_per_cycle) == 3
    assert len(result.trough_per_cycle) == 3


def test_cytotoxic_cmax_positive():
    result = _make_result()
    assert all(c > 0 for c in result.cmax_per_cycle)


def test_cytotoxic_cumulative_auc():
    result = _make_result(n_cycles=4)
    assert abs(result.cumulative_auc - sum(result.auc_per_cycle)) < 1e-6


def test_cytotoxic_times_length():
    result = _make_result(n_cycles=2, interval_h=24.0, dt_h=0.5)
    # Expect roughly n_cycles * interval_h / dt_h + 1 steps
    expected = int(round(2 * 24.0 / 0.5)) + 1
    assert abs(len(result.times_h) - expected) <= 2


def test_cytotoxic_concentrations_nonnegative():
    result = _make_result()
    assert all(c >= 0 for c in result.c_plasma_mg_L)


def test_cytotoxic_decay_after_peak():
    """After first dose peak, concentration should generally decay."""
    result = _make_result(n_cycles=1, interval_h=48.0, dt_h=0.5, cl_L_per_h=45.0, vd_L=700.0)
    concs = result.c_plasma_mg_L
    peak_idx = concs.index(max(concs))
    # Last value should be less than peak
    assert concs[-1] < concs[peak_idx]


def test_cytotoxic_notes_nonempty():
    result = _make_result()
    assert len(result.notes) > 0


def test_cytotoxic_one_cycle():
    result = _make_result(n_cycles=1, interval_h=24.0)
    assert len(result.cmax_per_cycle) >= 1
    assert result.cumulative_auc > 0


# ---------------------------------------------------------------------------
# Tests: targeted oral simulation
# ---------------------------------------------------------------------------


def test_targeted_dose_fixed():
    """For targeted drugs, dose_mg = dose_mg_m2 (ignores BSA)."""
    result = simulate_cancer_drug_pk(
        "imatinib",
        "targeted",
        dose_mg_m2=400.0,
        bsa_m2=1.8,
        cl_L_per_h=15.0,
        vd_L=300.0,
        interval_h=24.0,
        n_cycles=3,
        dt_h=0.5,
    )
    assert abs(result.dose_mg - 400.0) < 1e-6


def test_targeted_oral_route():
    result = simulate_cancer_drug_pk(
        "erlotinib",
        "targeted",
        dose_mg_m2=150.0,
        bsa_m2=1.7,
        cl_L_per_h=3.0,
        vd_L=100.0,
        interval_h=24.0,
        n_cycles=2,
        dt_h=0.5,
        ka_per_h=0.5,
    )
    assert result.drug_class == "targeted"
    assert result.cmax_per_cycle[0] > 0


def test_targeted_cmax_per_cycle_length():
    result = simulate_cancer_drug_pk(
        "imatinib",
        "targeted",
        dose_mg_m2=400.0,
        bsa_m2=1.8,
        cl_L_per_h=15.0,
        vd_L=300.0,
        interval_h=24.0,
        n_cycles=5,
        dt_h=0.5,
    )
    assert len(result.cmax_per_cycle) == 5


def test_targeted_auc_positive():
    result = simulate_cancer_drug_pk(
        "imatinib",
        "targeted",
        dose_mg_m2=400.0,
        bsa_m2=1.8,
        cl_L_per_h=15.0,
        vd_L=300.0,
        interval_h=24.0,
        n_cycles=2,
        dt_h=0.5,
    )
    assert all(a > 0 for a in result.auc_per_cycle)


def test_targeted_bsa_ignored():
    """Changing BSA should not affect dose for targeted drugs."""
    r1 = simulate_cancer_drug_pk(
        "drug", "targeted", 400.0, 1.5, 15.0, 300.0, interval_h=24.0, n_cycles=1
    )
    r2 = simulate_cancer_drug_pk(
        "drug", "targeted", 400.0, 2.0, 15.0, 300.0, interval_h=24.0, n_cycles=1
    )
    assert abs(r1.dose_mg - r2.dose_mg) < 1e-9


# ---------------------------------------------------------------------------
# Tests: nadir_prediction
# ---------------------------------------------------------------------------


def test_nadir_basic():
    times = [0.0, 1.0, 2.0, 3.0]
    conc = [10.0, 8.0, 4.0, 2.0]
    result = nadir_prediction(conc, times, myelosuppression_ec50=50.0, emax_myelosuppression=100.0)
    assert "nadir_effect_pct" in result
    assert "risk_category" in result


def test_nadir_severe():
    """High AUC relative to EC50 should give severe risk."""
    times = list(range(100))
    conc = [100.0] * 100
    result = nadir_prediction(conc, times, myelosuppression_ec50=1.0, emax_myelosuppression=100.0)
    assert result["risk_category"] == "severe"
    assert result["nadir_effect_pct"] > 80.0


def test_nadir_mild():
    """Low AUC relative to EC50 should give mild risk."""
    times = [0.0, 1.0]
    conc = [0.1, 0.05]
    result = nadir_prediction(
        conc, times, myelosuppression_ec50=1000.0, emax_myelosuppression=100.0
    )
    assert result["risk_category"] == "mild"
    assert result["nadir_effect_pct"] < 50.0


def test_nadir_moderate():
    """Intermediate effect should give moderate risk."""
    times = [0.0, 1.0]
    conc = [50.0, 50.0]
    # AUC = 50, EC50 = 50 → effect = 100*50/100 = 50%
    result = nadir_prediction(conc, times, myelosuppression_ec50=50.0, emax_myelosuppression=100.0)
    assert result["risk_category"] == "moderate"


def test_nadir_validates_empty_conc():
    with pytest.raises(ValueError, match="c_plasma_mg_L_list"):
        nadir_prediction([], [1.0, 2.0], 10.0, 100.0)


def test_nadir_validates_empty_times():
    with pytest.raises(ValueError, match="times_h"):
        nadir_prediction([1.0, 2.0], [], 10.0, 100.0)


def test_nadir_validates_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        nadir_prediction([1.0, 2.0], [0.0], 10.0, 100.0)


def test_nadir_validates_ec50():
    with pytest.raises(ValueError, match="myelosuppression_ec50"):
        nadir_prediction([1.0], [0.0], 0.0, 100.0)


def test_nadir_validates_emax():
    with pytest.raises(ValueError, match="emax_myelosuppression"):
        nadir_prediction([1.0], [0.0], 10.0, 0.0)


def test_nadir_returns_cumulative_auc():
    times = [0.0, 2.0]
    conc = [4.0, 4.0]
    result = nadir_prediction(conc, times, myelosuppression_ec50=8.0, emax_myelosuppression=100.0)
    assert abs(result["cumulative_auc_mg_h_per_L"] - 8.0) < 1e-6


def test_cytotoxic_higher_dose_higher_cmax():
    """Doubling dose should double initial Cmax for cytotoxic IV."""
    r1 = _make_result(dose_mg_m2=60.0, bsa_m2=1.0, cl_L_per_h=45.0, vd_L=700.0, n_cycles=1)
    r2 = _make_result(dose_mg_m2=120.0, bsa_m2=1.0, cl_L_per_h=45.0, vd_L=700.0, n_cycles=1)
    ratio = r2.cmax_per_cycle[0] / r1.cmax_per_cycle[0]
    assert 1.5 < ratio < 2.5
