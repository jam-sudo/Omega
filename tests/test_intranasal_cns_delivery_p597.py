"""Tests for Phase 597 — Intranasal CNS Drug Delivery."""

import pytest

from omega_pbpk.core.intranasal_cns_delivery_p597 import (
    IntranasalCNSDeliveryResult,
    simulate_intranasal_cns_delivery,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_default(**kwargs):
    """Run simulation with sensible defaults, override with kwargs."""
    params = dict(
        drug_name="TestDrug",
        dose_mg=5.0,
        logP=2.5,
        mw_Da=300.0,
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
        cl_brain_per_h=0.1,
        t_end_h=6.0,
        dt_h=0.02,
    )
    params.update(kwargs)
    return simulate_intranasal_cns_delivery(**params)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _run_default()
    assert isinstance(result, IntranasalCNSDeliveryResult)


# ---------------------------------------------------------------------------
# 2. List lengths are consistent
# ---------------------------------------------------------------------------


def test_list_lengths_equal():
    result = _run_default()
    n = len(result.times_h)
    assert len(result.c_nasal_mg_mL) == n
    assert len(result.c_olfactory_mg_mL) == n
    assert len(result.c_brain_mg_mL) == n
    assert len(result.c_systemic_mg_L) == n


def test_list_length_matches_steps():
    result = _run_default(t_end_h=4.0, dt_h=0.1)
    # int(round(4.0/0.1)) + 1 = 41
    assert len(result.times_h) == 41


# ---------------------------------------------------------------------------
# 3. Time axis starts at zero and is monotonically increasing
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    result = _run_default()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_monotonically_increasing():
    result = _run_default()
    for i in range(1, len(result.times_h)):
        assert result.times_h[i] > result.times_h[i - 1]


# ---------------------------------------------------------------------------
# 4. Concentrations are non-negative
# ---------------------------------------------------------------------------


def test_concentrations_non_negative():
    result = _run_default()
    assert all(c >= 0.0 for c in result.c_nasal_mg_mL)
    assert all(c >= 0.0 for c in result.c_olfactory_mg_mL)
    assert all(c >= 0.0 for c in result.c_brain_mg_mL)
    assert all(c >= 0.0 for c in result.c_systemic_mg_L)


# ---------------------------------------------------------------------------
# 5. Initial nasal concentration equals dose / nasal_vol
# ---------------------------------------------------------------------------


def test_initial_nasal_concentration():
    dose = 5.0
    result = _run_default(dose_mg=dose)
    # nasal_vol = 0.5 mL
    assert result.c_nasal_mg_mL[0] == pytest.approx(dose / 0.5)


# ---------------------------------------------------------------------------
# 6. Nasal concentration decreases over time
# ---------------------------------------------------------------------------


def test_nasal_concentration_decreases():
    result = _run_default()
    # Should be monotonically decreasing (exponential clearance)
    assert result.c_nasal_mg_mL[-1] < result.c_nasal_mg_mL[0]


# ---------------------------------------------------------------------------
# 7. Brain AUC > 0
# ---------------------------------------------------------------------------


def test_brain_auc_positive():
    result = _run_default()
    assert result.auc_brain_mg_h_per_mL > 0.0


# ---------------------------------------------------------------------------
# 8. Systemic AUC > 0
# ---------------------------------------------------------------------------


def test_systemic_auc_positive():
    result = _run_default()
    assert result.auc_systemic_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 9. Nose-to-brain ratio > 0
# ---------------------------------------------------------------------------


def test_nose_to_brain_ratio_positive():
    result = _run_default()
    assert result.nose_to_brain_ratio > 0.0


# ---------------------------------------------------------------------------
# 10. Cmax brain > 0
# ---------------------------------------------------------------------------


def test_cmax_brain_positive():
    result = _run_default()
    assert result.cmax_brain_mg_mL > 0.0


# ---------------------------------------------------------------------------
# 11. Tmax brain is within simulation window
# ---------------------------------------------------------------------------


def test_tmax_brain_in_window():
    t_end = 6.0
    result = _run_default(t_end_h=t_end)
    assert 0.0 <= result.tmax_brain_h <= t_end


# ---------------------------------------------------------------------------
# 12. High logP drug has higher CNS delivery than low logP
# ---------------------------------------------------------------------------


def test_high_logP_better_cns_delivery():
    high = _run_default(logP=4.0, mw_Da=250.0)
    low = _run_default(logP=0.5, mw_Da=250.0)
    assert high.auc_brain_mg_h_per_mL > low.auc_brain_mg_h_per_mL


# ---------------------------------------------------------------------------
# 13. High logP leads to higher nose-to-brain ratio
# ---------------------------------------------------------------------------


def test_high_logP_higher_nose_to_brain_ratio():
    high = _run_default(logP=4.0, mw_Da=250.0)
    low = _run_default(logP=0.5, mw_Da=250.0)
    assert high.nose_to_brain_ratio > low.nose_to_brain_ratio


# ---------------------------------------------------------------------------
# 14. Large MW reduces CNS penetration via olfactory route
# ---------------------------------------------------------------------------


def test_large_mw_reduces_cns_delivery():
    small = _run_default(logP=2.5, mw_Da=150.0)
    large = _run_default(logP=2.5, mw_Da=800.0)
    assert small.auc_brain_mg_h_per_mL > large.auc_brain_mg_h_per_mL


# ---------------------------------------------------------------------------
# 15. Direct CNS fraction is between 0 and 1
# ---------------------------------------------------------------------------


def test_direct_cns_fraction_in_range():
    result = _run_default()
    assert 0.0 < result.direct_cns_fraction < 1.0


# ---------------------------------------------------------------------------
# 16. Direct CNS fraction formula: k_olf / (k_olf + k_sys_abs)
# ---------------------------------------------------------------------------


def test_direct_cns_fraction_formula():
    # k_sys_abs = 0.3, logP=2.5, mw_Da=300 => k_olf = 0.2*2.5/(300/200)=0.5/1.5=0.333...
    # clamped to [0.01, 0.5] => 0.333
    result = _run_default(logP=2.5, mw_Da=300.0)
    # Compute expected
    logP_eff = max(0.1, 2.5)
    mw_eff = max(1.0, 300.0 / 200.0)
    k_olf = 0.2 * logP_eff / mw_eff
    k_olf = max(0.01, min(0.5, k_olf))
    k_sys_abs = 0.3
    expected = k_olf / (k_olf + k_sys_abs)
    assert result.direct_cns_fraction == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 17. Brain bioavailability is between 0 and 100
# ---------------------------------------------------------------------------


def test_brain_bioavailability_pct_in_range():
    result = _run_default()
    assert 0.0 <= result.brain_bioavailability_pct <= 100.0


# ---------------------------------------------------------------------------
# 18. Drug name stored correctly
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    result = _run_default(drug_name="FentanylAnalog")
    assert result.drug_name == "FentanylAnalog"


# ---------------------------------------------------------------------------
# 19. Dose stored correctly
# ---------------------------------------------------------------------------


def test_dose_stored():
    result = _run_default(dose_mg=2.0)
    assert result.dose_mg == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 20. Higher dose leads to proportionally higher AUC
# ---------------------------------------------------------------------------


def test_dose_proportionality():
    r1 = _run_default(dose_mg=1.0)
    r2 = _run_default(dose_mg=2.0)
    ratio = r2.auc_brain_mg_h_per_mL / r1.auc_brain_mg_h_per_mL
    assert ratio == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# 21. Validation: dose <= 0 raises
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _run_default(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _run_default(dose_mg=-1.0)


# ---------------------------------------------------------------------------
# 22. Validation: cl_sys <= 0 raises
# ---------------------------------------------------------------------------


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        _run_default(cl_sys_L_per_h=0.0)


# ---------------------------------------------------------------------------
# 23. Validation: vd_sys <= 0 raises
# ---------------------------------------------------------------------------


def test_validation_vd_sys_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        _run_default(vd_sys_L=0.0)


# ---------------------------------------------------------------------------
# 24. Notes field is a non-empty string
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = _run_default()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 25. Negative logP still works (clamped)
# ---------------------------------------------------------------------------


def test_negative_logP_works():
    result = _run_default(logP=-1.0)
    assert result.auc_brain_mg_h_per_mL > 0.0
    assert result.direct_cns_fraction > 0.0
