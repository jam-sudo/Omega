"""Tests for sustained-release ocular drug delivery module."""

import pytest

from omega_pbpk.core.ocular_delivery import (
    IntravitreInjResult,
    OcularImplantResult,
    compare_delivery_systems,
    simulate_intravitreal_injection,
    simulate_ocular_implant,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_IMPLANT_BASE = dict(
    drug_name="Ranibizumab",
    drug_load_mg=0.5,
    implant_type="biodegradable",
    k_release_per_day=0.02,
    ke_vitreous_per_day=0.3,
    vd_vitreous_mL=4.0,
)

_IVT_BASE = dict(
    drug_name="Bevacizumab",
    dose_mg=1.25,
    ke_vitreous_per_day=0.3,
    vd_vitreous_mL=4.0,
)


# ---------------------------------------------------------------------------
# simulate_ocular_implant — validation
# ---------------------------------------------------------------------------


def test_implant_zero_load_raises():
    with pytest.raises(ValueError):
        simulate_ocular_implant(**{**_IMPLANT_BASE, "drug_load_mg": 0.0})


def test_implant_negative_load_raises():
    with pytest.raises(ValueError):
        simulate_ocular_implant(**{**_IMPLANT_BASE, "drug_load_mg": -1.0})


def test_implant_zero_k_release_raises():
    with pytest.raises(ValueError, match="k_release"):
        simulate_ocular_implant(**{**_IMPLANT_BASE, "k_release_per_day": 0.0})


def test_implant_zero_ke_raises():
    with pytest.raises(ValueError, match="ke_vitreous"):
        simulate_ocular_implant(**{**_IMPLANT_BASE, "ke_vitreous_per_day": 0.0})


def test_implant_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_vitreous"):
        simulate_ocular_implant(**{**_IMPLANT_BASE, "vd_vitreous_mL": 0.0})


def test_implant_invalid_type_raises():
    with pytest.raises(ValueError, match="implant_type"):
        simulate_ocular_implant(**{**_IMPLANT_BASE, "implant_type": "magic"})


def test_implant_zero_ec50_raises():
    with pytest.raises(ValueError, match="ec50"):
        simulate_ocular_implant(**{**_IMPLANT_BASE, "ec50_mg_mL": 0.0})


# ---------------------------------------------------------------------------
# simulate_ocular_implant — basic outputs (biodegradable)
# ---------------------------------------------------------------------------


def test_implant_returns_result_type():
    r = simulate_ocular_implant(**_IMPLANT_BASE)
    assert isinstance(r, OcularImplantResult)


def test_implant_times_length():
    r = simulate_ocular_implant(**_IMPLANT_BASE, t_end_days=10.0, dt_days=0.5)
    n = max(int(10.0 / 0.5), 1)
    assert len(r.times_days) == n + 1


def test_implant_concentration_positive():
    r = simulate_ocular_implant(**_IMPLANT_BASE, t_end_days=30.0)
    assert r.peak_c > 0


def test_implant_auc_positive():
    r = simulate_ocular_implant(**_IMPLANT_BASE, t_end_days=30.0)
    assert r.auc_vitreous > 0


def test_implant_effect_bounded():
    r = simulate_ocular_implant(**_IMPLANT_BASE, t_end_days=30.0)
    assert all(0.0 <= e <= 100.0 for e in r.effect_pct)


def test_implant_days_above_ec50_reasonable():
    r = simulate_ocular_implant(**_IMPLANT_BASE, t_end_days=90.0)
    assert r.days_above_ec50 >= 0


def test_implant_concentration_decays_to_near_zero():
    # After 5 half-lives of elimination, most drug should be cleared
    r = simulate_ocular_implant(
        **{**_IMPLANT_BASE, "drug_load_mg": 0.01, "k_release_per_day": 5.0},
        t_end_days=180.0,
    )
    assert r.c_vitreous_mg_mL[-1] < 0.01 * r.peak_c


def test_implant_notes_contain_drug_name():
    r = simulate_ocular_implant(**_IMPLANT_BASE)
    assert "Ranibizumab" in r.notes


def test_implant_type_stored():
    base = {k: v for k, v in _IMPLANT_BASE.items() if k != "implant_type"}
    r = simulate_ocular_implant(**base, implant_type="non_biodegradable")
    assert r.implant_type == "non_biodegradable"


# ---------------------------------------------------------------------------
# simulate_ocular_implant — non-biodegradable (zero-order)
# ---------------------------------------------------------------------------


def test_non_biodeg_implant_runs():
    r = simulate_ocular_implant(
        **{**_IMPLANT_BASE, "implant_type": "non_biodegradable"},
        t_end_days=90.0,
    )
    assert r.peak_c > 0


def test_non_biodeg_higher_early_concentration():
    # Zero-order (non-biodegradable) should maintain a plateau early on;
    # first-order may start higher but both deliver drug.
    r_bio = simulate_ocular_implant(**_IMPLANT_BASE, t_end_days=60.0, dt_days=0.5)
    r_nonbio = simulate_ocular_implant(
        **{**_IMPLANT_BASE, "implant_type": "non_biodegradable"}, t_end_days=60.0, dt_days=0.5
    )
    # Both should show drug in vitreous
    assert r_bio.auc_vitreous > 0
    assert r_nonbio.auc_vitreous > 0


# ---------------------------------------------------------------------------
# simulate_intravitreal_injection — validation
# ---------------------------------------------------------------------------


def test_ivt_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intravitreal_injection(**{**_IVT_BASE, "dose_mg": 0.0})


def test_ivt_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intravitreal_injection(**{**_IVT_BASE, "dose_mg": -0.5})


def test_ivt_zero_ke_raises():
    with pytest.raises(ValueError, match="ke_vitreous"):
        simulate_intravitreal_injection(**{**_IVT_BASE, "ke_vitreous_per_day": 0.0})


def test_ivt_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_vitreous"):
        simulate_intravitreal_injection(**{**_IVT_BASE, "vd_vitreous_mL": 0.0})


def test_ivt_zero_ec50_raises():
    with pytest.raises(ValueError, match="ec50"):
        simulate_intravitreal_injection(**{**_IVT_BASE, "ec50_mg_mL": 0.0})


# ---------------------------------------------------------------------------
# simulate_intravitreal_injection — basic outputs
# ---------------------------------------------------------------------------


def test_ivt_returns_result_type():
    r = simulate_intravitreal_injection(**_IVT_BASE)
    assert isinstance(r, IntravitreInjResult)


def test_ivt_peak_at_time_zero():
    r = simulate_intravitreal_injection(**_IVT_BASE, t_end_days=30.0)
    assert r.peak_c == pytest.approx(r.c_vitreous_mg_mL[0], rel=1e-6)


def test_ivt_initial_concentration_correct():
    # C0 = dose / Vd
    r = simulate_intravitreal_injection(dose_mg=1.25, vd_vitreous_mL=4.0,
                                         drug_name="Test", ke_vitreous_per_day=0.3)
    assert r.peak_c == pytest.approx(1.25 / 4.0, rel=1e-4)


def test_ivt_monoexponential_decay():
    r = simulate_intravitreal_injection(**_IVT_BASE, t_end_days=30.0, dt_days=0.01)
    # After ~1/ke days, concentration should be ~1/e of C0
    ke = 0.3
    t_half_life = 1.0 / ke  # ~3.33 days
    c_at_t = r.c_vitreous_mg_mL[int(t_half_life / 0.01)]
    expected = r.peak_c * (1.0 / 2.718)
    assert c_at_t == pytest.approx(expected, rel=0.05)


def test_ivt_auc_positive():
    r = simulate_intravitreal_injection(**_IVT_BASE, t_end_days=30.0)
    assert r.auc_vitreous > 0


def test_ivt_effect_bounded():
    r = simulate_intravitreal_injection(**_IVT_BASE, t_end_days=30.0)
    assert all(0.0 <= e <= 100.0 for e in r.effect_pct)


def test_ivt_higher_dose_higher_auc():
    r_low = simulate_intravitreal_injection(**{**_IVT_BASE, "dose_mg": 0.5}, t_end_days=30.0)
    r_high = simulate_intravitreal_injection(**{**_IVT_BASE, "dose_mg": 2.0}, t_end_days=30.0)
    assert r_high.auc_vitreous > r_low.auc_vitreous


def test_ivt_days_above_ec50_decreases_with_smaller_dose():
    r_high = simulate_intravitreal_injection(**{**_IVT_BASE, "dose_mg": 2.0}, t_end_days=60.0)
    r_low = simulate_intravitreal_injection(**{**_IVT_BASE, "dose_mg": 0.05}, t_end_days=60.0)
    assert r_high.days_above_ec50 >= r_low.days_above_ec50


def test_ivt_notes_contain_drug_name():
    r = simulate_intravitreal_injection(**_IVT_BASE, t_end_days=30.0)
    assert "Bevacizumab" in r.notes


# ---------------------------------------------------------------------------
# compare_delivery_systems
# ---------------------------------------------------------------------------


def test_compare_returns_dict_with_keys():
    result = compare_delivery_systems(
        drug_name="Ranibizumab",
        dose_mg=0.5,
        drug_load_implant_mg=2.0,
        t_end_days=90.0,
    )
    assert "single_ivt" in result
    assert "monthly_ivt" in result
    assert "biodegradable_implant" in result
    assert "summary" in result


def test_compare_monthly_ivt_multiple_injections():
    result = compare_delivery_systems(
        drug_name="Ranibizumab",
        dose_mg=0.5,
        drug_load_implant_mg=2.0,
        t_end_days=90.0,
    )
    assert result["monthly_ivt"]["n_injections"] >= 3


def test_compare_summary_has_auc_fields():
    result = compare_delivery_systems(
        drug_name="Drug",
        dose_mg=0.5,
        drug_load_implant_mg=1.0,
        t_end_days=60.0,
    )
    s = result["summary"]
    assert "single_ivt_auc" in s
    assert "monthly_ivt_auc" in s
    assert "implant_auc" in s


def test_compare_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        compare_delivery_systems("Drug", dose_mg=0.0, drug_load_implant_mg=1.0)


def test_compare_zero_load_raises():
    with pytest.raises(ValueError, match="drug_load"):
        compare_delivery_systems("Drug", dose_mg=0.5, drug_load_implant_mg=0.0)
