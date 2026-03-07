"""Tests for Phase 710 — Integrated ADME property calculator."""

import pytest

from omega_pbpk.prediction.adme_property_integrator import (
    ADMEProfile,
    compare_adme_optimization,
    compute_adme_profile,
    screen_adme,
)

# --- Helpers ---

IDEAL_PHYS = {"logP": 2.0, "mw": 300.0, "psa": 50.0, "n_h_donors": 1, "n_h_acceptors": 3}
PROBLEMATIC_PHYS = {
    "logP": 7.0,
    "mw": 600.0,
    "psa": 150.0,
    "n_h_donors": 6,
    "n_h_acceptors": 12,
}


def make_profile(logP=2.0, mw=300.0, psa=80.0, n_h_donors=1, n_h_acceptors=3, **kwargs):
    phys = {
        "logP": logP,
        "mw": mw,
        "psa": psa,
        "n_h_donors": n_h_donors,
        "n_h_acceptors": n_h_acceptors,
        **kwargs,
    }
    return compute_adme_profile("Compound", {}, phys)


# --- Return type ---


def test_returns_adme_profile():
    result = make_profile()
    assert isinstance(result, ADMEProfile)


# --- Absorption ---


def test_fa_in_range():
    result = make_profile()
    assert 0.0 < result.fa_predicted < 1.0


def test_high_psa_lowers_fa():
    low_psa = make_profile(psa=30.0)
    high_psa = make_profile(psa=180.0)
    assert high_psa.fa_predicted < low_psa.fa_predicted


# --- Distribution ---


def test_high_logP_higher_vd():
    low_logP = make_profile(logP=0.5)
    high_logP = make_profile(logP=5.0)
    assert high_logP.vd_L_estimated > low_logP.vd_L_estimated


def test_vd_positive():
    result = make_profile()
    assert result.vd_L_estimated > 0.0


# --- Lipinski ---


def test_lipinski_pass_for_drug_like():
    result = compute_adme_profile(
        "GoodDrug", {}, {"logP": 2.0, "mw": 300.0, "psa": 50.0, "n_h_donors": 1, "n_h_acceptors": 3}
    )
    assert result.lipinski_pass is True


def test_lipinski_fail_for_high_mw():
    result = compute_adme_profile(
        "BigDrug", {}, {"logP": 2.0, "mw": 600.0, "psa": 50.0, "n_h_donors": 1, "n_h_acceptors": 3}
    )
    assert result.lipinski_pass is False


# --- F_oral ---


def test_f_oral_in_range():
    result = make_profile()
    assert 0.0 < result.f_oral_estimated < 1.0


# --- t½ ---


def test_t_half_positive():
    result = make_profile()
    assert result.t_half_h_estimated > 0.0


# --- fe_urine ---


def test_fe_urine_in_range():
    result = make_profile()
    assert 0.0 <= result.fe_urine_estimated <= 1.0


def test_low_logP_high_fe_urine():
    result = make_profile(logP=-1.0)
    assert result.fe_urine_estimated == pytest.approx(0.8)


def test_high_logP_low_fe_urine():
    result = make_profile(logP=4.0)
    assert result.fe_urine_estimated == pytest.approx(0.1)


# --- Overall developability ---


def test_developability_valid_values():
    result = make_profile()
    assert result.overall_developability in {"excellent", "good", "moderate", "poor"}


def test_ideal_drug_good_or_excellent_developability():
    result = compute_adme_profile("Ideal", {}, IDEAL_PHYS)
    assert result.overall_developability in {"excellent", "good"}


def test_problematic_drug_poor_or_moderate_developability():
    result = compute_adme_profile("Problematic", {}, PROBLEMATIC_PHYS)
    assert result.overall_developability in {"poor", "moderate"}


# --- Notes ---


def test_notes_nonempty():
    result = make_profile()
    assert len(result.notes) > 0


# --- screen_adme ---


def test_screen_adme_sorted_descending():
    compounds = [
        {"compound_name": "A", "physicochemical": {"logP": 5.0, "mw": 490.0, "psa": 80.0}},
        {"compound_name": "B", "physicochemical": {"logP": 2.0, "mw": 300.0, "psa": 50.0}},
        {"compound_name": "C", "physicochemical": {"logP": 7.0, "mw": 700.0, "psa": 200.0}},
    ]
    results = screen_adme(compounds)
    for i in range(len(results) - 1):
        assert results[i].f_oral_estimated >= results[i + 1].f_oral_estimated


def test_screen_adme_returns_all():
    compounds = [
        {"compound_name": "A", "physicochemical": {"logP": 2.0, "mw": 300.0}},
        {"compound_name": "B", "physicochemical": {"logP": 3.0, "mw": 400.0}},
    ]
    results = screen_adme(compounds)
    assert len(results) == 2


# --- compare_adme_optimization ---


def test_compare_adme_base_first():
    base = {"compound_name": "Base", "physicochemical": {"logP": 2.0, "mw": 300.0}}
    variants = [
        {"compound_name": "V1", "physicochemical": {"logP": 3.0, "mw": 300.0}},
        {"compound_name": "V2", "physicochemical": {"logP": 4.0, "mw": 300.0}},
    ]
    results = compare_adme_optimization(base, variants)
    assert results[0].compound_name == "Base"


def test_compare_adme_returns_all():
    base = {"compound_name": "Base", "physicochemical": {"logP": 2.0, "mw": 300.0}}
    variants = [
        {"compound_name": "V1", "physicochemical": {"logP": 3.0, "mw": 300.0}},
    ]
    results = compare_adme_optimization(base, variants)
    assert len(results) == 2


# --- Validation ---


def test_zero_mw_raises():
    with pytest.raises(ValueError):
        compute_adme_profile("Bad", {}, {"logP": 2.0, "mw": 0.0})


def test_negative_mw_raises():
    with pytest.raises(ValueError):
        compute_adme_profile("Bad", {}, {"logP": 2.0, "mw": -100.0})


def test_negative_psa_raises():
    with pytest.raises(ValueError):
        compute_adme_profile("Bad", {}, {"logP": 2.0, "mw": 300.0, "psa": -1.0})
