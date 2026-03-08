"""Tests for Phase 480 — Skin pH Effect on Drug Absorption."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.skin_ph_absorption_p480 import (
    SkinPhAbsorptionResult,
    compare_body_sites,
    predict_skin_ph_absorption,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_skin_ph_absorption("TestDrug", "arm", mw_Da=300.0, logP=2.0)
    assert isinstance(result, SkinPhAbsorptionResult)


# ---------------------------------------------------------------------------
# Neutral drug
# ---------------------------------------------------------------------------


def test_neutral_drug_kp_effective_equals_kp_neutral():
    result = predict_skin_ph_absorption(
        "Neutral", "arm", mw_Da=300.0, logP=2.0, ionization_type="neutral"
    )
    assert result.kp_effective_cm_per_h == pytest.approx(result.kp_neutral_cm_per_h, rel=1e-6)


def test_neutral_drug_fraction_unionized_is_one():
    result = predict_skin_ph_absorption(
        "Neutral", "arm", mw_Da=300.0, logP=2.0, ionization_type="neutral"
    )
    assert result.fraction_unionized == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Acid drug ionization
# ---------------------------------------------------------------------------


def test_acid_drug_below_pka_more_unionized():
    # At skin pH 4.2 (foot) with pKa=7.0: most of drug is unionized (pH << pKa)
    result_foot = predict_skin_ph_absorption(
        "AcidDrug", "foot", mw_Da=250.0, logP=1.5, pKa=7.0, ionization_type="acid"
    )
    result_groin = predict_skin_ph_absorption(
        "AcidDrug", "groin", mw_Da=250.0, logP=1.5, pKa=7.0, ionization_type="acid"
    )
    # foot pH=4.2 < pKa=7 → more unionized than groin pH=6.5
    assert result_foot.fraction_unionized > result_groin.fraction_unionized


def test_acid_drug_acidic_ph_higher_kp():
    # foot (pH=4.2) vs groin (pH=6.5) for acid pKa=7
    result_foot = predict_skin_ph_absorption(
        "AcidDrug", "foot", mw_Da=250.0, logP=1.5, pKa=7.0, ionization_type="acid"
    )
    result_groin = predict_skin_ph_absorption(
        "AcidDrug", "groin", mw_Da=250.0, logP=1.5, pKa=7.0, ionization_type="acid"
    )
    assert result_foot.kp_effective_cm_per_h > result_groin.kp_effective_cm_per_h


def test_acid_drug_fraction_unionized_range():
    result = predict_skin_ph_absorption(
        "AcidDrug", "forehead", mw_Da=300.0, logP=2.0, pKa=5.5, ionization_type="acid"
    )
    assert 0.0 <= result.fraction_unionized <= 1.0


# ---------------------------------------------------------------------------
# Base drug at acidic skin pH
# ---------------------------------------------------------------------------


def test_base_drug_acidic_ph_more_ionized():
    # Basic drug at acidic skin (foot pH=4.2) is mostly ionized (pKa=7 >> pH=4.2)
    result_foot = predict_skin_ph_absorption(
        "BaseDrug", "foot", mw_Da=250.0, logP=2.0, pKa=7.0, ionization_type="base"
    )
    # fraction_unionized should be low (most is ionized BH+)
    assert result_foot.fraction_unionized < 0.1


def test_base_drug_lower_kp_than_neutral():
    # Base drug at acidic skin has lower Kp than neutral drug (same parameters)
    result_base = predict_skin_ph_absorption(
        "BaseDrug", "arm", mw_Da=300.0, logP=2.0, pKa=8.0, ionization_type="base"
    )
    result_neutral = predict_skin_ph_absorption(
        "NeutralDrug", "arm", mw_Da=300.0, logP=2.0, ionization_type="neutral"
    )
    # At arm pH=5.0, pKa=8 base is mostly ionized → lower Kp
    assert result_base.kp_effective_cm_per_h < result_neutral.kp_effective_cm_per_h


# ---------------------------------------------------------------------------
# Permeation rate
# ---------------------------------------------------------------------------


def test_permeation_rate_positive():
    result = predict_skin_ph_absorption("Drug", "arm", mw_Da=300.0, logP=2.0)
    assert result.permeation_rate_mg_per_h > 0


def test_permeation_rate_equals_flux_times_area():
    result = predict_skin_ph_absorption(
        "Drug", "arm", mw_Da=300.0, logP=2.0, dose_conc_mg_per_mL=5.0, area_cm2=20.0
    )
    expected = result.flux_mg_per_cm2_per_h * 20.0
    assert result.permeation_rate_mg_per_h == pytest.approx(expected, rel=1e-6)


def test_permeation_rate_scales_with_area():
    r1 = predict_skin_ph_absorption("Drug", "arm", mw_Da=300.0, logP=2.0, area_cm2=10.0)
    r2 = predict_skin_ph_absorption("Drug", "arm", mw_Da=300.0, logP=2.0, area_cm2=20.0)
    assert r2.permeation_rate_mg_per_h == pytest.approx(2 * r1.permeation_rate_mg_per_h, rel=1e-6)


# ---------------------------------------------------------------------------
# Kp clamping
# ---------------------------------------------------------------------------


def test_kp_effective_within_bounds():
    result = predict_skin_ph_absorption(
        "Drug", "arm", mw_Da=500.0, logP=-2.0, pKa=7.0, ionization_type="base"
    )
    assert result.kp_effective_cm_per_h >= 1e-7
    assert result.kp_effective_cm_per_h <= 1.0


def test_kp_neutral_positive():
    result = predict_skin_ph_absorption("Drug", "arm", mw_Da=200.0, logP=3.0)
    assert result.kp_neutral_cm_per_h > 0


# ---------------------------------------------------------------------------
# permeation_enhancement_vs_arm
# ---------------------------------------------------------------------------


def test_arm_site_enhancement_is_one():
    result = predict_skin_ph_absorption("Drug", "arm", mw_Da=300.0, logP=2.0)
    assert result.permeation_enhancement_vs_arm == pytest.approx(1.0, rel=1e-6)


def test_back_site_enhancement_same_as_arm_neutral():
    # back and arm have same pH=5.0 → same enhancement for neutral
    result = predict_skin_ph_absorption(
        "Drug", "back", mw_Da=300.0, logP=2.0, ionization_type="neutral"
    )
    assert result.permeation_enhancement_vs_arm == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_body_sites
# ---------------------------------------------------------------------------


def test_compare_body_sites_returns_six():
    results = compare_body_sites("Drug", mw_Da=300.0, logP=2.0)
    assert len(results) == 6


def test_compare_body_sites_sorted_descending():
    results = compare_body_sites("Drug", mw_Da=300.0, logP=2.0, ionization_type="neutral")
    rates = [r.permeation_rate_mg_per_h for r in results]
    assert rates == sorted(rates, reverse=True)


def test_compare_body_sites_all_sites_present():
    results = compare_body_sites("Drug", mw_Da=300.0, logP=2.0)
    sites = {r.body_site for r in results}
    expected = {"forehead", "arm", "scalp", "foot", "groin", "back"}
    assert sites == expected


def test_compare_body_sites_acid_drug_sorted():
    results = compare_body_sites("AcidDrug", mw_Da=250.0, logP=2.0, pKa=6.0, ionization_type="acid")
    rates = [r.permeation_rate_mg_per_h for r in results]
    assert rates == sorted(rates, reverse=True)


def test_compare_body_sites_neutral_all_same_rate():
    # For neutral drug all sites have same Kp → same rate (area=10 for all)
    results = compare_body_sites("Neutral", mw_Da=300.0, logP=2.0, ionization_type="neutral")
    rates = [r.permeation_rate_mg_per_h for r in results]
    # All rates should be approximately equal
    assert max(rates) == pytest.approx(min(rates), rel=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_body_site():
    with pytest.raises(ValueError, match="body_site"):
        predict_skin_ph_absorption("Drug", "elbow", mw_Da=300.0, logP=2.0)


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_skin_ph_absorption("Drug", "arm", mw_Da=0.0, logP=2.0)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_skin_ph_absorption("Drug", "arm", mw_Da=-100.0, logP=2.0)


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_skin_ph_absorption(
            "Drug", "arm", mw_Da=300.0, logP=2.0, ionization_type="zwitterion"
        )


def test_invalid_dose_conc_zero():
    with pytest.raises(ValueError, match="dose_conc_mg_per_mL"):
        predict_skin_ph_absorption("Drug", "arm", mw_Da=300.0, logP=2.0, dose_conc_mg_per_mL=0.0)


def test_invalid_area_zero():
    with pytest.raises(ValueError, match="area_cm2"):
        predict_skin_ph_absorption("Drug", "arm", mw_Da=300.0, logP=2.0, area_cm2=0.0)
