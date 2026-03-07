"""
Tests for Phase 284 — pH-Partition Theory GI Absorption Predictor.
"""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.ph_partition_gi_absorption import (
    GIAbsorptionResult,
    compare_ionization_types,
    predict_gi_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(**kwargs) -> GIAbsorptionResult:
    defaults = dict(
        drug_name="TestDrug",
        pka=7.0,
        ionization_type="neutral",
        logp=1.5,
        mw=300.0,
        dose_mg=100.0,
        solubility_mg_mL_at_neutral=1.0,
    )
    defaults.update(kwargs)
    return predict_gi_absorption(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, GIAbsorptionResult)


def test_field_preservation():
    result = make_result(
        drug_name="Aspirin",
        pka=3.5,
        ionization_type="acid",
        logp=1.19,
        mw=180.2,
        dose_mg=500.0,
        solubility_mg_mL_at_neutral=3.0,
    )
    assert result.drug_name == "Aspirin"
    assert result.pka == 3.5
    assert result.ionization_type == "acid"
    assert result.logp == 1.19
    assert result.mw == 180.2
    assert result.dose_mg == 500.0


# ---------------------------------------------------------------------------
# f_abs in [0, 1]
# ---------------------------------------------------------------------------


def test_f_abs_in_range():
    result = make_result()
    assert 0.0 <= result.f_abs <= 1.0


def test_f_abs_neutral_drug():
    result = make_result(ionization_type="neutral")
    assert 0.0 <= result.f_abs <= 1.0


def test_f_abs_acid_drug():
    result = make_result(pka=4.0, ionization_type="acid", logp=2.0)
    assert 0.0 <= result.f_abs <= 1.0


def test_f_abs_base_drug():
    result = make_result(pka=9.0, ionization_type="base", logp=2.0)
    assert 0.0 <= result.f_abs <= 1.0


# ---------------------------------------------------------------------------
# Ionization-specific behavior
# ---------------------------------------------------------------------------


def test_acid_drug_lower_absorption_than_neutral_at_intestinal_ph():
    """Acid drug with low pKa is mostly ionized in intestine; neutral is not."""
    # Low pKa acid: ionized at intestinal pH (6.5-7.2), lowering effective perm
    acid = predict_gi_absorption(
        drug_name="AcidDrug",
        pka=3.0,
        ionization_type="acid",
        logp=2.0,
        mw=200.0,
        dose_mg=50.0,
        solubility_mg_mL_at_neutral=2.0,
    )
    neutral = predict_gi_absorption(
        drug_name="NeutralDrug",
        pka=3.0,
        ionization_type="neutral",
        logp=2.0,
        mw=200.0,
        dose_mg=50.0,
        solubility_mg_mL_at_neutral=2.0,
    )
    # Acid drug mostly ionized in intestine → lower permeability → lower f_abs
    assert acid.f_abs < neutral.f_abs


def test_base_drug_ionized_at_intestinal_ph_with_high_pka():
    """High pKa base is mostly ionized at intestinal pH."""
    high_pka_base = predict_gi_absorption(
        drug_name="HighPKaBase",
        pka=10.0,
        ionization_type="base",
        logp=2.0,
        mw=200.0,
        dose_mg=50.0,
        solubility_mg_mL_at_neutral=2.0,
    )
    neutral = predict_gi_absorption(
        drug_name="NeutralDrug2",
        pka=10.0,
        ionization_type="neutral",
        logp=2.0,
        mw=200.0,
        dose_mg=50.0,
        solubility_mg_mL_at_neutral=2.0,
    )
    # High pKa base: mostly ionized in intestine (major absorption site)
    assert high_pka_base.f_abs < neutral.f_abs


def test_high_logp_high_peff():
    """High logP should yield higher permeability."""
    low_logp = make_result(logp=-2.0, ionization_type="neutral")
    high_logp = make_result(logp=4.0, ionization_type="neutral")
    assert high_logp.peff_cm_per_s > low_logp.peff_cm_per_s


# ---------------------------------------------------------------------------
# Peff positive
# ---------------------------------------------------------------------------


def test_peff_positive():
    result = make_result()
    assert result.peff_cm_per_s > 0.0


def test_peff_within_caps():
    low = make_result(logp=-5.0)
    high = make_result(logp=10.0)
    assert low.peff_cm_per_s >= 1e-6
    assert high.peff_cm_per_s <= 1e-3


# ---------------------------------------------------------------------------
# BCS class
# ---------------------------------------------------------------------------


def test_bcs_class_valid_values():
    result = make_result()
    assert result.bcs_class in ("I", "II", "III", "IV")


def test_bcs_class_I_high_logp_high_sol():
    # High logP → high perm; high solubility
    result = make_result(logp=4.0, solubility_mg_mL_at_neutral=1.0, ionization_type="neutral")
    assert result.bcs_class == "I"


def test_bcs_class_III_low_logp_high_sol():
    # Low logP → low perm; high solubility
    result = make_result(logp=-3.0, solubility_mg_mL_at_neutral=1.0, ionization_type="neutral")
    assert result.bcs_class == "III"


def test_bcs_class_II_high_logp_low_sol():
    # High logP → high perm; low solubility
    result = make_result(logp=4.0, solubility_mg_mL_at_neutral=0.01, ionization_type="neutral")
    assert result.bcs_class == "II"


def test_bcs_class_IV_low_logp_low_sol():
    # Low logP → low perm; low solubility
    result = make_result(logp=-3.0, solubility_mg_mL_at_neutral=0.01, ionization_type="neutral")
    assert result.bcs_class == "IV"


# ---------------------------------------------------------------------------
# Absorption class
# ---------------------------------------------------------------------------


def test_absorption_class_valid_values():
    result = make_result()
    assert result.absorption_class in ("complete", "partial", "poor")


def test_absorption_class_complete():
    # High logP, good solubility, neutral → high f_abs
    result = make_result(logp=4.0, solubility_mg_mL_at_neutral=5.0, ionization_type="neutral")
    assert result.absorption_class == "complete"


def test_absorption_class_poor():
    # Low logP, low solubility, large dose → poor absorption
    result = make_result(
        logp=-5.0, solubility_mg_mL_at_neutral=0.001, dose_mg=1000.0, ionization_type="neutral"
    )
    assert result.absorption_class == "poor"


# ---------------------------------------------------------------------------
# Rate limiting factor
# ---------------------------------------------------------------------------


def test_rate_limiting_factor_valid_values():
    result = make_result()
    assert result.rate_limiting_factor in ("permeability", "solubility", "neither")


def test_rate_limiting_factor_neither_bcs_I():
    result = make_result(logp=4.0, solubility_mg_mL_at_neutral=1.0, ionization_type="neutral")
    assert result.rate_limiting_factor == "neither"


def test_rate_limiting_factor_permeability_bcs_III():
    result = make_result(logp=-3.0, solubility_mg_mL_at_neutral=1.0, ionization_type="neutral")
    assert result.rate_limiting_factor == "permeability"


def test_rate_limiting_factor_solubility_bcs_II():
    result = make_result(logp=4.0, solubility_mg_mL_at_neutral=0.01, ionization_type="neutral")
    assert result.rate_limiting_factor == "solubility"


# ---------------------------------------------------------------------------
# Segmental fa
# ---------------------------------------------------------------------------


def test_segmental_fa_length():
    result = make_result()
    assert len(result.segmental_fa) == 5


def test_segmental_fa_all_in_range():
    result = make_result()
    for fa in result.segmental_fa:
        assert 0.0 <= fa <= 1.0


# ---------------------------------------------------------------------------
# compare_ionization_types
# ---------------------------------------------------------------------------


def test_compare_ionization_types_returns_three():
    results = compare_ionization_types(
        drug_name="TestDrug",
        pka=7.0,
        logp=2.0,
        mw=300.0,
        dose_mg=100.0,
        solubility_mg_mL_at_neutral=1.0,
    )
    assert len(results) == 3


def test_compare_ionization_types_sorted_descending():
    results = compare_ionization_types(
        drug_name="TestDrug",
        pka=5.0,
        logp=2.0,
        mw=300.0,
        dose_mg=100.0,
        solubility_mg_mL_at_neutral=1.0,
    )
    for i in range(len(results) - 1):
        assert results[i].f_abs >= results[i + 1].f_abs


def test_compare_ionization_types_all_three_types():
    results = compare_ionization_types(
        drug_name="TestDrug",
        pka=7.0,
        logp=2.0,
        mw=300.0,
        dose_mg=100.0,
        solubility_mg_mL_at_neutral=1.0,
    )
    types = {r.ionization_type for r in results}
    assert types == {"acid", "base", "neutral"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_pka_below_zero():
    with pytest.raises(ValueError, match="pka"):
        make_result(pka=-1.0)


def test_invalid_pka_above_14():
    with pytest.raises(ValueError, match="pka"):
        make_result(pka=15.0)


def test_invalid_logp_below_minus5():
    with pytest.raises(ValueError, match="logp"):
        make_result(logp=-6.0)


def test_invalid_logp_above_10():
    with pytest.raises(ValueError, match="logp"):
        make_result(logp=11.0)


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        make_result(mw=0.0)


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        make_result(dose_mg=0.0)


def test_invalid_solubility_zero():
    with pytest.raises(ValueError, match="solubility"):
        make_result(solubility_mg_mL_at_neutral=0.0)


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        make_result(ionization_type="zwitterion")
