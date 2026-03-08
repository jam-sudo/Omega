"""Tests for gut microbiome drug metabolism (Phase 559)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.gut_microbiome_metabolism_p559 import (
    GutMicrobiomeMetabolismResult,
    compare_drug_classes,
    predict_microbiome_metabolism,
)

# ---------- Return type ----------


def test_return_type():
    r = predict_microbiome_metabolism("TestDrug", "other", 100.0)
    assert isinstance(r, GutMicrobiomeMetabolismResult)


def test_result_is_frozen():
    r = predict_microbiome_metabolism("TestDrug", "other", 100.0)
    with pytest.raises(AttributeError):
        r.drug_name = "Changed"


# ---------- fraction_metabolized in [0, 1] ----------


def test_fraction_metabolized_in_range():
    r = predict_microbiome_metabolism("D", "other", 50.0)
    assert 0.0 <= r.fraction_metabolized_colon <= 1.0


def test_fraction_metabolized_high_cl():
    r = predict_microbiome_metabolism(
        "D", "other", 50.0, intrinsic_clearance_microbiome_mL_per_min=100.0
    )
    assert 0.0 <= r.fraction_metabolized_colon <= 1.0
    assert r.fraction_metabolized_colon > 0.9


def test_fraction_metabolized_low_cl():
    r = predict_microbiome_metabolism(
        "D", "other", 50.0, intrinsic_clearance_microbiome_mL_per_min=0.001
    )
    assert r.fraction_metabolized_colon < 0.01


# ---------- fa_effective in [0, 1] ----------


def test_fa_effective_in_range_prodrug():
    r = predict_microbiome_metabolism("D", "prodrug_azo", 100.0)
    assert 0.0 <= r.fa_effective <= 1.0


def test_fa_effective_in_range_other():
    r = predict_microbiome_metabolism("D", "other", 100.0)
    assert 0.0 <= r.fa_effective <= 1.0


# ---------- Prodrug activation ----------


def test_prodrug_azo_activation():
    r = predict_microbiome_metabolism("Sulfasalazine", "prodrug_azo", 500.0)
    assert r.microbiome_effect == "activation"
    assert r.active_metabolite_fraction == r.fraction_metabolized_colon
    assert r.fa_effective == r.active_metabolite_fraction


def test_prodrug_nitro_activation():
    r = predict_microbiome_metabolism("NitroDrug", "prodrug_nitro", 200.0)
    assert r.microbiome_effect == "activation"
    assert r.active_metabolite_fraction > 0.0


def test_glucuronide_activation():
    r = predict_microbiome_metabolism("GlucDrug", "glucuronide_conjugate", 100.0)
    assert r.microbiome_effect == "activation"
    assert r.active_metabolite_fraction == r.fraction_metabolized_colon


def test_other_inactivation():
    r = predict_microbiome_metabolism("OtherDrug", "other", 100.0)
    assert r.microbiome_effect == "inactivation"
    assert r.active_metabolite_fraction == 0.0
    assert r.fa_effective == pytest.approx(1.0 - r.fraction_metabolized_colon)


# ---------- Higher CLint → higher fraction metabolized ----------


def test_higher_clint_higher_metabolism():
    r_low = predict_microbiome_metabolism(
        "D", "other", 100.0, intrinsic_clearance_microbiome_mL_per_min=0.5
    )
    r_high = predict_microbiome_metabolism(
        "D", "other", 100.0, intrinsic_clearance_microbiome_mL_per_min=5.0
    )
    assert r_high.fraction_metabolized_colon > r_low.fraction_metabolized_colon


def test_higher_transit_time_higher_metabolism():
    r_short = predict_microbiome_metabolism("D", "other", 100.0, gut_transit_time_h=2.0)
    r_long = predict_microbiome_metabolism("D", "other", 100.0, gut_transit_time_h=12.0)
    assert r_long.fraction_metabolized_colon > r_short.fraction_metabolized_colon


# ---------- Drug class effects ----------


def test_prodrug_fa_equals_active_fraction():
    for dc in ("prodrug_azo", "prodrug_nitro", "glucuronide_conjugate"):
        r = predict_microbiome_metabolism("D", dc, 100.0)
        assert r.fa_effective == r.active_metabolite_fraction


def test_other_fa_complement_of_metabolized():
    r = predict_microbiome_metabolism("D", "other", 100.0)
    assert r.fa_effective == pytest.approx(1.0 - r.fraction_metabolized_colon)


# ---------- Compare function ----------


def test_compare_returns_list_of_4():
    results = compare_drug_classes("D", 100.0)
    assert isinstance(results, list)
    assert len(results) == 4


def test_compare_sorted_by_fa_effective_desc():
    results = compare_drug_classes("D", 100.0)
    for i in range(1, len(results)):
        assert results[i - 1].fa_effective >= results[i].fa_effective


def test_compare_all_drug_classes_present():
    results = compare_drug_classes("D", 100.0)
    classes = {r.drug_class for r in results}
    assert classes == {"prodrug_azo", "prodrug_nitro", "glucuronide_conjugate", "other"}


# ---------- Validation errors ----------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_microbiome_metabolism("D", "other", 0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_microbiome_metabolism("D", "other", -10.0)


def test_invalid_clint_zero():
    with pytest.raises(ValueError, match="intrinsic_clearance"):
        predict_microbiome_metabolism(
            "D", "other", 100.0, intrinsic_clearance_microbiome_mL_per_min=0.0
        )


def test_invalid_transit_time():
    with pytest.raises(ValueError, match="gut_transit_time"):
        predict_microbiome_metabolism("D", "other", 100.0, gut_transit_time_h=-1.0)


def test_invalid_drug_class():
    with pytest.raises(ValueError, match="drug_class"):
        predict_microbiome_metabolism("D", "invalid_class", 100.0)
