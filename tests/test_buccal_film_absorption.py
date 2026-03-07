"""Tests for Phase 1060 — Drug Buccal Film Absorption Predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.buccal_film_absorption import (
    BuccalFilmResult,
    compare_film_types,
    predict_buccal_film_absorption,
)

_ALL_FILM_TYPES = {"HPMC", "PVA", "chitosan", "Eudragit"}

# --- helpers ----------------------------------------------------------------


def _base(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        mw_Da=250.0,
        logp=2.0,
        drug_loading_mg=5.0,
    )
    defaults.update(kwargs)
    return predict_buccal_film_absorption(**defaults)


# --- type and dataclass tests -----------------------------------------------


def test_return_type():
    result = _base()
    assert isinstance(result, BuccalFilmResult)


def test_frozen_dataclass():
    result = _base()
    with pytest.raises((AttributeError, TypeError)):
        result.fa_buccal = 0.5  # type: ignore[misc]


# --- field value range tests ------------------------------------------------


def test_fa_buccal_in_range():
    result = _base()
    assert 0 < result.fa_buccal <= 0.9


def test_f_systemic_in_range():
    result = _base()
    assert 0 <= result.f_systemic <= 1.0


def test_erosion_time_positive():
    result = _base()
    assert result.erosion_time_h > 0.0


def test_flux_positive():
    result = _base()
    assert result.flux_ug_cm2_h > 0.0


def test_mucoadhesion_score_positive():
    result = _base()
    assert result.mucoadhesion_score > 0.0


def test_permeation_enhancer_effect_ge_one():
    result = _base()
    assert result.permeation_enhancer_effect >= 1.0


def test_notes_nonempty():
    result = _base()
    assert len(result.notes) > 0


def test_tmax_positive():
    result = _base()
    assert result.tmax_h > 0.0


def test_tmax_half_erosion_time():
    """tmax should be erosion_time / 2."""
    result = _base()
    assert result.tmax_h == pytest.approx(result.erosion_time_h / 2.0)


# --- film type-specific tests -----------------------------------------------


def test_hpmc_film_type():
    result = _base(film_type="HPMC")
    assert result.film_type == "HPMC"
    assert result.mucoadhesion_score == pytest.approx(80.0)


def test_pva_film_type():
    result = _base(film_type="PVA")
    assert result.mucoadhesion_score == pytest.approx(60.0)


def test_chitosan_mucoadhesion():
    result = _base(film_type="chitosan")
    assert result.mucoadhesion_score == pytest.approx(90.0)


def test_chitosan_higher_mucoadhesion_than_pva():
    """Chitosan should have higher mucoadhesion score than PVA."""
    r_chitosan = _base(film_type="chitosan")
    r_pva = _base(film_type="PVA")
    assert r_chitosan.mucoadhesion_score > r_pva.mucoadhesion_score


def test_eudragit_film_type():
    result = _base(film_type="Eudragit")
    assert result.film_type == "Eudragit"
    assert result.mucoadhesion_score == pytest.approx(70.0)


def test_erosion_time_varies_by_film():
    """Different film types should have different erosion times."""
    r_hpmc = _base(film_type="HPMC")
    r_pva = _base(film_type="PVA")
    # HPMC k_erode=0.1 → longer erosion than PVA k_erode=0.2
    assert r_hpmc.erosion_time_h > r_pva.erosion_time_h


# --- enhancer tests ---------------------------------------------------------


def test_no_enhancer_effect_is_one():
    result = _base(permeation_enhancer="none")
    assert result.permeation_enhancer_effect == pytest.approx(1.0)


def test_sls_enhancer_higher_flux_than_none():
    """SLS enhancer should produce higher flux than no enhancer."""
    r_sls = _base(permeation_enhancer="sodium_lauryl_sulfate")
    r_none = _base(permeation_enhancer="none")
    assert r_sls.flux_ug_cm2_h > r_none.flux_ug_cm2_h


def test_oleic_acid_enhancer():
    result = _base(permeation_enhancer="oleic_acid")
    assert result.permeation_enhancer_effect == pytest.approx(2.5)


def test_menthol_enhancer():
    result = _base(permeation_enhancer="menthol")
    assert result.permeation_enhancer_effect == pytest.approx(1.5)


# --- compare_film_types tests -----------------------------------------------


def test_compare_film_types_returns_list():
    results = compare_film_types("TestDrug", 250.0, 2.0, 5.0)
    assert isinstance(results, list)


def test_compare_film_types_length():
    results = compare_film_types("TestDrug", 250.0, 2.0, 5.0)
    assert len(results) == 4


def test_compare_film_types_all_film_types_present():
    results = compare_film_types("TestDrug", 250.0, 2.0, 5.0)
    returned_types = {r.film_type for r in results}
    assert returned_types == _ALL_FILM_TYPES


def test_compare_film_types_sorted_desc():
    results = compare_film_types("TestDrug", 250.0, 2.0, 5.0)
    f_sys = [r.f_systemic for r in results]
    assert f_sys == sorted(f_sys, reverse=True)


# --- validation tests -------------------------------------------------------


def test_loading_zero_raises():
    with pytest.raises(ValueError, match="drug_loading_mg"):
        _base(drug_loading_mg=0.0)


def test_loading_negative_raises():
    with pytest.raises(ValueError, match="drug_loading_mg"):
        _base(drug_loading_mg=-1.0)


def test_invalid_film_type_raises():
    with pytest.raises(ValueError, match="film_type"):
        _base(film_type="cellulose")


def test_invalid_enhancer_raises():
    with pytest.raises(ValueError, match="permeation_enhancer"):
        _base(permeation_enhancer="ethanol")


def test_area_zero_raises():
    with pytest.raises(ValueError, match="film_area_cm2"):
        _base(film_area_cm2=0.0)


def test_area_negative_raises():
    with pytest.raises(ValueError, match="film_area_cm2"):
        _base(film_area_cm2=-1.0)


def test_thickness_zero_raises():
    with pytest.raises(ValueError, match="film_thickness_um"):
        _base(film_thickness_um=0.0)


def test_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        _base(ionization_type="amphoteric")


# --- field preservation tests -----------------------------------------------


def test_drug_name_preserved():
    result = _base(drug_name="Buprenorphine")
    assert result.drug_name == "Buprenorphine"


def test_drug_loading_preserved():
    result = _base(drug_loading_mg=10.0)
    assert result.drug_loading_mg == pytest.approx(10.0)
