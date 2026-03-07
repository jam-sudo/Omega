"""Tests for first-pass metabolism predictor."""

import pytest

from omega_pbpk.prediction.first_pass_predictor import (
    FirstPassResult,
    predict_first_pass,
    scan_clint_range,
)


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        clint_gut_mL_per_min=5.0,
        clint_hepatic_mL_per_min=20.0,
        fu_plasma=0.5,
        fa=1.0,
    )
    defaults.update(kwargs)
    return predict_first_pass(**defaults)


def test_returns_result():
    result = make_result()
    assert isinstance(result, FirstPassResult)


def test_f_oral_in_0_1():
    result = make_result()
    assert 0.0 <= result.f_oral <= 1.0


def test_high_clint_low_f():
    result = predict_first_pass(
        drug_name="HighCL",
        clint_gut_mL_per_min=500.0,
        clint_hepatic_mL_per_min=2000.0,
        fu_plasma=0.9,
        fu_gut=0.9,
    )
    assert result.f_oral < 0.2


def test_zero_clint_f_is_fa():
    fa = 0.8
    result = predict_first_pass(
        drug_name="ZeroCL",
        clint_gut_mL_per_min=0.0,
        clint_hepatic_mL_per_min=0.0,
        fu_plasma=0.5,
        fa=fa,
    )
    assert abs(result.f_oral - fa) < 1e-9


def test_fg_in_0_1():
    result = make_result()
    assert 0.0 <= result.fg_gut_wall <= 1.0


def test_fh_in_0_1():
    result = make_result()
    assert 0.0 <= result.fh_hepatic <= 1.0


def test_er_hepatic_in_0_1():
    result = make_result()
    assert 0.0 <= result.er_hepatic <= 1.0


def test_er_gut_in_0_1():
    result = make_result()
    assert 0.0 <= result.er_gut <= 1.0


def test_fa_effect():
    r_high = predict_first_pass(
        drug_name="Drug",
        clint_gut_mL_per_min=5.0,
        clint_hepatic_mL_per_min=20.0,
        fa=1.0,
    )
    r_low = predict_first_pass(
        drug_name="Drug",
        clint_gut_mL_per_min=5.0,
        clint_hepatic_mL_per_min=20.0,
        fa=0.5,
    )
    assert r_low.f_oral < r_high.f_oral


def test_primary_route_liver_dominant():
    result = predict_first_pass(
        drug_name="LiverDrug",
        clint_gut_mL_per_min=1.0,
        clint_hepatic_mL_per_min=500.0,
        fu_plasma=0.8,
        fu_gut=0.1,
    )
    assert result.primary_route == "liver_dominant"


def test_primary_route_gut_dominant():
    result = predict_first_pass(
        drug_name="GutDrug",
        clint_gut_mL_per_min=500.0,
        clint_hepatic_mL_per_min=1.0,
        fu_plasma=0.1,
        fu_gut=0.9,
    )
    assert result.primary_route == "gut_dominant"


def test_primary_route_minimal():
    result = predict_first_pass(
        drug_name="MinDrug",
        clint_gut_mL_per_min=0.001,
        clint_hepatic_mL_per_min=0.001,
        fu_plasma=0.5,
    )
    assert result.primary_route == "minimal"


def test_f_oral_pct_consistent():
    result = make_result()
    assert abs(result.f_oral_pct - result.f_oral * 100.0) < 0.001


def test_scan_returns_list():
    results = scan_clint_range("Drug", [10.0, 20.0, 50.0])
    assert isinstance(results, list)


def test_scan_length_matches():
    values = [5.0, 10.0, 20.0, 50.0, 100.0]
    results = scan_clint_range("Drug", values)
    assert len(results) == len(values)


def test_scan_higher_clint_lower_f():
    values = [1.0, 10.0, 100.0, 1000.0]
    results = scan_clint_range("Drug", values)
    f_orals = [r.f_oral for r in results]
    for i in range(len(f_orals) - 1):
        assert f_orals[i] >= f_orals[i + 1]


def test_mm_saturation_dose_linearity():
    result = predict_first_pass(
        drug_name="MMDrug",
        clint_gut_mL_per_min=0.0,
        clint_hepatic_mL_per_min=1000.0,
        fu_plasma=0.9,
        dose_mg=500.0,
        km_mg_L=1.0,  # very low Km relative to concentration
        vmax_mg_per_h=5.0,  # low Vmax → saturated at high dose
    )
    assert result.dose_linearity == "nonlinear"


def test_linear_without_mm():
    result = predict_first_pass(
        drug_name="LinearDrug",
        clint_gut_mL_per_min=5.0,
        clint_hepatic_mL_per_min=20.0,
    )
    assert result.dose_linearity == "linear"


def test_invalid_clint_negative_raises():
    with pytest.raises(ValueError):
        predict_first_pass(
            drug_name="Bad",
            clint_gut_mL_per_min=-1.0,
            clint_hepatic_mL_per_min=10.0,
        )


def test_invalid_fu_raises():
    with pytest.raises(ValueError):
        predict_first_pass(
            drug_name="Bad",
            clint_gut_mL_per_min=5.0,
            clint_hepatic_mL_per_min=10.0,
            fu_plasma=0.0,
        )


def test_invalid_fa_raises():
    with pytest.raises(ValueError):
        predict_first_pass(
            drug_name="Bad",
            clint_gut_mL_per_min=5.0,
            clint_hepatic_mL_per_min=10.0,
            fa=1.5,
        )


def test_notes_nonempty():
    result = make_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
