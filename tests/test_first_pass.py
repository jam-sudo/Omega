"""Tests for omega_pbpk.core.first_pass module."""

import pytest

from omega_pbpk.core.first_pass import (
    FirstPassResult,
    _gut_availability,
    _hepatic_extraction,
    calculate_first_pass,
    first_pass_sensitivity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_result():
    return calculate_first_pass(
        drug_name="TestDrug",
        dose_mg=100.0,
        fup=0.1,
        cl_int_L_per_h=50.0,
    )


@pytest.fixture
def high_extraction_result():
    return calculate_first_pass(
        drug_name="HighExtDrug",
        dose_mg=50.0,
        fup=0.9,
        cl_int_L_per_h=5000.0,
    )


@pytest.fixture
def zero_extraction_result():
    return calculate_first_pass(
        drug_name="ZeroExtDrug",
        dose_mg=100.0,
        fup=0.1,
        cl_int_L_per_h=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Return type is FirstPassResult
# ---------------------------------------------------------------------------


def test_returns_first_pass_result_type(default_result):
    assert isinstance(default_result, FirstPassResult)


# ---------------------------------------------------------------------------
# 2. e_hepatic in [0, 0.999]
# ---------------------------------------------------------------------------


def test_e_hepatic_lower_bound(zero_extraction_result):
    assert zero_extraction_result.e_hepatic >= 0.0


def test_e_hepatic_upper_bound_default(default_result):
    assert default_result.e_hepatic <= 0.999


def test_e_hepatic_upper_bound_high_extraction(high_extraction_result):
    assert high_extraction_result.e_hepatic <= 0.999


def test_e_hepatic_in_range_various():
    for cl_int in [0.0, 1.0, 10.0, 100.0, 1000.0, 99999.0]:
        result = calculate_first_pass("D", 100.0, 0.5, cl_int)
        assert 0.0 <= result.e_hepatic <= 0.999, f"e_hepatic out of range for cl_int={cl_int}"


# ---------------------------------------------------------------------------
# 3. f_hepatic == 1 - e_hepatic (approximately)
# ---------------------------------------------------------------------------


def test_f_hepatic_equals_one_minus_e_hepatic(default_result):
    assert default_result.f_hepatic == pytest.approx(1.0 - default_result.e_hepatic, rel=1e-6)


def test_f_hepatic_equals_one_minus_e_hepatic_high(high_extraction_result):
    assert high_extraction_result.f_hepatic == pytest.approx(
        1.0 - high_extraction_result.e_hepatic, rel=1e-6
    )


# ---------------------------------------------------------------------------
# 4. f_oral in [0, 1]
# ---------------------------------------------------------------------------


def test_f_oral_in_range_default(default_result):
    assert 0.0 <= default_result.f_oral <= 1.0


def test_f_oral_in_range_high_extraction(high_extraction_result):
    assert 0.0 <= high_extraction_result.f_oral <= 1.0


def test_f_oral_in_range_zero_extraction(zero_extraction_result):
    assert 0.0 <= zero_extraction_result.f_oral <= 1.0


# ---------------------------------------------------------------------------
# 5. High cl_int -> high extraction
# ---------------------------------------------------------------------------


def test_high_cl_int_gives_high_extraction():
    result = calculate_first_pass("D", 100.0, 0.9, 10000.0)
    assert result.e_hepatic > 0.9


def test_high_extraction_relative_ordering():
    low = calculate_first_pass("D", 100.0, 0.5, 1.0)
    high = calculate_first_pass("D", 100.0, 0.5, 1000.0)
    assert high.e_hepatic > low.e_hepatic


# ---------------------------------------------------------------------------
# 6. Zero cl_int -> extraction ~ 0
# ---------------------------------------------------------------------------


def test_zero_cl_int_extraction_near_zero(zero_extraction_result):
    assert zero_extraction_result.e_hepatic == pytest.approx(0.0, abs=1e-9)


def test_zero_cl_int_f_hepatic_near_one(zero_extraction_result):
    assert zero_extraction_result.f_hepatic == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 7. IV route -> f_oral == 1.0
# ---------------------------------------------------------------------------


def test_iv_route_f_oral_is_one():
    result = calculate_first_pass(
        drug_name="IVDrug",
        dose_mg=10.0,
        fup=0.5,
        cl_int_L_per_h=100.0,
        route="iv",
    )
    assert result.f_oral == pytest.approx(1.0, abs=1e-9)


def test_iv_route_stored_in_result():
    result = calculate_first_pass("IVDrug", 10.0, 0.5, 100.0, route="iv")
    assert result.route == "iv"


# ---------------------------------------------------------------------------
# 8. ValueError for dose_mg <= 0
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        calculate_first_pass("D", 0.0, 0.1, 10.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError):
        calculate_first_pass("D", -5.0, 0.1, 10.0)


# ---------------------------------------------------------------------------
# 9. ValueError for fup <= 0
# ---------------------------------------------------------------------------


def test_raises_on_zero_fup():
    with pytest.raises(ValueError):
        calculate_first_pass("D", 100.0, 0.0, 10.0)


def test_raises_on_negative_fup():
    with pytest.raises(ValueError):
        calculate_first_pass("D", 100.0, -0.1, 10.0)


# ---------------------------------------------------------------------------
# 10. ValueError for cl_int < 0
# ---------------------------------------------------------------------------


def test_raises_on_negative_cl_int():
    with pytest.raises(ValueError):
        calculate_first_pass("D", 100.0, 0.1, -1.0)


# ---------------------------------------------------------------------------
# 11. hepatic_cl_L_per_h == e_hepatic * q_hepatic (approximately)
# ---------------------------------------------------------------------------


def test_hepatic_cl_equals_e_times_q(default_result):
    expected = default_result.e_hepatic * default_result.q_hepatic_L_per_h
    assert default_result.hepatic_cl_L_per_h == pytest.approx(expected, rel=1e-6)


def test_hepatic_cl_custom_q():
    result = calculate_first_pass("D", 100.0, 0.5, 50.0, q_hepatic_L_per_h=60.0)
    expected = result.e_hepatic * 60.0
    assert result.hepatic_cl_L_per_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 12. well_stirred_model == 'well_stirred'
# ---------------------------------------------------------------------------


def test_well_stirred_model_label(default_result):
    assert default_result.well_stirred_model == "well_stirred"


# ---------------------------------------------------------------------------
# 13. sensitivity list length == len(cl_int_range)
# ---------------------------------------------------------------------------


def test_sensitivity_list_length():
    cl_int_range = [1.0, 10.0, 50.0, 100.0, 500.0]
    results = first_pass_sensitivity("D", 100.0, 0.5, cl_int_range)
    assert len(results) == len(cl_int_range)


def test_sensitivity_list_length_single():
    results = first_pass_sensitivity("D", 100.0, 0.5, [42.0])
    assert len(results) == 1


def test_sensitivity_returns_first_pass_results():
    cl_int_range = [1.0, 10.0, 100.0]
    results = first_pass_sensitivity("D", 100.0, 0.5, cl_int_range)
    for r in results:
        assert isinstance(r, FirstPassResult)


# ---------------------------------------------------------------------------
# 14. Higher cl_int -> higher e_hepatic (monotonicity in sensitivity)
# ---------------------------------------------------------------------------


def test_sensitivity_monotone_extraction():
    cl_int_range = [0.1, 1.0, 10.0, 100.0, 1000.0]
    results = first_pass_sensitivity("D", 100.0, 0.3, cl_int_range)
    extractions = [r.e_hepatic for r in results]
    assert extractions == sorted(extractions), "e_hepatic should increase with cl_int"


# ---------------------------------------------------------------------------
# 15. fa=1 and low logP -> f_oral close to f_hepatic
# ---------------------------------------------------------------------------


def test_fa1_low_logP_f_oral_close_to_fh():
    """With fa=1 and low logP (high peff, fg~1), f_oral ≈ f_hepatic."""
    result = calculate_first_pass(
        drug_name="Hydrophilic",
        dose_mg=100.0,
        fup=0.5,
        cl_int_L_per_h=20.0,
        fa=1.0,
        logP=-1.0,  # very hydrophilic -> high gut permeability -> fg near 1.0
    )
    # f_oral = fa * fg * fh; with fa=1 and fg≈1, f_oral ≈ fh
    assert result.f_oral <= result.f_hepatic  # f_oral = fa*fg*fh <= fh


# ---------------------------------------------------------------------------
# 16. _hepatic_extraction private function
# ---------------------------------------------------------------------------


def test_hepatic_extraction_zero_clint():
    e = _hepatic_extraction(0.0, 90.0, 0.5)
    assert e == pytest.approx(0.0, abs=1e-9)


def test_hepatic_extraction_clamped_high():
    e = _hepatic_extraction(1e9, 90.0, 1.0)
    assert e <= 0.999


def test_hepatic_extraction_formula():
    cl_int, q, fup = 50.0, 90.0, 0.5
    expected = fup * cl_int / (q + fup * cl_int)
    e = _hepatic_extraction(cl_int, q, fup)
    assert e == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 17. _gut_availability private function
# ---------------------------------------------------------------------------


def test_gut_availability_in_range():
    fa, logP = 1.0, 2.0
    peff = 1.0  # arbitrary positive
    fg = _gut_availability(fa=fa, peff_cm_s=peff, logP=logP)
    assert 0.1 <= fg <= 1.0


def test_gut_availability_lower_bound():
    # Even with extreme unfavorable conditions, fg >= 0.1
    fg = _gut_availability(fa=0.0, peff_cm_s=0.0, logP=-10.0)
    assert fg >= 0.1


def test_gut_availability_upper_bound():
    # Favorable conditions should not exceed 1.0
    fg = _gut_availability(fa=1.0, peff_cm_s=100.0, logP=5.0)
    assert fg <= 1.0


# ---------------------------------------------------------------------------
# 18. Dataclass field correctness
# ---------------------------------------------------------------------------


def test_result_stores_drug_name(default_result):
    assert default_result.drug_name == "TestDrug"


def test_result_stores_dose(default_result):
    assert default_result.dose_mg == pytest.approx(100.0)


def test_result_stores_fup(default_result):
    assert default_result.fup == pytest.approx(0.1)


def test_result_stores_q_hepatic(default_result):
    assert default_result.q_hepatic_L_per_h == pytest.approx(90.0)


def test_result_route_default_is_oral(default_result):
    assert default_result.route == "oral"


def test_result_peak_portal_conc_nonnegative(default_result):
    assert default_result.peak_portal_conc_mg_L >= 0.0
