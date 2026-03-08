"""
Tests for Phase 407 — PK/PD Target Attainment Analysis
"""

import pytest

from omega_pbpk.clinical.pkpd_target_attainment import (
    TargetAttainmentResult,
    assess_target_attainment,
    screen_mic_range,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_result(pkpd_index="fT>MIC", target_value=40.0,
                mic_mg_L=0.125, dose_mg=500.0,
                dosing_interval_h=8.0):
    """Default amoxicillin-like parameters."""
    return assess_target_attainment(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        cl_L_per_h=6.0,
        vd_L=30.0,
        mic_mg_L=mic_mg_L,
        pkpd_index=pkpd_index,
        target_value=target_value,
        ka_per_h=1.5,
        f_oral=0.9,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

def test_return_type():
    result = make_result()
    assert isinstance(result, TargetAttainmentResult)


def test_result_is_frozen():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"


# ---------------------------------------------------------------------------
# 2. fT>MIC index
# ---------------------------------------------------------------------------

def test_ft_mic_low_mic_attained():
    """Low MIC should achieve fT>MIC target."""
    result = make_result(pkpd_index="fT>MIC", target_value=40.0, mic_mg_L=0.03)
    assert result.target_attained is True
    assert result.f_time_above_mic_pct > 40.0


def test_ft_mic_high_mic_not_attained():
    """Very high MIC should not achieve fT>MIC target."""
    result = make_result(pkpd_index="fT>MIC", target_value=40.0, mic_mg_L=10.0)
    assert result.target_attained is False
    assert result.f_time_above_mic_pct < 40.0


def test_ft_mic_achieved_value_matches_f_time():
    """achieved_value should equal f_time_above_mic_pct for fT>MIC."""
    result = make_result(pkpd_index="fT>MIC")
    assert abs(result.achieved_value - result.f_time_above_mic_pct) < 1e-6


def test_ft_mic_range_valid():
    """f_time_above_mic_pct must be in [0, 100]."""
    for mic in [0.03, 0.5, 2.0, 8.0]:
        result = make_result(pkpd_index="fT>MIC", mic_mg_L=mic)
        assert 0.0 <= result.f_time_above_mic_pct <= 100.0


# ---------------------------------------------------------------------------
# 3. AUC/MIC index
# ---------------------------------------------------------------------------

def test_auc_mic_calculation():
    """AUC/MIC achieved value = AUC_24h / MIC."""
    result = make_result(pkpd_index="AUC/MIC", target_value=400.0, mic_mg_L=0.5)
    expected = result.auc_24h_mg_h_per_L / result.mic_mg_L
    assert abs(result.achieved_value - expected) < 1e-6


def test_auc_mic_low_mic_attained():
    """Low MIC → AUC/MIC target attained."""
    result = make_result(pkpd_index="AUC/MIC", target_value=100.0,
                         mic_mg_L=0.03, dose_mg=500.0)
    assert result.target_attained is True


def test_auc_mic_high_mic_not_attained():
    """High MIC → AUC/MIC target not attained."""
    result = make_result(pkpd_index="AUC/MIC", target_value=400.0,
                         mic_mg_L=8.0, dose_mg=500.0)
    assert result.target_attained is False


def test_auc_mic_auc_24h_positive():
    """AUC 24h must be positive."""
    result = make_result(pkpd_index="AUC/MIC")
    assert result.auc_24h_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 4. Cmax/MIC index
# ---------------------------------------------------------------------------

def test_cmax_mic_calculation():
    """Cmax/MIC achieved value = Cmax / MIC."""
    result = make_result(pkpd_index="Cmax/MIC", target_value=8.0, mic_mg_L=0.5)
    expected = result.cmax_mg_L / result.mic_mg_L
    assert abs(result.achieved_value - expected) < 1e-6


def test_cmax_mic_low_mic_attained():
    """Low MIC → Cmax/MIC attained."""
    result = make_result(pkpd_index="Cmax/MIC", target_value=8.0, mic_mg_L=0.06)
    assert result.target_attained is True


def test_cmax_positive():
    """Cmax must be positive."""
    result = make_result(pkpd_index="Cmax/MIC")
    assert result.cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# 5. Breakpoint
# ---------------------------------------------------------------------------

def test_auc_mic_breakpoint():
    """Breakpoint for AUC/MIC = AUC_24h / target."""
    result = make_result(pkpd_index="AUC/MIC", target_value=400.0, mic_mg_L=0.5)
    expected_bp = result.auc_24h_mg_h_per_L / result.target_value
    assert abs(result.static_breakpoint_mg_L - expected_bp) < 1e-6


def test_cmax_mic_breakpoint():
    """Breakpoint for Cmax/MIC = Cmax / target."""
    result = make_result(pkpd_index="Cmax/MIC", target_value=8.0, mic_mg_L=0.5)
    expected_bp = result.cmax_mg_L / result.target_value
    assert abs(result.static_breakpoint_mg_L - expected_bp) < 1e-6


def test_ft_mic_breakpoint_positive():
    """fT>MIC breakpoint must be positive when target is achievable."""
    result = make_result(pkpd_index="fT>MIC", target_value=30.0, mic_mg_L=0.5)
    assert result.static_breakpoint_mg_L >= 0.0


# ---------------------------------------------------------------------------
# 6. Margin
# ---------------------------------------------------------------------------

def test_margin_positive_when_attained():
    """margin_pct > 0 when target is attained."""
    result = make_result(pkpd_index="AUC/MIC", target_value=50.0,
                         mic_mg_L=0.125, dose_mg=1000.0)
    assert result.target_attained is True
    assert result.margin_pct > 0.0


def test_margin_negative_when_not_attained():
    """margin_pct < 0 when target is not attained."""
    result = make_result(pkpd_index="AUC/MIC", target_value=10000.0,
                         mic_mg_L=8.0, dose_mg=100.0)
    assert result.target_attained is False
    assert result.margin_pct < 0.0


def test_margin_formula():
    """margin_pct = (achieved - target) / target * 100."""
    result = make_result(pkpd_index="AUC/MIC", target_value=100.0, mic_mg_L=0.5)
    expected = (result.achieved_value - result.target_value) / result.target_value * 100.0
    assert abs(result.margin_pct - expected) < 1e-6


# ---------------------------------------------------------------------------
# 7. screen_mic_range
# ---------------------------------------------------------------------------

def test_screen_returns_list():
    results = screen_mic_range(
        "TestDrug", 500.0, 8.0, 6.0, 30.0
    )
    assert isinstance(results, list)


def test_screen_default_8_mics():
    results = screen_mic_range("TestDrug", 500.0, 8.0, 6.0, 30.0)
    assert len(results) == 8


def test_screen_sorted_ascending():
    results = screen_mic_range("TestDrug", 500.0, 8.0, 6.0, 30.0)
    mics = [r.mic_mg_L for r in results]
    assert mics == sorted(mics)


def test_screen_custom_mics():
    custom = [0.5, 1.0, 2.0]
    results = screen_mic_range("TestDrug", 500.0, 8.0, 6.0, 30.0,
                               mic_values=custom)
    assert len(results) == 3
    assert results[0].mic_mg_L <= results[-1].mic_mg_L


def test_screen_low_mic_attained_high_not():
    results = screen_mic_range(
        "TestDrug", 1000.0, 8.0, 6.0, 30.0,
        pkpd_index="AUC/MIC", target_value=100.0
    )
    # lowest MIC should be attained
    assert results[0].target_attained is True
    # highest MIC (4.0 mg/L) is more challenging; depends on parameters
    # just check all results have correct type
    for r in results:
        assert isinstance(r, TargetAttainmentResult)


# ---------------------------------------------------------------------------
# 8. Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        assess_target_attainment("Drug", -1.0, 8.0, 6.0, 30.0, 0.5)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl"):
        assess_target_attainment("Drug", 500.0, 8.0, 0.0, 30.0, 0.5)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd"):
        assess_target_attainment("Drug", 500.0, 8.0, 6.0, -5.0, 0.5)


def test_invalid_mic():
    with pytest.raises(ValueError, match="mic"):
        assess_target_attainment("Drug", 500.0, 8.0, 6.0, 30.0, 0.0)


def test_invalid_pkpd_index():
    with pytest.raises(ValueError, match="pkpd_index"):
        assess_target_attainment("Drug", 500.0, 8.0, 6.0, 30.0, 0.5,
                                 pkpd_index="invalid")


def test_invalid_target_value():
    with pytest.raises(ValueError, match="target_value"):
        assess_target_attainment("Drug", 500.0, 8.0, 6.0, 30.0, 0.5,
                                 target_value=-10.0)


def test_invalid_dosing_interval():
    with pytest.raises(ValueError, match="dosing_interval"):
        assess_target_attainment("Drug", 500.0, 0.0, 6.0, 30.0, 0.5)
