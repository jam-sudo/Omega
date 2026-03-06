"""Tests for omega_pbpk.clinical.organ_impairment module."""

import pytest

from omega_pbpk.clinical.organ_impairment import (
    OrganImpairmentResult,
    _cardiac_cl_factor,
    _hepatic_cl_factor,
    _renal_cl_factor,
    adjust_for_organ_impairment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG = "vancomycin"
DOSE = 1000.0  # mg
INTERVAL = 12.0  # h
CL = 5.0  # L/h
VD = 50.0  # L


def default_result(**kwargs):
    params = dict(
        drug_name=DRUG,
        dose_mg=DOSE,
        dosing_interval_h=INTERVAL,
        cl_L_per_h=CL,
        vd_L=VD,
    )
    params.update(kwargs)
    return adjust_for_organ_impairment(**params)


# ---------------------------------------------------------------------------
# Return type and dataclass structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_organ_impairment_result(self):
        result = default_result()
        assert isinstance(result, OrganImpairmentResult)

    def test_dataclass_has_drug_name(self):
        result = default_result()
        assert result.drug_name == DRUG

    def test_dataclass_has_renal_function_str(self):
        result = default_result()
        assert isinstance(result.renal_function, str)

    def test_dataclass_has_hepatic_function_str(self):
        result = default_result()
        assert isinstance(result.hepatic_function, str)

    def test_dataclass_has_cardiac_function_str(self):
        result = default_result()
        assert isinstance(result.cardiac_function, str)

    def test_dataclass_has_adjusted_dose_mg(self):
        result = default_result()
        assert hasattr(result, "adjusted_dose_mg")

    def test_dataclass_has_adjusted_interval_h(self):
        result = default_result()
        assert hasattr(result, "adjusted_interval_h")

    def test_dataclass_has_dose_recommendation(self):
        result = default_result()
        assert isinstance(result.dose_recommendation, str)

    def test_pk_parameters_adjusted_is_dict(self):
        result = default_result()
        assert isinstance(result.pk_parameters_adjusted, dict)


# ---------------------------------------------------------------------------
# pk_parameters_adjusted keys
# ---------------------------------------------------------------------------


class TestPKParameters:
    def test_pk_parameters_has_CL_L_per_h(self):
        result = default_result()
        assert "CL_L_per_h" in result.pk_parameters_adjusted

    def test_pk_parameters_has_Vd_L(self):
        result = default_result()
        assert "Vd_L" in result.pk_parameters_adjusted

    def test_pk_parameters_has_t_half_h(self):
        result = default_result()
        assert "t_half_h" in result.pk_parameters_adjusted


# ---------------------------------------------------------------------------
# cl_adjustment_factor range and adjusted_dose
# ---------------------------------------------------------------------------


class TestCLFactor:
    def test_cl_factor_in_range_normal(self):
        result = default_result()
        assert 0 < result.cl_adjustment_factor <= 1.0

    def test_cl_factor_clamped_above(self):
        # Even with perfect organs, factor should not exceed 1.0
        result = default_result(crcl_mL_per_min=120.0, child_pugh_score=5, nyha_class=1)
        assert result.cl_adjustment_factor <= 1.0

    def test_cl_factor_clamped_below(self):
        # Worst-case impairment should not drop below 0.1
        result = default_result(
            crcl_mL_per_min=5.0,
            child_pugh_score=15,
            nyha_class=4,
            fraction_renal=0.9,
            fraction_hepatic=0.9,
        )
        assert result.cl_adjustment_factor >= 0.1

    def test_adjusted_dose_equals_dose_times_cl_factor(self):
        result = default_result()
        assert result.adjusted_dose_mg == pytest.approx(
            DOSE * result.cl_adjustment_factor, rel=1e-6
        )


# ---------------------------------------------------------------------------
# t_half_factor
# ---------------------------------------------------------------------------


class TestTHalfFactor:
    def test_t_half_factor_positive(self):
        result = default_result()
        assert result.t_half_adjustment_factor > 0

    def test_t_half_factor_equals_vd_over_cl(self):
        result = default_result()
        expected = result.vd_adjustment_factor / result.cl_adjustment_factor
        assert result.t_half_adjustment_factor == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Hepatic impairment
# ---------------------------------------------------------------------------


class TestHepaticImpairment:
    def test_child_pugh_c_lower_cl_than_normal(self):
        normal = default_result(child_pugh_score=5)
        severe = default_result(child_pugh_score=13)
        assert severe.cl_adjustment_factor < normal.cl_adjustment_factor

    def test_hepatic_factor_child_pugh_le6(self):
        assert _hepatic_cl_factor(5) == pytest.approx(1.0)
        assert _hepatic_cl_factor(6) == pytest.approx(1.0)

    def test_hepatic_factor_child_pugh_7_to_9(self):
        assert _hepatic_cl_factor(7) == pytest.approx(0.75)
        assert _hepatic_cl_factor(9) == pytest.approx(0.75)

    def test_hepatic_factor_child_pugh_10_to_12(self):
        assert _hepatic_cl_factor(10) == pytest.approx(0.5)
        assert _hepatic_cl_factor(12) == pytest.approx(0.5)

    def test_hepatic_factor_child_pugh_gt12(self):
        assert _hepatic_cl_factor(13) == pytest.approx(0.25)
        assert _hepatic_cl_factor(15) == pytest.approx(0.25)

    def test_hepatic_function_string_valid(self):
        result = default_result(child_pugh_score=5)
        assert len(result.hepatic_function) > 0


# ---------------------------------------------------------------------------
# Renal impairment
# ---------------------------------------------------------------------------


class TestRenalImpairment:
    def test_esrd_lower_cl_than_normal(self):
        normal = default_result(crcl_mL_per_min=100.0, fraction_renal=0.8)
        esrd = default_result(crcl_mL_per_min=5.0, fraction_renal=0.8)
        assert esrd.cl_adjustment_factor < normal.cl_adjustment_factor

    def test_renal_factor_normal_crcl(self):
        factor = _renal_cl_factor(100.0, fraction_renal=0.5)
        assert factor == pytest.approx(1.0, rel=1e-2)

    def test_renal_factor_zero_fraction(self):
        # Drug not renally cleared → renal impairment has no effect
        factor = _renal_cl_factor(5.0, fraction_renal=0.0)
        assert factor == pytest.approx(1.0)

    def test_renal_function_string_valid(self):
        result = default_result(crcl_mL_per_min=100.0)
        assert len(result.renal_function) > 0


# ---------------------------------------------------------------------------
# Cardiac impairment
# ---------------------------------------------------------------------------


class TestCardiacImpairment:
    def test_nyha4_lower_cl_than_normal(self):
        normal = default_result(nyha_class=1)
        severe = default_result(nyha_class=4)
        assert severe.cl_adjustment_factor < normal.cl_adjustment_factor

    def test_cardiac_factor_nyha1(self):
        assert _cardiac_cl_factor(1) == pytest.approx(1.0)

    def test_cardiac_factor_nyha2(self):
        assert _cardiac_cl_factor(2) == pytest.approx(0.85)

    def test_cardiac_factor_nyha3(self):
        assert _cardiac_cl_factor(3) == pytest.approx(0.70)

    def test_cardiac_factor_nyha4(self):
        assert _cardiac_cl_factor(4) == pytest.approx(0.50)

    def test_cardiac_function_string_valid(self):
        result = default_result(nyha_class=1)
        assert len(result.cardiac_function) > 0


# ---------------------------------------------------------------------------
# Combined impairment
# ---------------------------------------------------------------------------


class TestCombinedImpairment:
    def test_combined_lower_than_renal_alone(self):
        renal_only = default_result(
            crcl_mL_per_min=10.0,
            child_pugh_score=5,
            nyha_class=1,
            fraction_renal=0.5,
            fraction_hepatic=0.0,
        )
        combined = default_result(
            crcl_mL_per_min=10.0,
            child_pugh_score=13,
            nyha_class=4,
            fraction_renal=0.5,
            fraction_hepatic=0.5,
        )
        assert combined.cl_adjustment_factor < renal_only.cl_adjustment_factor

    def test_combined_lower_than_hepatic_alone(self):
        hepatic_only = default_result(
            crcl_mL_per_min=100.0,
            child_pugh_score=13,
            nyha_class=1,
            fraction_renal=0.0,
            fraction_hepatic=0.9,
        )
        combined = default_result(
            crcl_mL_per_min=10.0,
            child_pugh_score=13,
            nyha_class=4,
            fraction_renal=0.5,
            fraction_hepatic=0.9,
        )
        assert combined.cl_adjustment_factor < hepatic_only.cl_adjustment_factor


# ---------------------------------------------------------------------------
# Input validation — ValueError
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            adjust_for_organ_impairment(DRUG, 0.0, INTERVAL, CL, VD)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            adjust_for_organ_impairment(DRUG, -100.0, INTERVAL, CL, VD)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError):
            adjust_for_organ_impairment(DRUG, DOSE, INTERVAL, 0.0, VD)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError):
            adjust_for_organ_impairment(DRUG, DOSE, INTERVAL, -1.0, VD)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError):
            adjust_for_organ_impairment(DRUG, DOSE, INTERVAL, CL, 0.0)

    def test_vd_negative_raises(self):
        with pytest.raises(ValueError):
            adjust_for_organ_impairment(DRUG, DOSE, INTERVAL, CL, -10.0)
