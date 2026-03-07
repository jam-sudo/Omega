"""Tests for clinical/pop_pk_covariate.py — Phase 552."""

import math

import pytest

from omega_pbpk.clinical.pop_pk_covariate import (
    categorical_covariate,
    compute_individual_pk,
    covariate_sensitivity_table,
    exponential_covariate,
    power_covariate,
)

# ---------------------------------------------------------------------------
# power_covariate
# ---------------------------------------------------------------------------


class TestPowerCovariate:
    def test_double_weight_cl(self):
        """Double weight → CL *= 2^0.75 ≈ 1.682."""
        cl = power_covariate(10.0, covariate_value=140.0, covariate_reference=70.0, exponent=0.75)
        expected = 10.0 * (2.0**0.75)
        assert abs(cl - expected) < 1e-10

    def test_at_reference_no_change(self):
        """At covariate == reference, result equals typical."""
        result = power_covariate(5.0, covariate_value=70.0, covariate_reference=70.0)
        assert abs(result - 5.0) < 1e-12

    def test_vd_weight_exp_1(self):
        """Vd with exponent=1.0: proportional to weight."""
        vd = power_covariate(50.0, covariate_value=105.0, covariate_reference=70.0, exponent=1.0)
        assert abs(vd - 50.0 * 1.5) < 1e-10

    def test_zero_reference_raises(self):
        with pytest.raises(ValueError, match="covariate_reference"):
            power_covariate(10.0, covariate_value=70.0, covariate_reference=0.0)

    def test_zero_covariate_raises(self):
        with pytest.raises(ValueError, match="covariate_value"):
            power_covariate(10.0, covariate_value=0.0, covariate_reference=70.0)


# ---------------------------------------------------------------------------
# exponential_covariate
# ---------------------------------------------------------------------------


class TestExponentialCovariate:
    def test_at_reference_no_change(self):
        """At covariate == reference, factor = exp(0) = 1 → no change."""
        result = exponential_covariate(
            8.0, covariate_value=35.0, covariate_reference=35.0, theta=-0.005
        )
        assert abs(result - 8.0) < 1e-12

    def test_negative_theta_decreases_with_age(self):
        """Negative theta: older patient → lower CL."""
        cl_young = exponential_covariate(8.0, 35.0, 35.0, theta=-0.005)
        cl_old = exponential_covariate(8.0, 70.0, 35.0, theta=-0.005)
        assert cl_old < cl_young

    def test_positive_theta_increases_value(self):
        result = exponential_covariate(
            5.0, covariate_value=50.0, covariate_reference=35.0, theta=0.01
        )
        expected = 5.0 * math.exp(0.01 * 15)
        assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# categorical_covariate
# ---------------------------------------------------------------------------


class TestCategoricalCovariate:
    def test_female_lower_cl(self):
        """Female category factor < 1 → lower CL."""
        effects = {"male": 1.0, "female": 0.85}
        cl_m = categorical_covariate(10.0, "male", effects)
        cl_f = categorical_covariate(10.0, "female", effects)
        assert cl_f < cl_m
        assert abs(cl_f - 8.5) < 1e-12

    def test_unknown_category_defaults_to_one(self):
        """Unknown category → factor 1.0 → no change."""
        result = categorical_covariate(10.0, "other", {"male": 1.0, "female": 0.85})
        assert abs(result - 10.0) < 1e-12

    def test_known_category_factor_applied(self):
        effects = {"slow": 0.5, "fast": 1.5}
        result = categorical_covariate(4.0, "fast", effects)
        assert abs(result - 6.0) < 1e-12


# ---------------------------------------------------------------------------
# compute_individual_pk
# ---------------------------------------------------------------------------


class TestComputeIndividualPK:
    def test_heavier_patient_higher_cl(self):
        """Heavier patient → higher CL (positive allometry)."""
        pk_light = compute_individual_pk(
            10.0, 50.0, weight_kg=50.0, age_years=35, sex="male", egfr=100.0
        )
        pk_heavy = compute_individual_pk(
            10.0, 50.0, weight_kg=100.0, age_years=35, sex="male", egfr=100.0
        )
        assert pk_heavy["cl_individual"] > pk_light["cl_individual"]

    def test_heavier_patient_higher_vd(self):
        """Heavier patient → higher Vd."""
        pk_light = compute_individual_pk(
            10.0, 50.0, weight_kg=50.0, age_years=35, sex="male", egfr=100.0
        )
        pk_heavy = compute_individual_pk(
            10.0, 50.0, weight_kg=100.0, age_years=35, sex="male", egfr=100.0
        )
        assert pk_heavy["vd_individual"] > pk_light["vd_individual"]

    def test_low_egfr_lower_cl(self):
        """Impaired renal function → lower CL."""
        pk_normal = compute_individual_pk(
            10.0, 50.0, weight_kg=70.0, age_years=35, sex="male", egfr=100.0
        )
        pk_ckd = compute_individual_pk(
            10.0, 50.0, weight_kg=70.0, age_years=35, sex="male", egfr=20.0
        )
        assert pk_ckd["cl_individual"] < pk_normal["cl_individual"]

    def test_female_lower_cl(self):
        """Female sex → lower CL by default factor 0.9."""
        pk_m = compute_individual_pk(
            10.0, 50.0, weight_kg=70.0, age_years=35, sex="male", egfr=100.0
        )
        pk_f = compute_individual_pk(
            10.0, 50.0, weight_kg=70.0, age_years=35, sex="female", egfr=100.0
        )
        assert pk_f["cl_individual"] < pk_m["cl_individual"]
        assert abs(pk_f["cl_individual"] / pk_m["cl_individual"] - 0.9) < 1e-6

    def test_reference_individual(self):
        """70 kg, 35 y, male, eGFR=100 → cl_individual close to typical."""
        pk = compute_individual_pk(10.0, 50.0, weight_kg=70.0, age_years=35, sex="male", egfr=100.0)
        assert abs(pk["cl_individual"] - 10.0) < 1e-6

    def test_returns_required_keys(self):
        pk = compute_individual_pk(10.0, 50.0, weight_kg=70.0, age_years=35, sex="male", egfr=100.0)
        for key in ("cl_individual", "vd_individual", "ke", "t_half_h", "auc_dose", "eta_cl"):
            assert key in pk

    def test_ke_and_t_half_consistent(self):
        pk = compute_individual_pk(10.0, 50.0, weight_kg=70.0, age_years=35, sex="male", egfr=100.0)
        ke_expected = pk["cl_individual"] / pk["vd_individual"]
        t_half_expected = math.log(2) / ke_expected
        assert abs(pk["ke"] - ke_expected) < 1e-10
        assert abs(pk["t_half_h"] - t_half_expected) < 1e-10

    def test_eta_cl_zero(self):
        """Deterministic model has no IIV."""
        pk = compute_individual_pk(10.0, 50.0, weight_kg=70.0, age_years=35, sex="male", egfr=100.0)
        assert pk["eta_cl"] == 0.0

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            compute_individual_pk(10.0, 50.0, weight_kg=-5.0, age_years=35, sex="male", egfr=100.0)

    def test_negative_egfr_raises(self):
        with pytest.raises(ValueError, match="egfr"):
            compute_individual_pk(10.0, 50.0, weight_kg=70.0, age_years=35, sex="male", egfr=-10.0)


# ---------------------------------------------------------------------------
# covariate_sensitivity_table
# ---------------------------------------------------------------------------


class TestCovariateSensitivityTable:
    def test_returns_list_of_dicts(self):
        table = covariate_sensitivity_table(
            cl_typical=10.0,
            vd_typical=50.0,
            covariates_to_vary=[
                {"covariate": "weight", "values": [50.0, 70.0, 100.0], "reference": 70.0}
            ],
        )
        assert isinstance(table, list)
        assert len(table) == 3
        for row in table:
            assert "covariate" in row
            assert "value" in row
            assert "cl_fold_change" in row
            assert "vd_fold_change" in row
            assert "t_half_fold_change" in row

    def test_reference_weight_fold_change_one(self):
        """At the reference weight, CL fold-change should be ~1."""
        table = covariate_sensitivity_table(
            cl_typical=10.0,
            vd_typical=50.0,
            covariates_to_vary=[{"covariate": "weight", "values": [70.0], "reference": 70.0}],
        )
        assert abs(table[0]["cl_fold_change"] - 1.0) < 1e-6

    def test_low_egfr_fold_change_less_than_one(self):
        """Low eGFR → CL fold-change < 1."""
        table = covariate_sensitivity_table(
            cl_typical=10.0,
            vd_typical=50.0,
            covariates_to_vary=[{"covariate": "egfr", "values": [30.0], "reference": 100.0}],
        )
        assert table[0]["cl_fold_change"] < 1.0

    def test_multiple_covariates(self):
        """Multiple covariate specs return concatenated rows."""
        table = covariate_sensitivity_table(
            cl_typical=10.0,
            vd_typical=50.0,
            covariates_to_vary=[
                {"covariate": "weight", "values": [50.0, 100.0], "reference": 70.0},
                {"covariate": "egfr", "values": [50.0, 100.0], "reference": 100.0},
            ],
        )
        assert len(table) == 4

    def test_sex_covariate(self):
        """Female sex → CL fold-change of 0.9."""
        table = covariate_sensitivity_table(
            cl_typical=10.0,
            vd_typical=50.0,
            covariates_to_vary=[{"covariate": "sex", "values": ["female"], "reference": "male"}],
        )
        assert abs(table[0]["cl_fold_change"] - 0.9) < 1e-6
