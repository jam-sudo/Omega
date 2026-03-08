"""Tests for Phase 562 — Neonatal Drug Dosing Regimen."""

import pytest

from omega_pbpk.clinical.neonatal_dosing_regimen import (
    STANDARD_INTERVALS,
    NeonatalDosingResult,
    calculate_neonatal_dosing,
)


@pytest.fixture
def base_params():
    return dict(
        drug_name="DrugN",
        postnatal_age_days=7,
        weight_kg=3.0,
        gestational_age_weeks=38,
        target_css_mg_L=10.0,
    )


class TestReturnType:
    def test_returns_correct_type(self, base_params):
        result = calculate_neonatal_dosing(**base_params)
        assert isinstance(result, NeonatalDosingResult)

    def test_drug_name_preserved(self, base_params):
        result = calculate_neonatal_dosing(**base_params)
        assert result.drug_name == "DrugN"


class TestHalfLife:
    def test_t_half_positive(self, base_params):
        result = calculate_neonatal_dosing(**base_params)
        assert result.t_half_h > 0

    def test_premature_longer_half_life(self):
        term = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=7,
            weight_kg=3.0,
            gestational_age_weeks=39,
            target_css_mg_L=10.0,
        )
        preterm = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=7,
            weight_kg=1.5,
            gestational_age_weeks=28,
            target_css_mg_L=10.0,
        )
        # Premature should have longer half-life (lower CL, higher Vd)
        assert preterm.t_half_h > term.t_half_h


class TestClearance:
    def test_premature_lower_cl(self):
        term = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=7,
            weight_kg=3.0,
            gestational_age_weeks=39,
            target_css_mg_L=10.0,
        )
        preterm = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=7,
            weight_kg=1.5,
            gestational_age_weeks=28,
            target_css_mg_L=10.0,
        )
        assert preterm.cl_mL_per_min_per_kg < term.cl_mL_per_min_per_kg

    def test_older_pna_higher_cl(self):
        young = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=1,
            weight_kg=3.0,
            gestational_age_weeks=38,
            target_css_mg_L=10.0,
        )
        older = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=28,
            weight_kg=3.0,
            gestational_age_weeks=38,
            target_css_mg_L=10.0,
        )
        assert older.cl_mL_per_min_per_kg > young.cl_mL_per_min_per_kg


class TestVd:
    def test_vd_factor_gte_1(self, base_params):
        result = calculate_neonatal_dosing(**base_params)
        assert result.vd_L_per_kg >= 0.5  # baseline adult Vd

    def test_preterm_higher_vd(self):
        term = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=7,
            weight_kg=3.0,
            gestational_age_weeks=39,
            target_css_mg_L=10.0,
        )
        preterm = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=7,
            weight_kg=1.5,
            gestational_age_weeks=28,
            target_css_mg_L=10.0,
        )
        assert preterm.vd_L_per_kg >= term.vd_L_per_kg


class TestDosing:
    def test_dose_positive(self, base_params):
        result = calculate_neonatal_dosing(**base_params)
        assert result.recommended_dose_mg_per_kg > 0

    def test_interval_in_standard_set(self, base_params):
        result = calculate_neonatal_dosing(**base_params)
        assert result.dosing_interval_h in [float(x) for x in STANDARD_INTERVALS]

    def test_css_positive(self, base_params):
        result = calculate_neonatal_dosing(**base_params)
        assert result.css_avg_mg_L > 0


class TestDoseClass:
    def test_maintenance_only_short_half_life(self, base_params):
        # Term neonate with normal clearance should have short half-life
        result = calculate_neonatal_dosing(**base_params)
        if result.t_half_h <= 24.0:
            assert result.dose_class == "maintenance_only"

    def test_loading_dose_long_half_life(self):
        # Very premature with very low CL → long half-life
        result = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=0,
            weight_kg=0.8,
            gestational_age_weeks=24,
            target_css_mg_L=10.0,
        )
        if result.t_half_h > 24.0:
            assert result.dose_class == "loading_dose_needed"


class TestMaturationStage:
    def test_extremely_premature(self):
        result = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=1,
            weight_kg=0.7,
            gestational_age_weeks=25,
            target_css_mg_L=10.0,
        )
        assert result.maturation_stage == "extremely_premature"

    def test_premature(self):
        result = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=1,
            weight_kg=2.0,
            gestational_age_weeks=32,
            target_css_mg_L=10.0,
        )
        assert result.maturation_stage == "premature"

    def test_term(self):
        result = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=1,
            weight_kg=3.0,
            gestational_age_weeks=39,
            target_css_mg_L=10.0,
        )
        assert result.maturation_stage == "term"

    def test_post_term(self):
        result = calculate_neonatal_dosing(
            drug_name="X",
            postnatal_age_days=1,
            weight_kg=4.0,
            gestational_age_weeks=42,
            target_css_mg_L=10.0,
        )
        assert result.maturation_stage == "post_term"


class TestValidation:
    def test_negative_pna(self, base_params):
        base_params["postnatal_age_days"] = -1
        with pytest.raises(ValueError):
            calculate_neonatal_dosing(**base_params)

    def test_zero_weight(self, base_params):
        base_params["weight_kg"] = 0
        with pytest.raises(ValueError):
            calculate_neonatal_dosing(**base_params)

    def test_negative_weight(self, base_params):
        base_params["weight_kg"] = -1.0
        with pytest.raises(ValueError):
            calculate_neonatal_dosing(**base_params)

    def test_zero_target_css(self, base_params):
        base_params["target_css_mg_L"] = 0
        with pytest.raises(ValueError):
            calculate_neonatal_dosing(**base_params)

    def test_low_gestational_age(self, base_params):
        base_params["gestational_age_weeks"] = 15
        with pytest.raises(ValueError):
            calculate_neonatal_dosing(**base_params)
