"""Tests for neonatal PK scaling module (Phase 882)."""

import pytest

from omega_pbpk.clinical.neonatal_pk_scaling import (
    NeonatalPKResult,
    compare_age_groups,
    scale_neonatal_pk,
)


class TestScaleNeonatalPK:
    def test_returns_result(self):
        result = scale_neonatal_pk(
            "Drug", age_days=14, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert isinstance(result, NeonatalPKResult)

    def test_drug_name_preserved(self):
        result = scale_neonatal_pk(
            "Amoxicillin", age_days=7, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert result.drug_name == "Amoxicillin"

    def test_age_days_preserved(self):
        result = scale_neonatal_pk(
            "Drug", age_days=21, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert result.age_days == 21

    def test_weight_preserved(self):
        result = scale_neonatal_pk(
            "Drug", age_days=14, weight_kg=3.5, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert result.weight_kg == 3.5

    def test_adult_cl_preserved(self):
        result = scale_neonatal_pk(
            "Drug", age_days=14, weight_kg=3.0, adult_cl_L_per_h=8.0, adult_vd_L=50.0
        )
        assert result.adult_cl_L_per_h == 8.0

    def test_adult_vd_preserved(self):
        result = scale_neonatal_pk(
            "Drug", age_days=14, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=100.0
        )
        assert result.adult_vd_L == 100.0

    def test_gfr_fraction_range(self):
        for age in [0, 7, 14, 28, 90, 180, 365]:
            result = scale_neonatal_pk(
                "Drug", age_days=age, weight_kg=5.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
            )
            assert 0.02 <= result.gfr_fraction_adult <= 1.0

    def test_gfr_increases_with_age(self):
        """GFR fraction should generally increase with age."""
        r1 = scale_neonatal_pk(
            "Drug", age_days=1, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        r2 = scale_neonatal_pk(
            "Drug", age_days=180, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert r2.gfr_fraction_adult > r1.gfr_fraction_adult

    def test_cyp3a4_maturation_range(self):
        for age in [0, 7, 28, 90, 365]:
            result = scale_neonatal_pk(
                "Drug", age_days=age, weight_kg=5.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
            )
            assert 0.05 <= result.cyp3a4_maturation_fraction <= 1.0

    def test_cyp3a4_increases_with_age(self):
        r1 = scale_neonatal_pk(
            "Drug", age_days=1, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        r2 = scale_neonatal_pk(
            "Drug", age_days=360, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert r2.cyp3a4_maturation_fraction > r1.cyp3a4_maturation_fraction

    def test_neonatal_cl_positive(self):
        result = scale_neonatal_pk(
            "Drug", age_days=14, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert result.neonatal_cl_L_per_h > 0.0

    def test_neonatal_cl_less_than_adult(self):
        """Neonate CL should be much lower than adult (weight + maturation scaled)."""
        result = scale_neonatal_pk(
            "Drug",
            age_days=7,
            weight_kg=3.0,
            adult_cl_L_per_h=5.0,
            adult_vd_L=50.0,
            primary_elimination="renal",
        )
        assert result.neonatal_cl_L_per_h < result.adult_cl_L_per_h

    def test_hepatic_elimination_uses_cyp(self):
        """Hepatic elimination: CL scale uses CYP3A4 fraction."""
        result = scale_neonatal_pk(
            "Drug",
            age_days=7,
            weight_kg=3.0,
            adult_cl_L_per_h=5.0,
            adult_vd_L=50.0,
            primary_elimination="hepatic",
        )
        expected_scale = result.cyp3a4_maturation_fraction * (3.0 / 70.0) ** 0.75
        assert abs(result.cl_scaling_factor - expected_scale) < 1e-6

    def test_protein_binding_adjustment_neonate(self):
        result = scale_neonatal_pk(
            "Drug", age_days=0, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert result.protein_binding_adjustment < 1.0

    def test_protein_binding_adjustment_older_infant(self):
        result = scale_neonatal_pk(
            "Drug", age_days=90, weight_kg=5.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert result.protein_binding_adjustment == 1.0

    def test_predicted_t_half_positive(self):
        result = scale_neonatal_pk(
            "Drug", age_days=14, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert result.predicted_t_half_h > 0.0

    def test_t_half_formula(self):
        result = scale_neonatal_pk(
            "Drug", age_days=14, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        expected = 0.693 * result.neonatal_vd_L / result.neonatal_cl_L_per_h
        assert abs(result.predicted_t_half_h - expected) < 1e-6

    def test_notes_very_early_neonate(self):
        result = scale_neonatal_pk(
            "Drug", age_days=3, weight_kg=2.5, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert "Extreme caution" in result.notes

    def test_notes_neonate(self):
        result = scale_neonatal_pk(
            "Drug", age_days=14, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert "Neonatal" in result.notes

    def test_notes_infant(self):
        result = scale_neonatal_pk(
            "Drug", age_days=90, weight_kg=5.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
        )
        assert "Infant" in result.notes

    def test_validation_age_out_of_range(self):
        with pytest.raises(ValueError):
            scale_neonatal_pk(
                "Drug", age_days=400, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
            )

    def test_validation_negative_age(self):
        with pytest.raises(ValueError):
            scale_neonatal_pk(
                "Drug", age_days=-1, weight_kg=3.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
            )

    def test_validation_negative_weight(self):
        with pytest.raises(ValueError):
            scale_neonatal_pk(
                "Drug", age_days=14, weight_kg=-1.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0
            )


class TestCompareAgeGroups:
    def test_default_ages(self):
        results = compare_age_groups("Drug", weight_kg=5.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0)
        ages = [r.age_days for r in results]
        assert ages == [1, 7, 14, 28, 90, 180, 270, 365]

    def test_sorted_ascending(self):
        results = compare_age_groups("Drug", weight_kg=5.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0)
        ages = [r.age_days for r in results]
        assert ages == sorted(ages)

    def test_custom_ages(self):
        results = compare_age_groups(
            "Drug", weight_kg=5.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0, ages_days=[3, 10, 20]
        )
        assert len(results) == 3
        assert [r.age_days for r in results] == [3, 10, 20]

    def test_returns_neonatal_pk_results(self):
        results = compare_age_groups("Drug", weight_kg=5.0, adult_cl_L_per_h=5.0, adult_vd_L=50.0)
        for r in results:
            assert isinstance(r, NeonatalPKResult)
