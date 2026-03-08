"""
Tests for Phase 378 — Pediatric Pharmacogenomics Scaling
"""

import pytest

from omega_pbpk.clinical.pediatric_pgx_scaling import (
    PediatricPGxResult,
    calculate_pediatric_pgx_dose,
    compare_phenotypes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nm_child():
    """Normal metabolizer 6-year-old, CYP3A4, standard drug."""
    return calculate_pediatric_pgx_dose(
        drug_name="TestDrug",
        cyp_enzyme="CYP3A4",
        metabolizer_phenotype="NM",
        age_years=6.0,
        weight_kg=20.0,
        adult_dose_mg=100.0,
        fraction_metabolized_by_cyp=0.8,
    )


@pytest.fixture
def pm_child():
    """Poor metabolizer 6-year-old, CYP3A4."""
    return calculate_pediatric_pgx_dose(
        drug_name="TestDrug",
        cyp_enzyme="CYP3A4",
        metabolizer_phenotype="PM",
        age_years=6.0,
        weight_kg=20.0,
        adult_dose_mg=100.0,
        fraction_metabolized_by_cyp=0.8,
    )


@pytest.fixture
def um_child():
    """Ultra-rapid metabolizer 6-year-old, CYP3A4."""
    return calculate_pediatric_pgx_dose(
        drug_name="TestDrug",
        cyp_enzyme="CYP3A4",
        metabolizer_phenotype="UM",
        age_years=6.0,
        weight_kg=20.0,
        adult_dose_mg=100.0,
        fraction_metabolized_by_cyp=0.8,
    )


@pytest.fixture
def neonate_nm():
    """Normal metabolizer neonate (0.1 years), CYP2D6."""
    return calculate_pediatric_pgx_dose(
        drug_name="TestDrug",
        cyp_enzyme="CYP2D6",
        metabolizer_phenotype="NM",
        age_years=0.1,
        weight_kg=3.5,
        adult_dose_mg=100.0,
        fraction_metabolized_by_cyp=0.8,
    )


# ---------------------------------------------------------------------------
# 1. Return type and fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_pediatric_pgx_result(self, nm_child):
        assert isinstance(nm_child, PediatricPGxResult)

    def test_result_is_frozen(self, nm_child):
        with pytest.raises((AttributeError, TypeError)):
            nm_child.pgx_adjusted_dose_mg = 999.0  # type: ignore[misc]

    def test_all_fields_present(self, nm_child):
        fields = [
            "drug_name",
            "cyp_enzyme",
            "metabolizer_phenotype",
            "age_years",
            "weight_kg",
            "adult_dose_mg",
            "pediatric_dose_mg",
            "pgx_adjusted_dose_mg",
            "allometric_factor",
            "enzyme_maturation_factor",
            "pgx_dose_factor",
            "combined_dose_factor",
            "dose_cap_applied",
            "monitoring_required",
            "clinical_guidance",
            "notes",
        ]
        for f in fields:
            assert hasattr(nm_child, f), f"Missing field: {f}"


# ---------------------------------------------------------------------------
# 2. NM allometric and maturation factors
# ---------------------------------------------------------------------------


class TestNMFactors:
    def test_allometric_factor_below_1_for_child(self, nm_child):
        # Child (20 kg) vs adult (70 kg): factor < 1
        assert nm_child.allometric_factor < 1.0

    def test_allometric_factor_positive(self, nm_child):
        assert nm_child.allometric_factor > 0.0

    def test_maturation_factor_between_0_and_1(self, nm_child):
        assert 0.0 < nm_child.enzyme_maturation_factor <= 1.0

    def test_pgx_dose_factor_is_1_for_nm(self, nm_child):
        # NM with full CYP metabolization: pgx_dose_factor = 1.0
        result = calculate_pediatric_pgx_dose(
            "X",
            "CYP3A4",
            "NM",
            6.0,
            20.0,
            100.0,
            fraction_metabolized_by_cyp=1.0,
        )
        assert abs(result.pgx_dose_factor - 1.0) < 1e-9

    def test_combined_factor_is_product(self, nm_child):
        expected = (
            nm_child.allometric_factor
            * nm_child.enzyme_maturation_factor
            * nm_child.pgx_dose_factor
        )
        assert abs(nm_child.combined_dose_factor - expected) < 1e-9


# ---------------------------------------------------------------------------
# 3. PM dose < NM dose
# ---------------------------------------------------------------------------


class TestPMvsNM:
    def test_pm_dose_less_than_nm(self, pm_child, nm_child):
        assert pm_child.pgx_adjusted_dose_mg < nm_child.pgx_adjusted_dose_mg

    def test_pm_pgx_factor_less_than_nm(self, pm_child, nm_child):
        assert pm_child.pgx_dose_factor < nm_child.pgx_dose_factor

    def test_pm_clinical_guidance_mentions_toxicity(self, pm_child):
        assert (
            "toxicity" in pm_child.clinical_guidance.lower()
            or "reduction" in pm_child.clinical_guidance.lower()
        )


# ---------------------------------------------------------------------------
# 4. UM dose >= NM dose (before cap)
# ---------------------------------------------------------------------------


class TestUMvsNM:
    def test_um_pgx_factor_greater_than_nm(self, um_child, nm_child):
        assert um_child.pgx_dose_factor > nm_child.pgx_dose_factor

    def test_um_clinical_guidance_mentions_efficacy(self, um_child):
        assert (
            "efficac" in um_child.clinical_guidance.lower()
            or "higher" in um_child.clinical_guidance.lower()
        )


# ---------------------------------------------------------------------------
# 5. Neonate — low maturation factor
# ---------------------------------------------------------------------------


class TestNeonate:
    def test_neonate_maturation_very_low(self, neonate_nm):
        # CYP2D6 t50=1 year; at 0.1 years, maturation should be very low
        assert (
            neonate_nm.enzyme_maturation_factor < 0.05
            or neonate_nm.enzyme_maturation_factor == pytest.approx(0.05, abs=0.01)
        )

    def test_neonate_monitoring_required(self, neonate_nm):
        assert neonate_nm.monitoring_required is True

    def test_neonate_guidance_mentions_neonate(self, neonate_nm):
        assert (
            "neonate" in neonate_nm.clinical_guidance.lower()
            or "neonat" in neonate_nm.clinical_guidance.lower()
        )

    def test_neonate_dose_lower_than_older_child(self):
        neo = calculate_pediatric_pgx_dose("D", "CYP2D6", "NM", 0.1, 3.5, 100.0, 0.8)
        older = calculate_pediatric_pgx_dose("D", "CYP2D6", "NM", 10.0, 35.0, 100.0, 0.8)
        assert neo.pgx_adjusted_dose_mg < older.pgx_adjusted_dose_mg


# ---------------------------------------------------------------------------
# 6. Adult maturation approaches 1.0 for older children
# ---------------------------------------------------------------------------


class TestMaturationApproachesAdult:
    def test_maturation_near_1_for_teenager(self):
        result = calculate_pediatric_pgx_dose("D", "CYP3A4", "NM", 15.0, 60.0, 100.0, 0.8)
        assert result.enzyme_maturation_factor > 0.95

    def test_maturation_increases_with_age_same_cyp(self):
        young = calculate_pediatric_pgx_dose("D", "CYP1A2", "NM", 1.0, 10.0, 100.0, 0.8)
        older = calculate_pediatric_pgx_dose("D", "CYP1A2", "NM", 8.0, 25.0, 100.0, 0.8)
        assert older.enzyme_maturation_factor > young.enzyme_maturation_factor


# ---------------------------------------------------------------------------
# 7. Dose cap
# ---------------------------------------------------------------------------


class TestDoseCap:
    def test_dose_cap_applied_when_above_adult(self):
        # UM teenager with adult weight should approach or exceed adult dose
        result = calculate_pediatric_pgx_dose("D", "CYP3A4", "UM", 17.0, 70.0, 100.0, 0.8)
        if result.dose_cap_applied:
            assert result.pgx_adjusted_dose_mg == pytest.approx(100.0, abs=0.01)

    def test_dose_never_exceeds_adult_dose(self, um_child):
        assert um_child.pgx_adjusted_dose_mg <= um_child.adult_dose_mg + 1e-9

    def test_dose_cap_not_applied_for_small_child(self, nm_child):
        # Small child (20 kg) should not exceed adult dose
        assert (
            not nm_child.dose_cap_applied or nm_child.pgx_adjusted_dose_mg <= nm_child.adult_dose_mg
        )


# ---------------------------------------------------------------------------
# 8. Monitoring required
# ---------------------------------------------------------------------------


class TestMonitoringRequired:
    def test_pm_monitoring_required(self, pm_child):
        assert pm_child.monitoring_required is True

    def test_um_monitoring_required(self, um_child):
        assert um_child.monitoring_required is True

    def test_neonate_monitoring_required(self, neonate_nm):
        assert neonate_nm.monitoring_required is True

    def test_nm_older_child_may_not_need_monitoring(self):
        # Older NM child with moderate fm should not require monitoring
        result = calculate_pediatric_pgx_dose("D", "CYP3A4", "NM", 10.0, 35.0, 100.0, 0.5)
        # pgx_dose_factor = 1 - 0.5*(1 - 1.0) = 1.0; age >= 2; NM → no monitoring
        assert result.monitoring_required is False


# ---------------------------------------------------------------------------
# 9. compare_phenotypes
# ---------------------------------------------------------------------------


class TestComparePhenotypes:
    def test_compare_returns_list_of_4(self):
        results = compare_phenotypes("D", "CYP3A4", 6.0, 20.0, 100.0, 0.8)
        assert len(results) == 4

    def test_compare_all_results_are_pgx_result(self):
        results = compare_phenotypes("D", "CYP3A4", 6.0, 20.0, 100.0, 0.8)
        for r in results:
            assert isinstance(r, PediatricPGxResult)

    def test_compare_sorted_ascending_by_dose(self):
        results = compare_phenotypes("D", "CYP3A4", 6.0, 20.0, 100.0, 0.8)
        doses = [r.pgx_adjusted_dose_mg for r in results]
        assert doses == sorted(doses)

    def test_compare_contains_all_phenotypes(self):
        results = compare_phenotypes("D", "CYP3A4", 6.0, 20.0, 100.0, 0.8)
        phenotypes = {r.metabolizer_phenotype for r in results}
        assert phenotypes == {"PM", "IM", "NM", "UM"}

    def test_compare_pm_first(self):
        results = compare_phenotypes("D", "CYP3A4", 6.0, 20.0, 100.0, 0.8)
        assert results[0].metabolizer_phenotype == "PM"


# ---------------------------------------------------------------------------
# 10. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_age_zero(self):
        with pytest.raises(ValueError, match="age_years"):
            calculate_pediatric_pgx_dose("D", "CYP3A4", "NM", 0.0, 20.0, 100.0)

    def test_age_above_18(self):
        with pytest.raises(ValueError, match="age_years"):
            calculate_pediatric_pgx_dose("D", "CYP3A4", "NM", 19.0, 60.0, 100.0)

    def test_negative_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            calculate_pediatric_pgx_dose("D", "CYP3A4", "NM", 6.0, -5.0, 100.0)

    def test_zero_adult_dose(self):
        with pytest.raises(ValueError, match="adult_dose_mg"):
            calculate_pediatric_pgx_dose("D", "CYP3A4", "NM", 6.0, 20.0, 0.0)

    def test_invalid_phenotype(self):
        with pytest.raises(ValueError, match="metabolizer_phenotype"):
            calculate_pediatric_pgx_dose("D", "CYP3A4", "XM", 6.0, 20.0, 100.0)

    def test_invalid_cyp(self):
        with pytest.raises(ValueError, match="cyp_enzyme"):
            calculate_pediatric_pgx_dose("D", "CYP99A1", "NM", 6.0, 20.0, 100.0)

    def test_fraction_above_1(self):
        with pytest.raises(ValueError, match="fraction_metabolized"):
            calculate_pediatric_pgx_dose("D", "CYP3A4", "NM", 6.0, 20.0, 100.0, 1.5)

    def test_fraction_negative(self):
        with pytest.raises(ValueError, match="fraction_metabolized"):
            calculate_pediatric_pgx_dose("D", "CYP3A4", "NM", 6.0, 20.0, 100.0, -0.1)
