"""Tests for pharmacogenomics module (Phase 112)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pharmacogenomics import (
    PGxResult,
    batch_pgx_screen,
    cyp_phenotype_distribution,
    predict_pgx_dose,
)


class TestPredictPGxDose:
    def test_returns_result_type(self):
        r = predict_pgx_dose("Codeine", "CYP2D6", "*1", "*1", 100.0)
        assert isinstance(r, PGxResult)

    def test_em_phenotype_for_star1_star1(self):
        r = predict_pgx_dose("Drug", "CYP2D6", "*1", "*1", 100.0)
        assert r.phenotype == "EM"

    def test_pm_phenotype_for_null_alleles(self):
        r = predict_pgx_dose("Drug", "CYP2D6", "*4", "*4", 100.0)
        assert r.phenotype == "PM"

    def test_um_phenotype_for_duplication(self):
        r = predict_pgx_dose("Drug", "CYP2D6", "*1xN", "*1", 100.0)
        assert r.phenotype == "UM"

    def test_im_phenotype(self):
        r = predict_pgx_dose("Drug", "CYP2D6", "*1", "*4", 100.0)
        assert r.phenotype in ("IM", "EM")

    def test_em_dose_equals_standard(self):
        r = predict_pgx_dose("Drug", "CYP2D6", "*1", "*1", 100.0, fm=0.8)
        assert abs(r.recommended_dose_mg - 100.0) < 0.1

    def test_pm_dose_lower_than_em(self):
        r_em = predict_pgx_dose("Drug", "CYP2D6", "*1", "*1", 100.0)
        r_pm = predict_pgx_dose("Drug", "CYP2D6", "*4", "*4", 100.0)
        assert r_pm.recommended_dose_mg < r_em.recommended_dose_mg

    def test_um_dose_higher_than_em(self):
        r_em = predict_pgx_dose("Drug", "CYP2D6", "*1", "*1", 100.0)
        r_um = predict_pgx_dose("Drug", "CYP2D6", "*1xN", "*1xN", 100.0)
        assert r_um.recommended_dose_mg > r_em.recommended_dose_mg

    def test_cl_scaling_factor_positive(self):
        r = predict_pgx_dose("Drug", "CYP2D6", "*1", "*4", 100.0)
        assert r.cl_scaling_factor > 0.0

    def test_activity_score_non_negative(self):
        r = predict_pgx_dose("Drug", "CYP2D6", "*4", "*4", 100.0)
        assert r.activity_score >= 0.0

    def test_clinical_note_is_string(self):
        r = predict_pgx_dose("Drug", "CYP2D6", "*1", "*1", 100.0)
        assert isinstance(r.clinical_note, str)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            predict_pgx_dose("Drug", "CYP2D6", "*1", "*1", 0.0)

    def test_invalid_fm_raises(self):
        with pytest.raises(ValueError):
            predict_pgx_dose("Drug", "CYP2D6", "*1", "*1", 100.0, fm=0.0)


class TestBatchScreen:
    def test_returns_list(self):
        patients = [
            {"drug_name": "Drug", "cyp_enzyme": "CYP2D6", "allele1": "*1", "allele2": "*4", "standard_dose_mg": 100.0},
            {"drug_name": "Drug", "cyp_enzyme": "CYP2D6", "allele1": "*4", "allele2": "*4", "standard_dose_mg": 100.0},
        ]
        results = batch_pgx_screen(patients)
        assert len(results) == 2

    def test_empty_returns_empty(self):
        assert batch_pgx_screen([]) == []


class TestPhenotypeDistribution:
    def test_returns_dict_with_four_phenotypes(self):
        dist = cyp_phenotype_distribution("CYP2D6", "caucasian")
        for ph in ("UM", "EM", "IM", "PM"):
            assert ph in dist

    def test_frequencies_sum_to_1(self):
        dist = cyp_phenotype_distribution("CYP2D6", "caucasian")
        assert abs(sum(dist.values()) - 1.0) < 0.01

    def test_unknown_enzyme_returns_default(self):
        dist = cyp_phenotype_distribution("CYP99X", "caucasian")
        assert "EM" in dist
