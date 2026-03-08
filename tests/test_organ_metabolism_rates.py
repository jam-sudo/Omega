"""Tests for Phase 553 — Organ-Specific Drug Metabolism Rates."""

import pytest

from omega_pbpk.core.organ_metabolism_rates import (
    ORGAN_PARAMS,
    OrganMetabolismResult,
    predict_organ_metabolism,
    screen_organ_metabolism,
)


class TestPredictOrganMetabolism:
    """Tests for predict_organ_metabolism."""

    def test_returns_organ_metabolism_result(self):
        r = predict_organ_metabolism("DrugA", "liver", 1.0)
        assert isinstance(r, OrganMetabolismResult)

    def test_liver_organ_weight(self):
        r = predict_organ_metabolism("DrugA", "liver", 1.0)
        assert r.organ_weight_g == 1500.0

    def test_liver_blood_flow(self):
        r = predict_organ_metabolism("DrugA", "liver", 1.0)
        assert r.blood_flow_L_per_h == 90.0

    def test_intestine_params(self):
        r = predict_organ_metabolism("DrugA", "intestine", 1.0)
        assert r.organ_weight_g == 1000.0
        assert r.blood_flow_L_per_h == 36.0

    def test_kidney_params(self):
        r = predict_organ_metabolism("DrugA", "kidney", 1.0)
        assert r.organ_weight_g == 300.0
        assert r.blood_flow_L_per_h == 72.0

    def test_lung_params(self):
        r = predict_organ_metabolism("DrugA", "lung", 1.0)
        assert r.organ_weight_g == 500.0
        assert r.blood_flow_L_per_h == 400.0

    def test_extraction_ratio_in_range(self):
        r = predict_organ_metabolism("DrugA", "liver", 5.0, fup=0.5)
        assert 0 <= r.extraction_ratio <= 1

    def test_extraction_ratio_low_clint(self):
        r = predict_organ_metabolism("DrugA", "liver", 0.01, fup=0.1)
        assert r.extraction_ratio < 0.1

    def test_higher_clint_higher_extraction(self):
        r_low = predict_organ_metabolism("DrugA", "liver", 0.5, fup=0.1)
        r_high = predict_organ_metabolism("DrugA", "liver", 5.0, fup=0.1)
        assert r_high.extraction_ratio > r_low.extraction_ratio

    def test_higher_clint_higher_cl_organ(self):
        r_low = predict_organ_metabolism("DrugA", "liver", 0.5)
        r_high = predict_organ_metabolism("DrugA", "liver", 5.0)
        assert r_high.cl_organ_L_per_h > r_low.cl_organ_L_per_h

    def test_well_stirred_model_calculation(self):
        """Verify well-stirred formula: CL = Q*fu*CLint/(Q + fu*CLint)."""
        clint_per_g = 2.0
        fup = 0.2
        r = predict_organ_metabolism("DrugA", "liver", clint_per_g, fup=fup)
        cl_int_total = clint_per_g * 1500.0 * 60.0 / 1000.0  # L/h
        q = 90.0
        expected_cl = q * fup * cl_int_total / (q + fup * cl_int_total)
        assert abs(r.cl_organ_L_per_h - expected_cl) < 1e-6

    def test_extraction_ratio_equals_cl_over_q(self):
        r = predict_organ_metabolism("DrugA", "kidney", 3.0, fup=0.3)
        expected_e = r.cl_organ_L_per_h / r.blood_flow_L_per_h
        assert abs(r.extraction_ratio - expected_e) < 1e-9

    def test_single_organ_contribution_100(self):
        r = predict_organ_metabolism("DrugA", "liver", 1.0)
        assert r.contribution_pct == 100.0

    def test_drug_name_stored(self):
        r = predict_organ_metabolism("Metoprolol", "liver", 1.0)
        assert r.drug_name == "Metoprolol"

    def test_organ_case_insensitive(self):
        r = predict_organ_metabolism("DrugA", "LIVER", 1.0)
        assert r.organ == "liver"

    def test_invalid_organ_raises(self):
        with pytest.raises(ValueError, match="Unknown organ"):
            predict_organ_metabolism("DrugA", "brain", 1.0)

    def test_negative_clint_raises(self):
        with pytest.raises(ValueError):
            predict_organ_metabolism("DrugA", "liver", -1.0)

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            predict_organ_metabolism("", "liver", 1.0)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            predict_organ_metabolism("DrugA", "liver", 1.0, fup=0.0)

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError):
            predict_organ_metabolism("DrugA", "liver", 1.0, fup=1.5)

    def test_zero_clint_gives_zero_clearance(self):
        r = predict_organ_metabolism("DrugA", "liver", 0.0)
        assert r.cl_organ_L_per_h == 0.0
        assert r.extraction_ratio == 0.0


class TestScreenOrganMetabolism:
    """Tests for screen_organ_metabolism."""

    def test_returns_list_of_four(self):
        results = screen_organ_metabolism("DrugA", 1.0)
        assert len(results) == 4

    def test_all_results_are_correct_type(self):
        results = screen_organ_metabolism("DrugA", 1.0)
        for r in results:
            assert isinstance(r, OrganMetabolismResult)

    def test_sorted_by_cl_descending(self):
        results = screen_organ_metabolism("DrugA", 1.0)
        cls = [r.cl_organ_L_per_h for r in results]
        assert cls == sorted(cls, reverse=True)

    def test_contributions_sum_to_100(self):
        results = screen_organ_metabolism("DrugA", 2.0, fup=0.3)
        total = sum(r.contribution_pct for r in results)
        assert abs(total - 100.0) < 1e-6

    def test_all_four_organs_present(self):
        results = screen_organ_metabolism("DrugA", 1.0)
        organs = {r.organ for r in results}
        assert organs == {"liver", "intestine", "kidney", "lung"}

    def test_higher_clint_increases_all_clearances(self):
        r_low = screen_organ_metabolism("DrugA", 0.5)
        r_high = screen_organ_metabolism("DrugA", 5.0)
        for organ in ORGAN_PARAMS:
            cl_low = next(r.cl_organ_L_per_h for r in r_low if r.organ == organ)
            cl_high = next(r.cl_organ_L_per_h for r in r_high if r.organ == organ)
            assert cl_high > cl_low
