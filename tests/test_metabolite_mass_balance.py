"""Tests for Phase 664 - Metabolite Mass Balance Tracker."""

import pytest

from omega_pbpk.core.metabolite_mass_balance import (
    MetaboliteMassBalanceResult,
    simulate_metabolite_mass_balance,
)


def _default():
    return simulate_metabolite_mass_balance(
        drug_name="DrugB",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        fm_values=[0.6, 0.3],
        cl_met_L_per_h_values=[2.0, 1.0],
        vd_met_L_values=[30.0, 20.0],
    )


class TestReturnType:
    def test_return_type(self):
        r = _default()
        assert isinstance(r, MetaboliteMassBalanceResult)

    def test_list_lengths(self):
        r = _default()
        n = len(r.times_h)
        assert len(r.c_parent_mg_L) == n
        assert len(r.mass_parent_mg) == n
        assert len(r.mass_recovered_pct) == n

    def test_drug_name(self):
        r = _default()
        assert r.drug_name == "DrugB"

    def test_dose_preserved(self):
        r = _default()
        assert r.dose_mg == 100.0

    def test_n_metabolites(self):
        r = _default()
        assert r.n_metabolites == 2


class TestMetabolites:
    def test_c_metabolites_length(self):
        r = _default()
        assert len(r.c_metabolites_mg_L) == r.n_metabolites

    def test_each_metabolite_list_length(self):
        r = _default()
        n = len(r.times_h)
        for m_list in r.c_metabolites_mg_L:
            assert len(m_list) == n

    def test_auc_metabolites_length(self):
        r = _default()
        assert len(r.auc_metabolites_mg_h_per_L) == r.n_metabolites

    def test_fm_values_preserved(self):
        r = _default()
        assert r.fm_values == [0.6, 0.3]

    def test_mass_metabolites_length(self):
        r = _default()
        assert len(r.mass_metabolites_mg) == r.n_metabolites

    def test_mass_metabolites_sublists_length(self):
        r = _default()
        n = len(r.times_h)
        for m_list in r.mass_metabolites_mg:
            assert len(m_list) == n


class TestMassBalance:
    def test_mass_balance_final_reasonable(self):
        r = _default()
        assert r.mass_balance_final_pct > 0

    def test_mass_recovered_starts_near_90(self):
        r = _default()
        # f_oral = 0.9, so initial mass = dose * 0.9 = 90%
        assert abs(r.mass_recovered_pct[0] - 90.0) < 1.0

    def test_mass_recovered_decreases(self):
        r = _default()
        assert r.mass_recovered_pct[-1] < r.mass_recovered_pct[0]

    def test_custom_f_oral(self):
        r = simulate_metabolite_mass_balance(
            "D",
            100.0,
            5.0,
            50.0,
            fm_values=[0.5],
            cl_met_L_per_h_values=[2.0],
            vd_met_L_values=[30.0],
            f_oral=0.5,
        )
        assert abs(r.mass_recovered_pct[0] - 50.0) < 1.0


class TestAUC:
    def test_auc_parent_positive(self):
        r = _default()
        assert r.auc_parent_mg_h_per_L > 0

    def test_auc_metabolites_positive(self):
        r = _default()
        for auc in r.auc_metabolites_mg_h_per_L:
            assert auc > 0

    def test_higher_fm_higher_metabolite_auc(self):
        r2 = simulate_metabolite_mass_balance(
            "D",
            100.0,
            5.0,
            50.0,
            fm_values=[0.7, 0.2],
            cl_met_L_per_h_values=[2.0, 2.0],
            vd_met_L_values=[30.0, 30.0],
        )
        assert r2.auc_metabolites_mg_h_per_L[0] > r2.auc_metabolites_mg_h_per_L[1]


class TestValidation:
    def test_fm_sum_exceeds_one_raises(self):
        with pytest.raises(ValueError):
            simulate_metabolite_mass_balance(
                "D",
                100.0,
                5.0,
                50.0,
                fm_values=[0.6, 0.5],
                cl_met_L_per_h_values=[2.0, 1.0],
                vd_met_L_values=[30.0, 20.0],
            )

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_metabolite_mass_balance(
                "D",
                0.0,
                5.0,
                50.0,
                fm_values=[0.5],
                cl_met_L_per_h_values=[2.0],
                vd_met_L_values=[30.0],
            )

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_metabolite_mass_balance(
                "D",
                -10.0,
                5.0,
                50.0,
                fm_values=[0.5],
                cl_met_L_per_h_values=[2.0],
                vd_met_L_values=[30.0],
            )

    def test_mismatched_cl_met_length_raises(self):
        with pytest.raises(ValueError):
            simulate_metabolite_mass_balance(
                "D",
                100.0,
                5.0,
                50.0,
                fm_values=[0.5, 0.3],
                cl_met_L_per_h_values=[2.0],
                vd_met_L_values=[30.0, 20.0],
            )

    def test_mismatched_vd_met_length_raises(self):
        with pytest.raises(ValueError):
            simulate_metabolite_mass_balance(
                "D",
                100.0,
                5.0,
                50.0,
                fm_values=[0.5],
                cl_met_L_per_h_values=[2.0],
                vd_met_L_values=[30.0, 20.0],
            )


class TestSingleMetabolite:
    def test_single_metabolite(self):
        r = simulate_metabolite_mass_balance(
            "D",
            100.0,
            5.0,
            50.0,
            fm_values=[0.9],
            cl_met_L_per_h_values=[2.0],
            vd_met_L_values=[30.0],
        )
        assert r.n_metabolites == 1
        assert len(r.c_metabolites_mg_L) == 1
        assert len(r.mass_metabolites_mg) == 1
        assert r.auc_parent_mg_h_per_L > 0
