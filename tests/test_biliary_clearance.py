"""Tests for core.biliary_clearance — biliary CL model (Phase 249)."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.core.biliary_clearance import (
    BiliaryERResult,
    BiliaryPKResult,
    biliary_extraction_ratio,
    fe_vs_fm_sensitivity,
    simulate_biliary_pk,
)


# ---------------------------------------------------------------------------
# biliary_extraction_ratio tests
# ---------------------------------------------------------------------------


class TestBiliaryExtractionRatio:
    """Tests for biliary_extraction_ratio()."""

    def test_er_in_range(self):
        result = biliary_extraction_ratio(
            q_liver_L_per_h=90.0,
            cl_uptake_L_per_h=50.0,
            cl_bile_L_per_h=20.0,
            fu_plasma=0.1,
        )
        assert 0.0 <= result.er_hepatic <= 1.0

    def test_f_bile_plus_f_metabolized_sum_to_one(self):
        result = biliary_extraction_ratio(
            q_liver_L_per_h=90.0,
            cl_uptake_L_per_h=50.0,
            cl_bile_L_per_h=20.0,
            fu_plasma=0.1,
        )
        assert math.isclose(result.f_bile + result.f_metabolized, 1.0, abs_tol=1e-9)

    def test_higher_cl_bile_increases_er(self):
        common = dict(q_liver_L_per_h=90.0, cl_uptake_L_per_h=30.0, fu_plasma=0.2)
        low = biliary_extraction_ratio(cl_bile_L_per_h=5.0, **common)
        high = biliary_extraction_ratio(cl_bile_L_per_h=50.0, **common)
        assert high.er_hepatic > low.er_hepatic

    def test_zero_bile_cl_gives_f_bile_zero(self):
        result = biliary_extraction_ratio(
            q_liver_L_per_h=90.0,
            cl_uptake_L_per_h=40.0,
            cl_bile_L_per_h=0.0,
            fu_plasma=0.1,
        )
        assert result.f_bile == 0.0
        assert result.f_metabolized == 1.0

    def test_very_high_cl_er_approaches_one(self):
        result = biliary_extraction_ratio(
            q_liver_L_per_h=90.0,
            cl_uptake_L_per_h=5000.0,
            cl_bile_L_per_h=5000.0,
            fu_plasma=1.0,
        )
        assert result.er_hepatic > 0.99

    def test_result_type(self):
        result = biliary_extraction_ratio(
            q_liver_L_per_h=90.0,
            cl_uptake_L_per_h=30.0,
            cl_bile_L_per_h=10.0,
            fu_plasma=0.2,
        )
        assert isinstance(result, BiliaryERResult)

    def test_cl_hepatic_total_positive(self):
        result = biliary_extraction_ratio(
            q_liver_L_per_h=90.0,
            cl_uptake_L_per_h=30.0,
            cl_bile_L_per_h=10.0,
            fu_plasma=0.2,
        )
        assert result.cl_hepatic_total > 0

    def test_invalid_q_liver_raises(self):
        with pytest.raises(ValueError, match="q_liver"):
            biliary_extraction_ratio(
                q_liver_L_per_h=0.0,
                cl_uptake_L_per_h=30.0,
                cl_bile_L_per_h=10.0,
                fu_plasma=0.2,
            )

    def test_invalid_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            biliary_extraction_ratio(
                q_liver_L_per_h=90.0,
                cl_uptake_L_per_h=30.0,
                cl_bile_L_per_h=10.0,
                fu_plasma=0.0,
            )


# ---------------------------------------------------------------------------
# simulate_biliary_pk tests
# ---------------------------------------------------------------------------


class TestSimulateBiliaryPK:
    """Tests for simulate_biliary_pk()."""

    def _default_result(self, **kwargs):
        params = dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            cl_bile_fraction=0.3,
            vd_L=50.0,
            route="iv",
            erc_fraction=0.5,
            k_reabs_per_h=0.1,
            t_end_h=24.0,
            dt_h=0.1,
        )
        params.update(kwargs)
        return simulate_biliary_pk(**params)

    def test_result_type(self):
        result = self._default_result()
        assert isinstance(result, BiliaryPKResult)

    def test_cmax_positive(self):
        result = self._default_result()
        assert result.cmax > 0

    def test_auc_positive(self):
        result = self._default_result()
        assert result.auc > 0

    def test_dose_linearity_auc(self):
        r1 = self._default_result(dose_mg=100.0)
        r2 = self._default_result(dose_mg=200.0)
        ratio = r2.auc / r1.auc
        assert 1.8 < ratio < 2.2, f"AUC linearity ratio = {ratio}"

    def test_dose_linearity_cmax(self):
        r1 = self._default_result(dose_mg=100.0)
        r2 = self._default_result(dose_mg=200.0)
        ratio = r2.cmax / r1.cmax
        assert 1.8 < ratio < 2.2, f"Cmax linearity ratio = {ratio}"

    def test_erc_zero_no_secondary_peak(self):
        result = self._default_result(erc_fraction=0.0)
        assert not result.secondary_peak_present

    def test_cl_bile_fraction_zero(self):
        result = self._default_result(cl_bile_fraction=0.0)
        # No biliary CL → no ERC even if erc_fraction > 0
        assert isinstance(result, BiliaryPKResult)
        assert result.cmax > 0

    def test_times_h_length_matches_conc(self):
        result = self._default_result()
        assert len(result.times_h) == len(result.c_plasma_mg_L)

    def test_tmax_nonnegative(self):
        result = self._default_result()
        assert result.tmax_h >= 0

    def test_secondary_peak_flag_is_bool(self):
        result = self._default_result()
        assert isinstance(result.secondary_peak_present, bool)

    def test_t_secondary_peak_type(self):
        result = self._default_result()
        assert result.t_secondary_peak_h is None or isinstance(result.t_secondary_peak_h, float)

    def test_notes_is_string(self):
        result = self._default_result()
        assert isinstance(result.notes, str)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            self._default_result(dose_mg=0.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_hepatic"):
            self._default_result(cl_hepatic_L_per_h=-1.0)

    def test_invalid_cl_bile_fraction_raises(self):
        with pytest.raises(ValueError, match="cl_bile_fraction"):
            self._default_result(cl_bile_fraction=1.5)

    def test_invalid_erc_fraction_raises(self):
        with pytest.raises(ValueError, match="erc_fraction"):
            self._default_result(erc_fraction=-0.1)

    def test_oral_route_works(self):
        result = simulate_biliary_pk(
            drug_name="OralDrug",
            dose_mg=50.0,
            cl_hepatic_L_per_h=3.0,
            cl_bile_fraction=0.2,
            vd_L=40.0,
            route="oral",
            ka_per_h=1.2,
            erc_fraction=0.3,
            t_end_h=24.0,
        )
        assert result.cmax > 0
        assert result.tmax_h > 0  # oral has delay


# ---------------------------------------------------------------------------
# fe_vs_fm_sensitivity tests
# ---------------------------------------------------------------------------


class TestFeVsFmSensitivity:
    """Tests for fe_vs_fm_sensitivity()."""

    def _run_sensitivity(self, fractions=None):
        if fractions is None:
            fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        return fe_vs_fm_sensitivity(
            drug_name="SensitivityDrug",
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
            bile_fractions=fractions,
            dose_mg=100.0,
            route="iv",
            erc_fraction=0.4,
            t_end_h=24.0,
        )

    def test_returns_list(self):
        results = self._run_sensitivity()
        assert isinstance(results, list)

    def test_length_matches_input(self):
        fractions = [0.1, 0.3, 0.7]
        results = self._run_sensitivity(fractions=fractions)
        assert len(results) == len(fractions)

    def test_sorted_ascending_by_bile_fraction(self):
        fractions = [0.7, 0.1, 0.4, 0.0]
        results = self._run_sensitivity(fractions=fractions)
        bfs = [r.cl_bile_fraction for r in results]
        assert bfs == sorted(bfs)

    def test_all_results_are_biliary_pk_result(self):
        results = self._run_sensitivity()
        assert all(isinstance(r, BiliaryPKResult) for r in results)

    def test_empty_list_raises(self):
        with pytest.raises((ValueError, Exception)):
            fe_vs_fm_sensitivity(
                drug_name="D",
                cl_hepatic_L_per_h=5.0,
                vd_L=50.0,
                bile_fractions=[],
            )
