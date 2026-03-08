"""Tests for Phase 476 — Topoisomerase Inhibitor PK/PD."""

import math

import pytest

from omega_pbpk.clinical.topoisomerase_pk_pd import (
    TopoisomerasePKPDResult,
    compare_topoisomerase_drugs,
    simulate_topoisomerase_pk_pd,
)

DRUGS = ["irinotecan", "topotecan", "doxorubicin", "etoposide"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _run(drug="irinotecan", dose=100.0, **kwargs):
    return simulate_topoisomerase_pk_pd(drug_name=drug, dose_mg=dose, **kwargs)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_correct_type(self):
        result = _run()
        assert isinstance(result, TopoisomerasePKPDResult)

    def test_drug_name_stored(self):
        result = _run(drug="topotecan")
        assert result.drug_name == "topotecan"

    def test_dose_stored(self):
        result = _run(dose=50.0)
        assert result.dose_mg == 50.0

    def test_drug_class_populated(self):
        r_i = _run(drug="irinotecan")
        assert r_i.drug_class == "topo_I"
        r_d = _run(drug="doxorubicin")
        assert r_d.drug_class == "topo_II"


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------
class TestPKMetrics:
    def test_cmax_positive(self):
        for drug in DRUGS:
            result = _run(drug=drug)
            assert result.cmax_central_mg_L > 0.0

    def test_auc_positive(self):
        for drug in DRUGS:
            result = _run(drug=drug)
            assert result.auc_central_mg_h_per_L > 0.0

    def test_higher_dose_higher_auc(self):
        r_low = _run(dose=50.0)
        r_high = _run(dose=200.0)
        assert r_high.auc_central_mg_h_per_L > r_low.auc_central_mg_h_per_L

    def test_double_dose_approximately_double_auc(self):
        r1 = _run(dose=100.0)
        r2 = _run(dose=200.0)
        ratio = r2.auc_central_mg_h_per_L / r1.auc_central_mg_h_per_L
        assert 1.9 <= ratio <= 2.1

    def test_cmax_at_t0(self):
        # IV bolus → Cmax = dose / Vd at t=0
        result = _run(dose=100.0, vd_central_L=10.0)
        expected_c0 = 100.0 / 10.0  # mg/L
        assert math.isclose(result.cmax_central_mg_L, expected_c0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# PD metrics
# ---------------------------------------------------------------------------
class TestPDMetrics:
    def test_kill_fraction_in_range(self):
        for drug in DRUGS:
            result = _run(drug=drug)
            assert 0.0 <= result.tumor_cell_kill_fraction <= 1.0

    def test_higher_dose_higher_kill(self):
        r_low = _run(dose=10.0)
        r_high = _run(dose=500.0)
        assert r_high.tumor_cell_kill_fraction >= r_low.tumor_cell_kill_fraction

    def test_kill_rate_starts_positive(self):
        # Drug present at t=0 → kill rate > 0
        result = _run()
        assert result.kill_rate_per_h[0] > 0.0

    def test_ic50_stored(self):
        result = _run(drug="doxorubicin")
        assert result.ic50_mg_L == pytest.approx(0.05, rel=1e-6)

    def test_emax_stored(self):
        result = _run(drug="irinotecan")
        assert result.emax == pytest.approx(0.9, rel=1e-6)


# ---------------------------------------------------------------------------
# Time-series
# ---------------------------------------------------------------------------
class TestTimeSeries:
    def test_times_start_at_zero(self):
        result = _run()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = _run(t_end_h=48.0, dt_h=0.1)
        assert abs(result.times_h[-1] - 48.0) <= 0.15

    def test_c_central_declines(self):
        result = _run(t_end_h=72.0)
        # Final concentration should be less than initial
        assert result.c_central_mg_L[-1] < result.c_central_mg_L[0]

    def test_kill_rate_non_negative(self):
        result = _run()
        for kr in result.kill_rate_per_h:
            assert kr >= 0.0


# ---------------------------------------------------------------------------
# compare_topoisomerase_drugs
# ---------------------------------------------------------------------------
class TestCompareTopoisomeraseDrugs:
    def test_returns_list_of_four(self):
        results = compare_topoisomerase_drugs(dose_mg=100.0)
        assert len(results) == 4

    def test_all_correct_type(self):
        results = compare_topoisomerase_drugs()
        for r in results:
            assert isinstance(r, TopoisomerasePKPDResult)

    def test_sorted_by_kill_fraction_descending(self):
        results = compare_topoisomerase_drugs()
        kf = [r.tumor_cell_kill_fraction for r in results]
        assert kf == sorted(kf, reverse=True)

    def test_all_drugs_represented(self):
        results = compare_topoisomerase_drugs()
        names = {r.drug_name for r in results}
        assert names == set(DRUGS)

    def test_default_dose_used(self):
        results = compare_topoisomerase_drugs()
        for r in results:
            assert r.dose_mg == 100.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_unknown_drug_raises(self):
        with pytest.raises(ValueError, match="Unknown drug"):
            simulate_topoisomerase_pk_pd("paclitaxel", dose_mg=100.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_topoisomerase_pk_pd("irinotecan", dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_topoisomerase_pk_pd("irinotecan", dose_mg=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_central_L"):
            simulate_topoisomerase_pk_pd("irinotecan", dose_mg=100.0, vd_central_L=0.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_central_L"):
            simulate_topoisomerase_pk_pd("irinotecan", dose_mg=100.0, vd_central_L=-5.0)
