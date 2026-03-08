"""Tests for Phase 524 — Intraperitoneal Drug Delivery PK."""

import pytest

from omega_pbpk.core.intraperitoneal_pk import (
    IntraperitonealPKResult,
    compare_ka_ip,
    simulate_intraperitoneal_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_hepatic_L_per_h=10.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    params.update(kwargs)
    return simulate_intraperitoneal_pk(**params)


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_ip_result(self):
        r = default_result()
        assert isinstance(r, IntraperitonealPKResult)

    def test_drug_name_stored(self):
        r = default_result(drug_name="DrugIP")
        assert r.drug_name == "DrugIP"

    def test_dose_stored(self):
        r = default_result(dose_mg=250.0)
        assert r.dose_mg == 250.0

    def test_ka_ip_stored(self):
        r = default_result(ka_ip_per_h=1.0)
        assert r.ka_ip_per_h == 1.0

    def test_times_h_is_list(self):
        r = default_result()
        assert isinstance(r.times_h, list)

    def test_c_systemic_is_list(self):
        r = default_result()
        assert isinstance(r.c_systemic_mg_L, list)

    def test_notes_is_str(self):
        r = default_result()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Hepatic extraction ratio
# ---------------------------------------------------------------------------


class TestHepExtractionRatio:
    def test_e_hepatic_between_0_and_1(self):
        r = default_result()
        assert 0.0 <= r.e_hepatic <= 1.0

    def test_higher_cl_hepatic_higher_e(self):
        r_low = default_result(cl_hepatic_L_per_h=1.0)
        r_high = default_result(cl_hepatic_L_per_h=50.0)
        assert r_high.e_hepatic > r_low.e_hepatic


# ---------------------------------------------------------------------------
# Time-series tests
# ---------------------------------------------------------------------------


class TestTimeSeries:
    def test_c_sys_starts_at_zero(self):
        r = default_result()
        assert r.c_systemic_mg_L[0] == 0.0

    def test_times_starts_at_zero(self):
        r = default_result()
        assert r.times_h[0] == 0.0

    def test_c_sys_rises_from_zero(self):
        r = default_result()
        assert max(r.c_systemic_mg_L) > 0.0

    def test_c_sys_rises_then_falls(self):
        r = default_result()
        # Tmax should be strictly between 0 and end time
        assert r.tmax_h > 0.0
        assert r.tmax_h < r.times_h[-1]
        # Concentration at end is less than Cmax
        assert r.c_systemic_mg_L[-1] < r.cmax_mg_L

    def test_list_lengths_match(self):
        r = default_result()
        assert len(r.times_h) == len(r.c_systemic_mg_L)


# ---------------------------------------------------------------------------
# PK metric tests
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_auc_positive(self):
        r = default_result()
        assert r.auc_mg_h_per_L > 0.0

    def test_cmax_positive(self):
        r = default_result()
        assert r.cmax_mg_L > 0.0

    def test_t_half_positive(self):
        r = default_result()
        assert r.t_half_h > 0.0

    def test_bioavailability_pct_between_0_and_100(self):
        r = default_result()
        assert 0.0 <= r.bioavailability_pct <= 100.0

    def test_first_pass_effect_pct_between_0_and_100(self):
        r = default_result()
        assert 0.0 <= r.first_pass_effect_pct <= 100.0


# ---------------------------------------------------------------------------
# Physiological relationships
# ---------------------------------------------------------------------------


class TestPhysiology:
    def test_higher_cl_hepatic_lower_bioavailability(self):
        r_low = default_result(cl_hepatic_L_per_h=1.0)
        r_high = default_result(cl_hepatic_L_per_h=60.0)
        assert r_high.bioavailability_pct < r_low.bioavailability_pct

    def test_higher_cl_hepatic_lower_auc(self):
        r_low = default_result(cl_hepatic_L_per_h=1.0)
        r_high = default_result(cl_hepatic_L_per_h=60.0)
        assert r_high.auc_mg_h_per_L < r_low.auc_mg_h_per_L

    def test_faster_ka_gives_higher_cmax(self):
        r_slow = default_result(ka_ip_per_h=0.1)
        r_fast = default_result(ka_ip_per_h=2.0)
        assert r_fast.cmax_mg_L > r_slow.cmax_mg_L

    def test_faster_ka_gives_earlier_tmax(self):
        r_slow = default_result(ka_ip_per_h=0.1)
        r_fast = default_result(ka_ip_per_h=2.0)
        assert r_fast.tmax_h < r_slow.tmax_h


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_cmax_scales_with_dose(self):
        r1 = default_result(dose_mg=100.0)
        r2 = default_result(dose_mg=200.0)
        ratio = r2.cmax_mg_L / r1.cmax_mg_L
        assert abs(ratio - 2.0) < 0.02

    def test_auc_scales_with_dose(self):
        r1 = default_result(dose_mg=100.0)
        r2 = default_result(dose_mg=200.0)
        ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
        assert abs(ratio - 2.0) < 0.02


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_intraperitoneal_pk("D", 0.0, 10.0, 5.0, 50.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_intraperitoneal_pk("D", -50.0, 10.0, 5.0, 50.0)

    def test_zero_cl_hepatic_raises(self):
        with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
            simulate_intraperitoneal_pk("D", 100.0, 0.0, 5.0, 50.0)

    def test_zero_cl_sys_raises(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            simulate_intraperitoneal_pk("D", 100.0, 10.0, 0.0, 50.0)

    def test_zero_vd_sys_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            simulate_intraperitoneal_pk("D", 100.0, 10.0, 5.0, 0.0)

    def test_zero_ka_ip_raises(self):
        with pytest.raises(ValueError, match="ka_ip_per_h"):
            simulate_intraperitoneal_pk("D", 100.0, 10.0, 5.0, 50.0, ka_ip_per_h=0.0)


# ---------------------------------------------------------------------------
# compare_ka_ip tests
# ---------------------------------------------------------------------------


class TestCompareKaIp:
    def test_returns_list(self):
        result = compare_ka_ip("Drug", 100.0, 10.0, 5.0, 50.0, [0.2, 0.5, 1.0])
        assert isinstance(result, list)

    def test_correct_length(self):
        result = compare_ka_ip("Drug", 100.0, 10.0, 5.0, 50.0, [0.2, 0.5, 1.0])
        assert len(result) == 3

    def test_sorted_descending_by_cmax(self):
        result = compare_ka_ip("Drug", 100.0, 10.0, 5.0, 50.0, [0.2, 0.5, 1.0, 2.0])
        for i in range(len(result) - 1):
            assert result[i].cmax_mg_L >= result[i + 1].cmax_mg_L

    def test_all_results_are_ip_type(self):
        results = compare_ka_ip("Drug", 100.0, 10.0, 5.0, 50.0, [0.5, 1.0])
        for r in results:
            assert isinstance(r, IntraperitonealPKResult)

    def test_fastest_ka_first(self):
        result = compare_ka_ip("Drug", 100.0, 10.0, 5.0, 50.0, [0.1, 0.5, 2.0])
        # Highest ka should give highest Cmax and appear first
        assert result[0].ka_ip_per_h == 2.0
