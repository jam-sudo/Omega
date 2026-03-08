"""Tests for Phase 691 — Hepatic Zonation."""

import pytest

from omega_pbpk.core.hepatic_zonation_p691 import (
    HepaticZonationResult,
    simulate_hepatic_zonation,
)

_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=400.0,
    cl_int_L_per_h=10.0,
    fu_plasma=0.5,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**_DEFAULTS, **overrides}
    return simulate_hepatic_zonation(**params)


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, HepaticZonationResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_all_lists_same_length(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_portal_mg_L) == n
        assert len(r.c_zone1_mg_L) == n
        assert len(r.c_zone3_mg_L) == n
        assert len(r.c_systemic_mg_L) == n


class TestConcentrations:
    def test_portal_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_portal_mg_L)

    def test_zone1_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_zone1_mg_L)

    def test_zone3_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_zone3_mg_L)

    def test_systemic_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_systemic_mg_L)


class TestExtractionRatio:
    def test_extraction_ratio_in_range(self):
        r = _run()
        assert 0 <= r.extraction_ratio <= 1

    def test_first_pass_loss_in_range(self):
        r = _run()
        assert 0 <= r.first_pass_loss_pct <= 100

    def test_first_pass_matches_extraction(self):
        r = _run()
        assert abs(r.first_pass_loss_pct - r.extraction_ratio * 100) < 0.01

    def test_higher_clint_higher_extraction(self):
        r_low = _run(cl_int_L_per_h=5.0)
        r_high = _run(cl_int_L_per_h=50.0)
        assert r_high.extraction_ratio > r_low.extraction_ratio


class TestZonation:
    def test_zonation_index_positive(self):
        r = _run()
        assert r.zonation_index > 0

    def test_zonation_index_default(self):
        r = _run()
        # Default fractions 0.7/0.3
        expected = 0.7 / 0.3
        assert abs(r.zonation_index - expected) < 0.01

    def test_fu_zone1_equals_fu_plasma(self):
        r = _run()
        assert r.fu_zone1 == 0.5

    def test_fu_zone3_equals_fu_plasma(self):
        r = _run()
        assert r.fu_zone3 == 0.5

    def test_clint_zone1_matches_fraction(self):
        r = _run()
        assert abs(r.clint_zone1_L_per_h - 10.0 * 0.3) < 0.01

    def test_clint_zone3_matches_fraction(self):
        r = _run()
        assert abs(r.clint_zone3_L_per_h - 10.0 * 0.7) < 0.01


class TestDoseScaling:
    def test_double_dose_doubles_systemic_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        cmax1 = max(r1.c_systemic_mg_L)
        cmax2 = max(r2.c_systemic_mg_L)
        ratio = cmax2 / cmax1 if cmax1 > 0 else 0
        assert abs(ratio - 2.0) < 0.15


class TestValidation:
    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-1)

    def test_zero_cl_int(self):
        with pytest.raises(ValueError, match="cl_int"):
            _run(cl_int_L_per_h=0)

    def test_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _run(fu_plasma=0)

    def test_fu_plasma_above_1(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _run(fu_plasma=1.5)

    def test_vd_zero(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=0)

    def test_cyp_fractions_bad_sum(self):
        with pytest.raises(ValueError, match="sum"):
            _run(cyp_zone1_fraction=0.5, cyp_zone3_fraction=0.3)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
