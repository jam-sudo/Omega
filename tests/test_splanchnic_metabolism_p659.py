"""Tests for splanchnic first-pass metabolism model (phase 659)."""

import pytest

from omega_pbpk.core.splanchnic_metabolism_p659 import (
    SplanchnicMetabolismResult,
    simulate_splanchnic_metabolism,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.5,
    cl_gut_L_per_h=1.0,
    cl_hepatic_L_per_h=10.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_splanchnic_metabolism(**params)


# ---------------------------------------------------------------------------
# Basic return-type & structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, SplanchnicMetabolismResult)

    def test_drug_name_stored(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_stored(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)

    def test_list_lengths_match(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_enterocyte_mg_g) == n
        assert len(r.c_portal_mg_L) == n
        assert len(r.c_hepatic_vein_mg_L) == n
        assert len(r.c_systemic_mg_L) == n


# ---------------------------------------------------------------------------
# Non-negativity
# ---------------------------------------------------------------------------


class TestNonNegativity:
    def test_c_enterocyte_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_enterocyte_mg_g)

    def test_c_portal_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_portal_mg_L)

    def test_c_hepatic_vein_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_hepatic_vein_mg_L)

    def test_c_systemic_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_systemic_mg_L)

    def test_auc_nonneg(self):
        r = _run()
        assert r.auc_systemic_mg_h_per_L >= 0


# ---------------------------------------------------------------------------
# Extraction & bioavailability
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_gut_wall_extraction_in_range(self):
        r = _run()
        assert 0.0 <= r.gut_wall_extraction <= 1.0

    def test_hepatic_extraction_in_range(self):
        r = _run()
        assert 0.0 <= r.hepatic_extraction <= 1.0

    def test_fg_equals_one_minus_gut_extraction(self):
        r = _run()
        assert abs(r.fg_gut_wall - (1.0 - r.gut_wall_extraction)) < 1e-10

    def test_fh_equals_one_minus_hepatic_extraction(self):
        r = _run()
        assert abs(r.fh_hepatic - (1.0 - r.hepatic_extraction)) < 1e-10

    def test_f_total_equals_fg_times_fh(self):
        r = _run()
        assert abs(r.f_total - r.fg_gut_wall * r.fh_hepatic) < 1e-10

    def test_high_cl_gut_lowers_fg(self):
        r_low = _run(cl_gut_L_per_h=0.5)
        r_high = _run(cl_gut_L_per_h=5.0)
        assert r_high.fg_gut_wall < r_low.fg_gut_wall

    def test_high_cl_hepatic_lowers_fh(self):
        r_low = _run(cl_hepatic_L_per_h=5.0)
        r_high = _run(cl_hepatic_L_per_h=50.0)
        assert r_high.fh_hepatic < r_low.fh_hepatic


# ---------------------------------------------------------------------------
# CYP3A4 activity effects
# ---------------------------------------------------------------------------


class TestCYP3A4:
    def test_reduced_gut_cyp3a4_increases_fg(self):
        r_full = _run(cyp3a4_gut_activity=1.0)
        r_half = _run(cyp3a4_gut_activity=0.5)
        assert r_half.fg_gut_wall > r_full.fg_gut_wall

    def test_reduced_hepatic_cyp3a4_increases_fh(self):
        r_full = _run(cyp3a4_hepatic_activity=1.0)
        r_half = _run(cyp3a4_hepatic_activity=0.5)
        assert r_half.fh_hepatic > r_full.fh_hepatic

    def test_reduced_cyp3a4_increases_f_total(self):
        r_full = _run(cyp3a4_gut_activity=1.0, cyp3a4_hepatic_activity=1.0)
        r_half = _run(cyp3a4_gut_activity=0.5, cyp3a4_hepatic_activity=0.5)
        assert r_half.f_total > r_full.f_total


# ---------------------------------------------------------------------------
# Systemic PK
# ---------------------------------------------------------------------------


class TestSystemicPK:
    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_systemic_mg_L > 0

    def test_auc_proportional_to_dose(self):
        r1 = _run(dose_mg=50.0)
        r2 = _run(dose_mg=100.0)
        ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
        assert abs(ratio - 2.0) < 0.3  # approximately dose-proportional

    def test_gut_to_portal_ratio_positive(self):
        r = _run()
        assert r.gut_to_portal_ratio_auc > 0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_cmax_scales_with_dose(self):
        r1 = _run(dose_mg=50.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
        assert 3.0 < ratio < 5.0  # approximately 4x


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_negative_cl_gut_raises(self):
        with pytest.raises(ValueError, match="cl_gut"):
            _run(cl_gut_L_per_h=-1.0)

    def test_negative_cl_hepatic_raises(self):
        with pytest.raises(ValueError, match="cl_hepatic"):
            _run(cl_hepatic_L_per_h=-1.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=-10.0)

    def test_negative_cyp3a4_gut_raises(self):
        with pytest.raises(ValueError, match="cyp3a4_gut"):
            _run(cyp3a4_gut_activity=-0.1)

    def test_negative_cyp3a4_hepatic_raises(self):
        with pytest.raises(ValueError, match="cyp3a4_hepatic"):
            _run(cyp3a4_hepatic_activity=-0.1)
