"""Tests for Phase 591 — Dose-Dependent Nonlinear PK."""

import pytest

from omega_pbpk.core.nonlinear_pk_p591 import (
    NonlinearPKResult,
    dose_proportionality_analysis,
    simulate_nonlinear_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        vmax_mg_h=20.0,
        km_mg_L=10.0,
        vd_L=50.0,
    )
    defaults.update(kwargs)
    return simulate_nonlinear_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_nonlinear_pk_result(self):
        r = _default()
        assert isinstance(r, NonlinearPKResult)

    def test_drug_name_stored(self):
        r = _default(drug_name="Phenytoin")
        assert r.drug_name == "Phenytoin"

    def test_dose_mg_stored(self):
        r = _default(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_vmax_stored(self):
        r = _default(vmax_mg_h=30.0)
        assert r.vmax_mg_h == 30.0

    def test_km_stored(self):
        r = _default(km_mg_L=5.0)
        assert r.km_mg_L == 5.0

    def test_vd_stored(self):
        r = _default(vd_L=40.0)
        assert r.vd_L == 40.0

    def test_times_is_list(self):
        r = _default()
        assert isinstance(r.times_h, list)

    def test_conc_is_list(self):
        r = _default()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_times_conc_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.c_plasma_mg_L)

    def test_auc_positive(self):
        r = _default()
        assert r.auc_mg_h_per_L > 0

    def test_cmax_positive(self):
        r = _default()
        assert r.cmax_mg_L > 0

    def test_tmax_nonneg(self):
        r = _default()
        assert r.tmax_h >= 0.0

    def test_t_half_positive(self):
        r = _default()
        assert r.t_half_effective_h >= 0.0

    def test_dose_prop_index_positive(self):
        r = _default()
        assert r.dose_proportionality_index > 0

    def test_nonlinearity_class_is_string(self):
        r = _default()
        assert r.nonlinearity_class in ("linear", "weakly_nonlinear", "strongly_nonlinear")

    def test_notes_is_string(self):
        r = _default()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# IV vs oral
# ---------------------------------------------------------------------------


class TestIVvsOral:
    def test_iv_cmax_equals_dose_over_vd(self):
        """IV: C(0) = dose / Vd."""
        r = _default(route="iv", dose_mg=100.0, vd_L=50.0)
        assert abs(r.c_plasma_mg_L[0] - 100.0 / 50.0) < 1e-6

    def test_iv_tmax_is_zero(self):
        r = _default(route="iv")
        assert r.tmax_h == 0.0

    def test_oral_tmax_positive(self):
        r = _default(route="oral", ka_per_h=1.0)
        assert r.tmax_h > 0.0

    def test_oral_initial_conc_zero(self):
        r = _default(route="oral")
        assert r.c_plasma_mg_L[0] == 0.0

    def test_oral_cmax_less_than_iv_cmax(self):
        """Oral Cmax < IV C(0) due to absorption rate and bioavailability."""
        r_iv = _default(route="iv", dose_mg=100.0)
        r_oral = _default(route="oral", dose_mg=100.0, f_oral=0.9)
        assert r_oral.cmax_mg_L < r_iv.c_plasma_mg_L[0]

    def test_linear_clearance_reduces_auc(self):
        r_no_lin = _default(cl_linear_L_per_h=0.0)
        r_with_lin = _default(cl_linear_L_per_h=5.0)
        assert r_with_lin.auc_mg_h_per_L < r_no_lin.auc_mg_h_per_L

    def test_conc_nonneg(self):
        r = _default()
        assert all(c >= 0.0 for c in r.c_plasma_mg_L)

    def test_times_start_at_zero(self):
        r = _default()
        assert r.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# Nonlinearity detection
# ---------------------------------------------------------------------------


class TestNonlinearityDetection:
    def test_high_dose_strongly_nonlinear(self):
        """Dose straddling Km → nonlinear (2x dose gives >1.1x AUC ratio).

        dose=50mg, Vd=10L → C0=5mg/L ≈ Km=5mg/L (near saturation).
        2x dose has C0=10 >> Km, so AUC grows super-proportionally.
        """
        r = simulate_nonlinear_pk(
            drug_name="HighDose",
            dose_mg=50.0,
            vmax_mg_h=20.0,
            km_mg_L=5.0,
            vd_L=10.0,
            t_end_h=24.0,
        )
        assert r.nonlinearity_class in ("weakly_nonlinear", "strongly_nonlinear")

    def test_low_dose_linear(self):
        """Very low dose relative to Km → linear kinetics."""
        r = simulate_nonlinear_pk(
            drug_name="LowDose",
            dose_mg=0.1,  # C << Km → linear approximation
            vmax_mg_h=100.0,
            km_mg_L=100.0,
            vd_L=10.0,
            t_end_h=24.0,
        )
        assert r.nonlinearity_class == "linear"

    def test_nonlinearity_class_values(self):
        r = _default()
        assert r.nonlinearity_class in ("linear", "weakly_nonlinear", "strongly_nonlinear")

    def test_dose_prop_index_equals_auc_over_dose(self):
        r = _default(dose_mg=50.0)
        expected = r.auc_mg_h_per_L / 50.0
        assert abs(r.dose_proportionality_index - expected) < 1e-9

    def test_higher_dose_higher_auc(self):
        r_low = _default(dose_mg=50.0)
        r_high = _default(dose_mg=200.0)
        assert r_high.auc_mg_h_per_L > r_low.auc_mg_h_per_L

    def test_higher_km_more_linear(self):
        """High Km → drug always in linear range → closer to linear class."""
        r = simulate_nonlinear_pk(
            drug_name="HighKm",
            dose_mg=10.0,
            vmax_mg_h=50.0,
            km_mg_L=1000.0,  # C << Km always
            vd_L=10.0,
        )
        assert r.nonlinearity_class == "linear"


# ---------------------------------------------------------------------------
# Dose proportionality analysis
# ---------------------------------------------------------------------------


class TestDoseProportionalityAnalysis:
    def test_returns_list(self):
        results = dose_proportionality_analysis(
            drug_name="Drug",
            doses_mg=[50.0, 100.0, 200.0],
            vmax_mg_h=20.0,
            km_mg_L=10.0,
            vd_L=50.0,
        )
        assert isinstance(results, list)

    def test_length_matches_doses(self):
        doses = [10.0, 50.0, 100.0, 200.0]
        results = dose_proportionality_analysis(
            drug_name="Drug",
            doses_mg=doses,
            vmax_mg_h=20.0,
            km_mg_L=10.0,
            vd_L=50.0,
        )
        assert len(results) == len(doses)

    def test_sorted_by_dose(self):
        doses = [200.0, 50.0, 100.0]
        results = dose_proportionality_analysis(
            drug_name="Drug",
            doses_mg=doses,
            vmax_mg_h=20.0,
            km_mg_L=10.0,
            vd_L=50.0,
        )
        sorted_doses = [r.dose_mg for r in results]
        assert sorted_doses == sorted(doses)

    def test_each_result_is_nonlinear_pk_result(self):
        results = dose_proportionality_analysis(
            drug_name="Drug",
            doses_mg=[100.0, 200.0],
            vmax_mg_h=20.0,
            km_mg_L=10.0,
            vd_L=50.0,
        )
        for r in results:
            assert isinstance(r, NonlinearPKResult)

    def test_auc_increases_with_dose(self):
        results = dose_proportionality_analysis(
            drug_name="Drug",
            doses_mg=[50.0, 100.0, 200.0],
            vmax_mg_h=20.0,
            km_mg_L=10.0,
            vd_L=50.0,
        )
        aucs = [r.auc_mg_h_per_L for r in results]
        assert aucs[0] < aucs[1] < aucs[2]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_zero_vmax_raises(self):
        with pytest.raises(ValueError, match="vmax_mg_h"):
            _default(vmax_mg_h=0.0)

    def test_zero_km_raises(self):
        with pytest.raises(ValueError, match="km_mg_L"):
            _default(km_mg_L=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default(vd_L=0.0)

    def test_negative_cl_linear_raises(self):
        with pytest.raises(ValueError, match="cl_linear"):
            _default(cl_linear_L_per_h=-1.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            _default(route="im")

    def test_f_oral_zero_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _default(f_oral=0.0, route="oral")

    def test_f_oral_above_one_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _default(f_oral=1.5, route="oral")
