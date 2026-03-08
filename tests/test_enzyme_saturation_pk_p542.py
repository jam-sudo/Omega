"""Tests for Phase 542: Drug Metabolizing Enzyme Saturation PK (enzyme_saturation_pk_p542)."""

import pytest

from omega_pbpk.core.enzyme_saturation_pk_p542 import (
    EnzymeSaturationResult,
    dose_proportionality_test,
    simulate_enzyme_saturation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_sim(**kwargs):
    defaults = dict(
        drug_name="DrugX",
        dose_mg=100.0,
        vmax_mg_per_h=10.0,
        km_mg_per_L=1.0,
        vd_L=10.0,
    )
    defaults.update(kwargs)
    return simulate_enzyme_saturation(**defaults)


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_enzyme_saturation_result(self):
        assert isinstance(_base_sim(), EnzymeSaturationResult)

    def test_all_fields_present(self):
        result = _base_sim()
        for attr in [
            "drug_name",
            "dose_mg",
            "vmax_mg_per_h",
            "km_mg_per_L",
            "vd_L",
            "cmax_mg_L",
            "times_h",
            "c_plasma_mg_L",
            "auc_mg_h_per_L",
            "t_half_h",
            "t_km_h",
            "saturation_pct_at_t0",
            "linearity_class",
            "notes",
        ]:
            assert hasattr(result, attr), f"Missing field: {attr}"

    def test_times_h_is_list(self):
        result = _base_sim()
        assert isinstance(result.times_h, list)

    def test_c_plasma_is_list(self):
        result = _base_sim()
        assert isinstance(result.c_plasma_mg_L, list)

    def test_drug_name_stored(self):
        result = _base_sim(drug_name="Ibuprofen")
        assert result.drug_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# Cmax and initial conditions
# ---------------------------------------------------------------------------
class TestCmax:
    def test_cmax_equals_dose_over_vd(self):
        result = _base_sim(dose_mg=50.0, vd_L=10.0)
        assert abs(result.cmax_mg_L - 50.0 / 10.0) < 1e-6

    def test_cmax_varies_with_dose(self):
        r1 = _base_sim(dose_mg=50.0)
        r2 = _base_sim(dose_mg=200.0)
        assert r2.cmax_mg_L > r1.cmax_mg_L

    def test_cmax_varies_with_vd(self):
        r1 = _base_sim(vd_L=5.0)
        r2 = _base_sim(vd_L=20.0)
        assert r1.cmax_mg_L > r2.cmax_mg_L

    def test_initial_concentration_equals_cmax_iv(self):
        result = _base_sim(dose_mg=100.0, vd_L=10.0)
        assert abs(result.c_plasma_mg_L[0] - result.cmax_mg_L) < 1e-6


# ---------------------------------------------------------------------------
# Concentration-time profile
# ---------------------------------------------------------------------------
class TestConcentrationProfile:
    def test_c_declines_over_time(self):
        result = _base_sim()
        # c[0] should be max; overall trend should be declining
        first = result.c_plasma_mg_L[0]
        last = result.c_plasma_mg_L[-1]
        assert first > last

    def test_c_nonnegative(self):
        result = _base_sim()
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)

    def test_times_start_at_zero(self):
        result = _base_sim()
        assert result.times_h[0] == 0.0

    def test_times_length_matches_concs(self):
        result = _base_sim()
        assert len(result.times_h) == len(result.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# Saturation metrics
# ---------------------------------------------------------------------------
class TestSaturationMetrics:
    def test_saturation_pct_formula(self):
        result = _base_sim(dose_mg=100.0, vd_L=10.0, km_mg_per_L=1.0)
        c0 = 100.0 / 10.0  # 10 mg/L
        expected_pct = c0 / (c0 + 1.0) * 100.0  # ~90.9%
        assert abs(result.saturation_pct_at_t0 - expected_pct) < 0.1

    def test_saturation_pct_range(self):
        result = _base_sim()
        assert 0.0 <= result.saturation_pct_at_t0 <= 100.0

    def test_high_cmax_high_saturation(self):
        result = _base_sim(dose_mg=1000.0, km_mg_per_L=1.0, vd_L=10.0)
        assert result.saturation_pct_at_t0 > 90.0

    def test_low_cmax_low_saturation(self):
        # Cmax=0.01, Km=100 → very low saturation
        result = _base_sim(dose_mg=0.1, km_mg_per_L=100.0, vd_L=10.0)
        assert result.saturation_pct_at_t0 < 1.0


# ---------------------------------------------------------------------------
# Linearity classification
# ---------------------------------------------------------------------------
class TestLinearityClass:
    def test_linear_class_low_cmax_km_ratio(self):
        # Cmax/Km < 1: dose=1, Vd=10 → Cmax=0.1, Km=1 → ratio=0.1 < 1
        result = _base_sim(dose_mg=1.0, km_mg_per_L=1.0, vd_L=10.0)
        assert result.linearity_class == "linear"

    def test_nonlinear_class_high_cmax_km_ratio(self):
        # Cmax/Km > 5: dose=100, Vd=10 → Cmax=10, Km=1 → ratio=10 > 5
        result = _base_sim(dose_mg=100.0, km_mg_per_L=1.0, vd_L=10.0)
        assert result.linearity_class == "nonlinear"

    def test_transition_class_mid_ratio(self):
        # Ratio ~2: dose=20, Vd=10 → Cmax=2, Km=1 → ratio=2
        result = _base_sim(dose_mg=20.0, km_mg_per_L=1.0, vd_L=10.0)
        assert result.linearity_class == "transition"

    def test_linearity_class_valid_string(self):
        result = _base_sim()
        assert result.linearity_class in {"linear", "nonlinear", "transition"}


# ---------------------------------------------------------------------------
# t_km_h
# ---------------------------------------------------------------------------
class TestTKm:
    def test_t_km_positive_when_saturated(self):
        # Cmax=10 >> Km=1, so there is a crossing time
        result = _base_sim(dose_mg=100.0, km_mg_per_L=1.0, vd_L=10.0, t_end_h=48.0)
        assert result.t_km_h > 0.0

    def test_t_km_zero_when_never_saturated(self):
        # Cmax < Km: dose=0.5, Vd=10 → Cmax=0.05, Km=1 → never saturated
        result = _base_sim(dose_mg=0.5, km_mg_per_L=1.0, vd_L=10.0)
        assert result.t_km_h == 0.0

    def test_t_km_within_simulation_time(self):
        result = _base_sim(dose_mg=100.0, km_mg_per_L=1.0, vd_L=10.0, t_end_h=48.0)
        assert result.t_km_h <= 48.0


# ---------------------------------------------------------------------------
# AUC and t_half
# ---------------------------------------------------------------------------
class TestAUCAndHalfLife:
    def test_auc_positive(self):
        result = _base_sim()
        assert result.auc_mg_h_per_L > 0.0

    def test_t_half_positive(self):
        result = _base_sim()
        assert result.t_half_h > 0.0

    def test_t_half_formula(self):
        # t_half = 0.693 * Vd / (Vmax/Km) = 0.693 * 10 / (10/1) = 0.693
        result = _base_sim(vmax_mg_per_h=10.0, km_mg_per_L=1.0, vd_L=10.0)
        expected = 0.693 * 10.0 / (10.0 / 1.0)
        assert abs(result.t_half_h - expected) < 1e-6


# ---------------------------------------------------------------------------
# Dose proportionality test
# ---------------------------------------------------------------------------
class TestDoseProportionality:
    def test_returns_list(self):
        result = dose_proportionality_test("Drug", [10.0, 50.0, 100.0], 5.0, 1.0, 10.0)
        assert isinstance(result, list)

    def test_sorted_by_dose_ascending(self):
        result = dose_proportionality_test("Drug", [100.0, 10.0, 50.0], 5.0, 1.0, 10.0)
        doses = [r["dose_mg"] for r in result]
        assert doses == sorted(doses)

    def test_dict_keys(self):
        result = dose_proportionality_test("Drug", [10.0, 100.0], 5.0, 1.0, 10.0)
        for r in result:
            assert "dose_mg" in r
            assert "auc_mg_h_per_L" in r

    def test_superproportional_auc_at_saturation(self):
        """At high doses (saturation), doubling dose → >2x AUC."""
        # Low Vmax/high dose → saturation
        result = dose_proportionality_test("Drug", [50.0, 100.0], 2.0, 1.0, 10.0)
        auc_low = result[0]["auc_mg_h_per_L"]
        auc_high = result[1]["auc_mg_h_per_L"]
        # At saturation, 2x dose should give more than 2x AUC
        assert auc_high > 2.0 * auc_low

    def test_auc_increases_with_dose(self):
        result = dose_proportionality_test("Drug", [10.0, 50.0, 200.0], 5.0, 1.0, 10.0)
        aucs = [r["auc_mg_h_per_L"] for r in result]
        assert aucs[0] < aucs[1] < aucs[2]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_enzyme_saturation("D", 0.0, 10.0, 1.0, 10.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_enzyme_saturation("D", -5.0, 10.0, 1.0, 10.0)

    def test_zero_vmax_raises(self):
        with pytest.raises(ValueError):
            simulate_enzyme_saturation("D", 100.0, 0.0, 1.0, 10.0)

    def test_zero_km_raises(self):
        with pytest.raises(ValueError):
            simulate_enzyme_saturation("D", 100.0, 10.0, 0.0, 10.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            simulate_enzyme_saturation("D", 100.0, 10.0, 1.0, 0.0)

    def test_negative_vmax_raises(self):
        with pytest.raises(ValueError):
            simulate_enzyme_saturation("D", 100.0, -5.0, 1.0, 10.0)
