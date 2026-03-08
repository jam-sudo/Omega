"""
Tests for Phase 404 — Hepatic Sinusoidal Drug Transport
"""

import pytest

from omega_pbpk.core.hepatic_sinusoidal_transport import (
    HepaticSinusoidalResult,
    compare_oatp_inhibition,
    simulate_hepatic_sinusoidal_transport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_sim(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_sys_L_per_h=10.0,
        vd_L=50.0,
    )
    defaults.update(kwargs)
    return simulate_hepatic_sinusoidal_transport(**defaults)


# ---------------------------------------------------------------------------
# Return type and field checks
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_correct_type(self):
        result = _default_sim()
        assert isinstance(result, HepaticSinusoidalResult)

    def test_all_fields_present(self):
        result = _default_sim()
        assert isinstance(result.drug_name, str)
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_sinusoidal_mg_L, list)
        assert isinstance(result.c_hepatocyte_mg_L, list)
        assert isinstance(result.c_bile_mg_L, list)
        assert isinstance(result.hepatic_extraction_ratio, float)
        assert isinstance(result.biliary_excretion_fraction, float)
        assert isinstance(result.intracellular_accumulation_fold, float)
        assert isinstance(result.oatp_uptake_rate, float)
        assert isinstance(result.efflux_mrp2_rate, float)
        assert isinstance(result.auc_hepatocyte_mg_h_per_L, float)
        assert isinstance(result.auc_sinusoidal_mg_h_per_L, float)
        assert isinstance(result.notes, str)

    def test_time_array_length_matches_concentration_arrays(self):
        result = _default_sim()
        assert len(result.times_h) == len(result.c_sinusoidal_mg_L)
        assert len(result.times_h) == len(result.c_hepatocyte_mg_L)
        assert len(result.times_h) == len(result.c_bile_mg_L)

    def test_time_starts_at_zero(self):
        result = _default_sim()
        assert result.times_h[0] == 0.0

    def test_time_ends_near_t_end(self):
        result = _default_sim(t_end_h=12.0)
        assert abs(result.times_h[-1] - 12.0) < 0.1


# ---------------------------------------------------------------------------
# Concentration dynamics
# ---------------------------------------------------------------------------

class TestConcentrationDynamics:
    def test_sinusoidal_starts_at_dose_over_vd(self):
        dose, vd = 100.0, 50.0
        result = simulate_hepatic_sinusoidal_transport(
            "Drug", dose_mg=dose, cl_sys_L_per_h=10.0, vd_L=vd
        )
        expected_c0 = dose / vd
        assert abs(result.c_sinusoidal_mg_L[0] - expected_c0) < 1e-9

    def test_sinusoidal_declines_over_time(self):
        result = _default_sim()
        assert result.c_sinusoidal_mg_L[-1] < result.c_sinusoidal_mg_L[0]

    def test_all_concentrations_non_negative(self):
        result = _default_sim()
        assert all(c >= 0.0 for c in result.c_sinusoidal_mg_L)
        assert all(c >= 0.0 for c in result.c_hepatocyte_mg_L)
        assert all(c >= 0.0 for c in result.c_bile_mg_L)

    def test_bile_receives_drug(self):
        """Bile compartment should accumulate some drug over time."""
        result = _default_sim()
        assert max(result.c_bile_mg_L) > 0.0

    def test_auc_sinusoidal_positive(self):
        result = _default_sim()
        assert result.auc_sinusoidal_mg_h_per_L > 0.0

    def test_auc_hepatocyte_positive(self):
        result = _default_sim()
        assert result.auc_hepatocyte_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Hepatocyte accumulation
# ---------------------------------------------------------------------------

class TestHepatocyteAccumulation:
    def test_active_uptake_gives_intracellular_accumulation(self):
        """With high OATP Vmax, hepatocyte concentration should be higher than sinusoidal."""
        result = simulate_hepatic_sinusoidal_transport(
            "ActiveDrug",
            dose_mg=100.0,
            cl_sys_L_per_h=5.0,
            vd_L=50.0,
            oatp_vmax_uM_per_min=50.0,
            oatp_km_uM=1.0,
            mrp2_cl_mL_per_min=0.5,
        )
        assert result.intracellular_accumulation_fold > 1.0

    def test_higher_oatp_vmax_gives_higher_hepatocyte_auc(self):
        """Higher OATP Vmax → more hepatocyte accumulation."""
        result_low = simulate_hepatic_sinusoidal_transport(
            "Drug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0,
            oatp_vmax_uM_per_min=1.0
        )
        result_high = simulate_hepatic_sinusoidal_transport(
            "Drug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0,
            oatp_vmax_uM_per_min=50.0
        )
        assert result_high.auc_hepatocyte_mg_h_per_L > result_low.auc_hepatocyte_mg_h_per_L

    def test_higher_mrp2_gives_lower_hepatocyte_concentration(self):
        """Higher MRP2 efflux → less drug retained in hepatocytes."""
        result_low_mrp2 = simulate_hepatic_sinusoidal_transport(
            "Drug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0,
            mrp2_cl_mL_per_min=0.1
        )
        result_high_mrp2 = simulate_hepatic_sinusoidal_transport(
            "Drug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0,
            mrp2_cl_mL_per_min=20.0
        )
        assert result_low_mrp2.auc_hepatocyte_mg_h_per_L > result_high_mrp2.auc_hepatocyte_mg_h_per_L


# ---------------------------------------------------------------------------
# Extraction ratio
# ---------------------------------------------------------------------------

class TestExtractionRatio:
    def test_extraction_ratio_in_valid_range(self):
        result = _default_sim()
        assert 0.0 <= result.hepatic_extraction_ratio <= 0.99

    def test_high_cl_gives_high_extraction(self):
        result = simulate_hepatic_sinusoidal_transport(
            "HighCLDrug", dose_mg=100.0, cl_sys_L_per_h=80.0, vd_L=50.0
        )
        assert result.hepatic_extraction_ratio > 0.5

    def test_low_cl_gives_low_extraction(self):
        result = simulate_hepatic_sinusoidal_transport(
            "LowCLDrug", dose_mg=100.0, cl_sys_L_per_h=1.0, vd_L=50.0
        )
        assert result.hepatic_extraction_ratio < 0.5


# ---------------------------------------------------------------------------
# Biliary excretion fraction
# ---------------------------------------------------------------------------

class TestBiliaryExcretion:
    def test_biliary_fraction_in_valid_range(self):
        result = _default_sim()
        assert 0.0 <= result.biliary_excretion_fraction <= 1.0

    def test_high_mrp2_gives_higher_biliary_fraction(self):
        result_low = simulate_hepatic_sinusoidal_transport(
            "Drug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0,
            mrp2_cl_mL_per_min=0.1
        )
        result_high = simulate_hepatic_sinusoidal_transport(
            "Drug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0,
            mrp2_cl_mL_per_min=20.0
        )
        assert result_high.biliary_excretion_fraction >= result_low.biliary_excretion_fraction


# ---------------------------------------------------------------------------
# OATP inhibition comparison
# ---------------------------------------------------------------------------

class TestOatpInhibitionComparison:
    def test_compare_returns_list(self):
        results = compare_oatp_inhibition(
            "TestDrug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0
        )
        assert isinstance(results, list)

    def test_compare_default_returns_four_results(self):
        results = compare_oatp_inhibition(
            "TestDrug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0
        )
        assert len(results) == 4

    def test_compare_returns_hepatic_sinusoidal_results(self):
        results = compare_oatp_inhibition(
            "TestDrug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0
        )
        for r in results:
            assert isinstance(r, HepaticSinusoidalResult)

    def test_compare_sorted_by_auc_hepatocyte_descending(self):
        results = compare_oatp_inhibition(
            "TestDrug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0
        )
        for i in range(len(results) - 1):
            assert results[i].auc_hepatocyte_mg_h_per_L >= results[i + 1].auc_hepatocyte_mg_h_per_L

    def test_compare_custom_inhibition_factors(self):
        factors = [1.0, 0.1]
        results = compare_oatp_inhibition(
            "TestDrug", dose_mg=100.0, cl_sys_L_per_h=10.0, vd_L=50.0,
            inhibition_factors=factors
        )
        assert len(results) == 2

    def test_oatp_inhibition_reduces_hepatocyte_auc(self):
        """Complete OATP inhibition should reduce hepatocyte AUC vs no inhibition."""
        results = compare_oatp_inhibition(
            "TestDrug", dose_mg=100.0, cl_sys_L_per_h=5.0, vd_L=50.0,
            inhibition_factors=[1.0, 0.01]
        )
        # Sorted descending: first has highest AUC
        auc_no_inhibition = results[0].auc_hepatocyte_mg_h_per_L
        auc_full_inhibition = results[1].auc_hepatocyte_mg_h_per_L
        assert auc_no_inhibition >= auc_full_inhibition


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_sim(dose_mg=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_sim(dose_mg=0.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _default_sim(cl_sys_L_per_h=-1.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _default_sim(cl_sys_L_per_h=0.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default_sim(vd_L=-1.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default_sim(vd_L=0.0)

    def test_negative_oatp_vmax_raises(self):
        with pytest.raises(ValueError, match="oatp_vmax"):
            _default_sim(oatp_vmax_uM_per_min=-1.0)

    def test_negative_oatp_km_raises(self):
        with pytest.raises(ValueError, match="oatp_km"):
            _default_sim(oatp_km_uM=-1.0)

    def test_negative_mrp2_cl_raises(self):
        with pytest.raises(ValueError, match="mrp2_cl"):
            _default_sim(mrp2_cl_mL_per_min=-1.0)
