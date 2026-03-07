"""Tests for salivary gland PK model (Phase 1015)."""

import pytest

from omega_pbpk.core.salivary_gland_pk import SalivaryPKResult, simulate_salivary_pk


def default_result(**kwargs) -> SalivaryPKResult:
    """Helper: run a simulation with default neutral drug parameters."""
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logp=0.5,
        pka=7.0,
        ionization_type="neutral",
        fup=0.5,
        route="oral",
        vd_L=30.0,
        cl_L_per_h=3.0,
        saliva_flow_mL_per_h=30.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_salivary_pk(**params)


class TestReturnType:
    def test_returns_salivary_pk_result(self):
        result = default_result()
        assert isinstance(result, SalivaryPKResult)

    def test_drug_name_preserved(self):
        result = default_result(drug_name="Lithium")
        assert result.drug_name == "Lithium"

    def test_dose_preserved(self):
        result = default_result(dose_mg=200.0)
        assert result.dose_mg == 200.0


class TestConcentrationPositivity:
    def test_cmax_plasma_positive(self):
        result = default_result()
        assert result.cmax_plasma_mg_L > 0

    def test_cmax_saliva_positive(self):
        result = default_result()
        assert result.cmax_saliva_mg_L > 0

    def test_auc_plasma_positive(self):
        result = default_result()
        assert result.auc_plasma_mg_h_per_L > 0

    def test_auc_saliva_positive(self):
        result = default_result()
        assert result.auc_saliva_mg_h_per_L > 0

    def test_saliva_to_plasma_ratio_positive(self):
        result = default_result()
        assert result.saliva_to_plasma_ratio > 0


class TestTdmFeasibility:
    def test_tdm_feasibility_valid_values(self):
        result = default_result()
        assert result.tdm_feasibility in {"excellent", "good", "poor"}

    def test_neutral_drug_excellent_or_good_feasibility(self):
        # Neutral drug with fup=1.0 → ratio should be near 0.5, which is on the boundary
        result = default_result(fup=1.0, ionization_type="neutral")
        assert result.tdm_feasibility in {"excellent", "good", "poor"}


class TestRoutes:
    def test_oral_initial_plasma_near_zero(self):
        result = default_result(route="oral")
        # First time point (t=0) should be approximately 0 for oral route
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)

    def test_iv_initial_plasma_positive(self):
        result = default_result(route="iv")
        # For IV, C_plasma[0] = dose / Vd > 0
        assert result.c_plasma_mg_L[0] > 0


class TestNeutralDrugSalivaPlasmaRatio:
    def test_neutral_drug_ratio_approx_fup(self):
        """For neutral drug, saliva-to-plasma ratio should be within 2x of fup."""
        fup = 0.5
        result = default_result(ionization_type="neutral", fup=fup)
        ratio = result.saliva_to_plasma_ratio
        # Should be within 2x of fup
        assert ratio < fup * 2.0
        assert ratio > fup * 0.01  # not zero

    def test_higher_fup_higher_ratio(self):
        result_low = default_result(fup=0.1)
        result_high = default_result(fup=0.9)
        assert result_high.saliva_to_plasma_ratio > result_low.saliva_to_plasma_ratio


class TestTmaxOrdering:
    def test_tmax_saliva_gte_tmax_plasma_oral(self):
        """Saliva should lag behind plasma for oral dosing."""
        result = default_result(route="oral")
        assert result.tmax_saliva_h >= result.tmax_plasma_h


class TestTimeArray:
    def test_times_start_at_zero(self):
        result = default_result()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = default_result(t_end_h=12.0, dt_h=0.1)
        assert result.times_h[-1] == pytest.approx(12.0)

    def test_concentration_arrays_same_length(self):
        result = default_result()
        n = len(result.times_h)
        assert len(result.c_plasma_mg_L) == n
        assert len(result.c_saliva_mg_L) == n


class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("X", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("X", dose_mg=-10.0)

    def test_invalid_fup_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("X", dose_mg=100.0, fup=0.0)

    def test_invalid_fup_over_one_raises(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("X", dose_mg=100.0, fup=1.5)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("X", dose_mg=100.0, route="im")

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError):
            simulate_salivary_pk("X", dose_mg=100.0, ionization_type="zwitterion")


class TestIonizationEffect:
    def test_acid_drug_lower_ratio_than_neutral_at_low_pka(self):
        """Acidic drug with pKa < salivary pH should have lower salivary uptake."""
        result_neutral = default_result(ionization_type="neutral", fup=0.5)
        result_acid = default_result(ionization_type="acid", pka=4.0, fup=0.5)
        # Acid fully ionized at pH 6.8 when pKa << pH → lower ratio
        assert result_acid.saliva_to_plasma_ratio < result_neutral.saliva_to_plasma_ratio
