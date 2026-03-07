"""Tests for bile acid-drug interaction module (Phase 311)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.bile_acid_interaction import (
    BileAbsorptionResult,
    BileTransporterDDIResult,
    CholesBoundResult,
    bile_acid_transporter_ddi,
    cholestyramine_binding,
    micellar_solubilization_capacity,
    simulate_bile_acid_effect_on_absorption,
)

# ---------------------------------------------------------------------------
# micellar_solubilization_capacity
# ---------------------------------------------------------------------------


class TestMicellarSolubilizationCapacity:
    def test_zero_bile_returns_one(self):
        """No bile acids → enhancement factor = 1.0 (no enhancement)."""
        result = micellar_solubilization_capacity(0.0, logP=2.0, mw=300.0)
        assert result == pytest.approx(1.0)

    def test_positive_bile_increases_enhancement(self):
        """Higher bile acid concentration → higher enhancement."""
        low = micellar_solubilization_capacity(1.0, logP=2.0, mw=300.0)
        high = micellar_solubilization_capacity(5.0, logP=2.0, mw=300.0)
        assert high > low > 1.0

    def test_higher_logP_increases_enhancement(self):
        """More lipophilic drug → stronger micellar partitioning."""
        hydrophilic = micellar_solubilization_capacity(5.0, logP=0.0, mw=300.0)
        lipophilic = micellar_solubilization_capacity(5.0, logP=4.0, mw=300.0)
        assert lipophilic > hydrophilic

    def test_negative_bile_raises(self):
        with pytest.raises(ValueError, match="c_bile_mM must be >= 0"):
            micellar_solubilization_capacity(-1.0, logP=2.0, mw=300.0)

    def test_fed_state_enhancement(self):
        """Fed state (5 mM) should give > 1.0 enhancement for logP=3 drug."""
        result = micellar_solubilization_capacity(5.0, logP=3.0, mw=350.0)
        assert result > 1.0

    def test_fasted_state_lower_than_fed(self):
        """Fasted state (1 mM) < fed state (5 mM) enhancement."""
        fasted = micellar_solubilization_capacity(1.0, logP=3.0, mw=350.0)
        fed = micellar_solubilization_capacity(5.0, logP=3.0, mw=350.0)
        assert fed > fasted

    def test_known_value(self):
        """K_bile = 0.1 * exp(0.5 * 2) = 0.1 * e = ~0.2718; c=5mM=0.005 mol/L.
        enhancement = 1 + 0.2718 * 0.005 = 1.001359."""
        k = 0.1 * math.exp(0.5 * 2.0)
        expected = 1.0 + k * 0.005
        result = micellar_solubilization_capacity(5.0, logP=2.0, mw=300.0)
        assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# cholestyramine_binding
# ---------------------------------------------------------------------------


class TestCholestyramineBinding:
    def test_basic_drug_high_binding(self):
        """Basic drug (pKa=9) is mostly ionized at pH 6.5 → high binding."""
        result = cholestyramine_binding(100.0, drug_pka=9.0, drug_logP=2.0)
        assert isinstance(result, CholesBoundResult)
        assert result.fraction_bound > 0.5

    def test_acidic_drug_low_binding(self):
        """Acidic drug (pKa=3) is not ionized at pH 6.5 → low binding."""
        result = cholestyramine_binding(100.0, drug_pka=3.0, drug_logP=2.0)
        assert result.fraction_bound < 0.1

    def test_dose_conservation(self):
        """dose_available + bound = original dose."""
        dose = 100.0
        result = cholestyramine_binding(dose, drug_pka=9.0, drug_logP=2.0)
        total = result.dose_available_mg + result.fraction_bound * dose
        assert total == pytest.approx(dose, rel=1e-4)

    def test_fraction_bound_range(self):
        """Fraction bound must be between 0 and 1."""
        result = cholestyramine_binding(100.0, drug_pka=7.4, drug_logP=2.0)
        assert 0.0 <= result.fraction_bound <= 1.0

    def test_zero_cholestyramine_no_binding(self):
        """Zero resin dose → no binding."""
        result = cholestyramine_binding(
            100.0, drug_pka=9.0, drug_logP=2.0, cholestyramine_dose_g=0.0
        )
        assert result.fraction_bound == pytest.approx(0.0)
        assert result.dose_available_mg == pytest.approx(100.0)

    def test_very_large_dose_no_binding(self):
        """Very large drug dose → fraction bound approaches 0."""
        result = cholestyramine_binding(
            100000.0, drug_pka=9.0, drug_logP=2.0, cholestyramine_dose_g=4.0
        )
        assert result.fraction_bound < 0.01

    def test_invalid_drug_dose(self):
        with pytest.raises(ValueError, match="drug_dose_mg must be > 0"):
            cholestyramine_binding(0.0, drug_pka=9.0, drug_logP=2.0)

    def test_invalid_negative_cholestyramine(self):
        with pytest.raises(ValueError, match="cholestyramine_dose_g must be >= 0"):
            cholestyramine_binding(100.0, drug_pka=9.0, drug_logP=2.0, cholestyramine_dose_g=-1.0)

    def test_result_is_frozen(self):
        result = cholestyramine_binding(100.0, drug_pka=9.0, drug_logP=2.0)
        with pytest.raises(Exception):
            result.fraction_bound = 0.5  # type: ignore[misc]

    def test_notes_string(self):
        result = cholestyramine_binding(100.0, drug_pka=9.0, drug_logP=2.0)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# bile_acid_transporter_ddi
# ---------------------------------------------------------------------------


class TestBileAcidTransporterDDI:
    def test_oatp_substrate_normal_baseline(self):
        """At normal bile acid level, some OATP inhibition expected for substrate."""
        result = bile_acid_transporter_ddi(True, False, "normal")
        assert isinstance(result, BileTransporterDDIResult)
        assert result.oatp_inhibition_pct > 0.0

    def test_cholestasis_increases_oatp_inhibition(self):
        """Cholestasis bile acid level → higher OATP inhibition than normal."""
        normal = bile_acid_transporter_ddi(True, False, "normal")
        cholestasis = bile_acid_transporter_ddi(True, False, "cholestasis")
        assert cholestasis.oatp_inhibition_pct > normal.oatp_inhibition_pct

    def test_non_oatp_substrate_no_inhibition(self):
        """Non-OATP substrate → zero OATP inhibition."""
        result = bile_acid_transporter_ddi(False, False, "cholestasis")
        assert result.oatp_inhibition_pct == 0.0

    def test_asbt_substrate_normal_no_reduction(self):
        """ASBT substrate at normal bile acid level → no absorption reduction."""
        result = bile_acid_transporter_ddi(False, True, "normal")
        assert result.absorption_reduction_pct == 0.0

    def test_asbt_substrate_cholestasis_reduction(self):
        """ASBT substrate at cholestasis → positive absorption reduction."""
        result = bile_acid_transporter_ddi(False, True, "cholestasis")
        assert result.absorption_reduction_pct > 0.0

    def test_after_meal_intermediate(self):
        """After-meal effect is between normal and cholestasis."""
        normal = bile_acid_transporter_ddi(True, True, "normal")
        meal = bile_acid_transporter_ddi(True, True, "after_meal")
        chol = bile_acid_transporter_ddi(True, True, "cholestasis")
        assert normal.oatp_inhibition_pct <= meal.oatp_inhibition_pct <= chol.oatp_inhibition_pct

    def test_invalid_bile_level(self):
        with pytest.raises(ValueError, match="bile_acid_level must be one of"):
            bile_acid_transporter_ddi(True, False, "low")

    def test_result_frozen(self):
        result = bile_acid_transporter_ddi(True, False, "normal")
        with pytest.raises(Exception):
            result.oatp_inhibition_pct = 99.0  # type: ignore[misc]

    def test_both_substrates(self):
        result = bile_acid_transporter_ddi(True, True, "cholestasis")
        assert result.oatp_inhibition_pct > 0.0
        assert result.absorption_reduction_pct > 0.0

    def test_absorption_reduction_max_40pct(self):
        """Absorption reduction capped at 40%."""
        result = bile_acid_transporter_ddi(False, True, "cholestasis")
        assert result.absorption_reduction_pct <= 40.0


# ---------------------------------------------------------------------------
# simulate_bile_acid_effect_on_absorption
# ---------------------------------------------------------------------------


class TestSimulateBileAcidEffectOnAbsorption:
    def _default_params(self, meal_state="fed"):
        return dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            logP=3.0,
            pka=7.0,
            solubility_mg_mL=0.1,
            cl_L_per_h=2.0,
            vd_L=20.0,
            meal_state=meal_state,
        )

    def test_returns_bile_absorption_result(self):
        result = simulate_bile_acid_effect_on_absorption(**self._default_params())
        assert isinstance(result, BileAbsorptionResult)

    def test_fed_state_higher_bile_conc(self):
        fed = simulate_bile_acid_effect_on_absorption(**self._default_params("fed"))
        fasted = simulate_bile_acid_effect_on_absorption(**self._default_params("fasted"))
        assert fed.bile_conc_mM == 5.0
        assert fasted.bile_conc_mM == 1.0

    def test_fed_greater_enhancement_than_fasted(self):
        fed = simulate_bile_acid_effect_on_absorption(**self._default_params("fed"))
        fasted = simulate_bile_acid_effect_on_absorption(**self._default_params("fasted"))
        assert fed.solubility_enhancement >= fasted.solubility_enhancement

    def test_cmax_positive(self):
        result = simulate_bile_acid_effect_on_absorption(**self._default_params())
        assert result.cmax > 0.0

    def test_auc_positive(self):
        result = simulate_bile_acid_effect_on_absorption(**self._default_params())
        assert result.auc > 0.0

    def test_times_and_concentrations_same_length(self):
        result = simulate_bile_acid_effect_on_absorption(**self._default_params())
        assert len(result.times_h) == len(result.c_plasma)

    def test_times_start_at_zero(self):
        result = simulate_bile_acid_effect_on_absorption(**self._default_params())
        assert result.times_h[0] == pytest.approx(0.0, abs=0.01)

    def test_times_end_at_48(self):
        result = simulate_bile_acid_effect_on_absorption(**self._default_params())
        assert result.times_h[-1] == pytest.approx(48.0, abs=0.1)

    def test_invalid_dose(self):
        params = self._default_params()
        params["dose_mg"] = 0.0
        with pytest.raises(ValueError, match="dose_mg must be > 0"):
            simulate_bile_acid_effect_on_absorption(**params)

    def test_invalid_solubility(self):
        params = self._default_params()
        params["solubility_mg_mL"] = 0.0
        with pytest.raises(ValueError, match="solubility_mg_mL must be > 0"):
            simulate_bile_acid_effect_on_absorption(**params)

    def test_invalid_cl(self):
        params = self._default_params()
        params["cl_L_per_h"] = -1.0
        with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
            simulate_bile_acid_effect_on_absorption(**params)

    def test_invalid_vd(self):
        params = self._default_params()
        params["vd_L"] = 0.0
        with pytest.raises(ValueError, match="vd_L must be > 0"):
            simulate_bile_acid_effect_on_absorption(**params)

    def test_invalid_meal_state(self):
        params = self._default_params()
        params["meal_state"] = "snack"
        with pytest.raises(ValueError, match="meal_state must be 'fed' or 'fasted'"):
            simulate_bile_acid_effect_on_absorption(**params)

    def test_result_frozen(self):
        result = simulate_bile_acid_effect_on_absorption(**self._default_params())
        with pytest.raises(Exception):
            result.cmax = 999.0  # type: ignore[misc]

    def test_enhancement_at_least_one(self):
        """Solubility enhancement must always be >= 1."""
        result = simulate_bile_acid_effect_on_absorption(**self._default_params())
        assert result.solubility_enhancement >= 1.0
